#include "FSM/State_RLHybrid.h"

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <algorithm>

State_RLHybrid::State_RLHybrid(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    // Primary policy (typically locomotion).
    auto primary_policy_dir = param::parser_policy_dir(cfg["primary_policy_dir"].as<std::string>());
    env_primary = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(primary_policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    primary_alg = std::make_unique<isaaclab::OrtRunner>((primary_policy_dir / "exported" / "policy.onnx").string());

    // Secondary policy (typically manipulation).
    auto secondary_policy_dir = param::parser_policy_dir(cfg["secondary_policy_dir"].as<std::string>());
    env_secondary = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(secondary_policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    secondary_alg = std::make_unique<isaaclab::OrtRunner>((secondary_policy_dir / "exported" / "policy.onnx").string());

    // Which action indices to blend. Defaults to G1 arm joints (15..28) in MuJoCo actuator order.
    // 0..11 legs, 12..14 waist, 15..21 left arm, 22..28 right arm.
    if (cfg["blend_joint_indices"]) {
        blend_joint_indices = cfg["blend_joint_indices"].as<std::vector<int>>();
    } else {
        blend_joint_indices.clear();
        for (int i = 15; i <= 28; ++i) blend_joint_indices.push_back(i);
    }

    blend_alpha = cfg["blend_alpha"] ? cfg["blend_alpha"].as<float>() : 0.0f;
    blend_alpha = std::clamp(blend_alpha, 0.0f, 1.0f);

    blended_processed_action.assign(29, 0.0f);

    // Safety: return to Passive on bad orientation.
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool { return isaaclab::mdp::bad_orientation(env_primary.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLHybrid::enter()
{
    // Gains from the primary deploy.yaml (locomotion tuned).
    for (int i = 0; i < env_primary->robot->data.joint_stiffness.size(); ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = env_primary->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env_primary->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env_primary->robot->update();

    policy_thread_running = true;
    policy_thread = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env_primary->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        auto sleepTill = clock::now() + dt;
        env_primary->reset();
        env_secondary->reset();

        while (policy_thread_running) {
            // Shared articulation: update once.
            env_primary->robot->update();

            auto obs_primary = env_primary->observation_manager->compute();
            auto obs_secondary = env_secondary->observation_manager->compute();

            static bool printed = false;
            if (!printed) {
                if (obs_primary.find("obs") != obs_primary.end()) {
                    std::cout << "[RLHybrid] primary obs len=" << obs_primary["obs"].size() << std::endl;
                }
                if (obs_secondary.find("obs") != obs_secondary.end()) {
                    std::cout << "[RLHybrid] secondary obs len=" << obs_secondary["obs"].size() << std::endl;
                }
                printed = true;
            }

            auto a_primary = primary_alg->act(obs_primary);
            auto a_secondary = secondary_alg->act(obs_secondary);

            env_primary->action_manager->process_action(a_primary);
            env_secondary->action_manager->process_action(a_secondary);

            auto p_primary = env_primary->action_manager->processed_actions();
            auto p_secondary = env_secondary->action_manager->processed_actions();
            if (p_primary.size() != p_secondary.size()) {
                throw std::runtime_error("Hybrid policies have different processed action dimensions.");
            }

            // Blend processed joint targets on selected indices.
            auto blended = p_primary;
            for (int idx : blend_joint_indices) {
                if (idx < 0 || static_cast<size_t>(idx) >= blended.size()) continue;
                blended[idx] = (1.0f - blend_alpha) * p_primary[idx] + blend_alpha * p_secondary[idx];
            }

            {
                std::lock_guard<std::mutex> lock(blended_mtx);
                blended_processed_action = std::move(blended);
            }

            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}

void State_RLHybrid::run()
{
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> lock(blended_mtx);
        action = blended_processed_action;
    }
    for (int i = 0; i < env_primary->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env_primary->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_RLHybrid::exit()
{
    policy_thread_running = false;
    if (policy_thread.joinable()) {
        policy_thread.join();
    }
}

