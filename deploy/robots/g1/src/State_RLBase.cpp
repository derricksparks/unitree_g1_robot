#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <algorithm>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

REGISTER_OBSERVATION(box_position_b)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>{0.0f, 0.0f, 0.0f}; }
    return cfg["box_position_b"].as<std::vector<float>>(std::vector<float>{0.65f, 0.0f, 0.78f});
}

REGISTER_OBSERVATION(shelf_position_b)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>{0.0f, 0.0f, 0.0f}; }
    return cfg["shelf_position_b"].as<std::vector<float>>(std::vector<float>{1.40f, 0.0f, 0.92f});
}

REGISTER_OBSERVATION(box_to_shelf_b)
{
    auto box = box_position_b(env, params);
    auto shelf = shelf_position_b(env, params);
    return std::vector<float>{
        shelf[0] - box[0],
        shelf[1] - box[1],
        shelf[2] - box[2],
    };
}

REGISTER_OBSERVATION(hand_to_box_b)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>(6, 0.0f); }
    return cfg["hand_to_box_b"].as<std::vector<float>>(std::vector<float>(6, 0.0f));
}

REGISTER_OBSERVATION(box_velocity_b)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>{0.0f, 0.0f, 0.0f}; }
    return cfg["box_velocity_b"].as<std::vector<float>>(std::vector<float>{0.0f, 0.0f, 0.0f});
}

REGISTER_OBSERVATION(box_lifted)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>{0.0f}; }
    return std::vector<float>{cfg["box_lifted"].as<float>(0.0f)};
}

REGISTER_OBSERVATION(task_stage)
{
    const auto cfg = env->cfg["commands"]["box_transport"];
    if(!cfg) { return std::vector<float>{1.0f}; }
    return std::vector<float>{cfg["task_stage"].as<float>(1.0f)};
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    const auto safety = env->cfg["safety"];
    if(safety) {
        const auto max_delta = safety["max_target_delta"].as<float>(0.0f);
        const auto min_q = safety["joint_target_min"].as<std::vector<float>>(std::vector<float>{});
        const auto max_q = safety["joint_target_max"].as<std::vector<float>>(std::vector<float>{});
        for(size_t i = 0; i < action.size(); ++i) {
            const int motor_id = static_cast<int>(env->robot->data.joint_ids_map[i]);
            if(max_delta > 0.0f) {
                const float current_q = lowstate->msg_.motor_state()[motor_id].q();
                action[i] = std::clamp(action[i], current_q - max_delta, current_q + max_delta);
            }
            if(i < min_q.size() && i < max_q.size()) {
                action[i] = std::clamp(action[i], min_q[i], max_q[i]);
            }
        }
    }
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}