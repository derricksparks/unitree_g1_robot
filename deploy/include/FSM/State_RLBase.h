// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <unordered_set>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        // If controlled_joint_ids is set, only update gains for those joints
        // (keeps locomotion gains stable when running an arm-only policy).
        if (!controlled_joint_ids.empty()) {
            for (int jid : controlled_joint_ids) {
                if (jid < 0 || jid >= static_cast<int>(env->robot->data.joint_stiffness.size())) continue;
                lowcmd->msg_.motor_cmd()[jid].kp() = env->robot->data.joint_stiffness[jid];
                lowcmd->msg_.motor_cmd()[jid].kd() = env->robot->data.joint_damping[jid];
                lowcmd->msg_.motor_cmd()[jid].dq() = 0;
                lowcmd->msg_.motor_cmd()[jid].tau() = 0;
            }
        } else {
            for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
                lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
                lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
                lowcmd->msg_.motor_cmd()[i].dq() = 0;
                lowcmd->msg_.motor_cmd()[i].tau() = 0;
            }
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unordered_set<int> controlled_joint_ids;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
