// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/envs/manager_based_rl_env.h"

class State_RLHybrid : public FSMState
{
public:
    State_RLHybrid(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_primary;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_secondary;
    std::unique_ptr<isaaclab::OrtRunner> primary_alg;
    std::unique_ptr<isaaclab::OrtRunner> secondary_alg;

    std::vector<int> blend_joint_indices;
    float blend_alpha = 0.0f;

    std::mutex blended_mtx;
    std::vector<float> blended_processed_action;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLHybrid)

