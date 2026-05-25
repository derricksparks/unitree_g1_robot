"""HumanoidVerse G1-12DoF locomotion observation/config bridge."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from lucky_bridge.lucky_joint_map import LuckyJointMap


HUMANOIDVERSE_G1_12DOF_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
HUMANOIDVERSE_G1_29DOF_JOINTS: tuple[str, ...] = (
    *HUMANOIDVERSE_G1_12DOF_JOINTS,
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HUMANOIDVERSE_G1_12DOF_DEFAULTS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
}

HUMANOIDVERSE_G1_12DOF_ACTION_SCALE = 0.25


def make_humanoidverse_12dof_policy_config() -> dict[str, Any]:
    return {
        "joint_names": list(HUMANOIDVERSE_G1_12DOF_JOINTS),
        "default_joint_pos": dict(HUMANOIDVERSE_G1_12DOF_DEFAULTS),
        "action_scales": {jn: float(HUMANOIDVERSE_G1_12DOF_ACTION_SCALE) for jn in HUMANOIDVERSE_G1_12DOF_JOINTS},
    }


def make_humanoidverse_29dof_policy_config() -> dict[str, Any]:
    defaults = dict(HUMANOIDVERSE_G1_12DOF_DEFAULTS)
    defaults.update(
        {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
    )
    return {
        "joint_names": list(HUMANOIDVERSE_G1_29DOF_JOINTS),
        "default_joint_pos": defaults,
        "action_scales": {jn: float(HUMANOIDVERSE_G1_12DOF_ACTION_SCALE) for jn in HUMANOIDVERSE_G1_29DOF_JOINTS},
    }


def humanoidverse_12dof_obs_dim() -> int:
    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + cmd_lin(2) + cmd_yaw(1) + dof_pos(12) + dof_vel(12) + actions(12)
    return 48


def make_humanoidverse_12dof_actor_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    cmd: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    return make_humanoidverse_actor_obs(model, data, joint_map, cmd, last_action)


def make_humanoidverse_actor_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    cmd: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    cmd_arr = np.asarray(cmd, dtype=np.float32).reshape(3)
    act_arr = np.asarray(last_action, dtype=np.float32).reshape(joint_map.action_dim)
    quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    rot_flat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_flat, quat)
    rot_world_from_body = rot_flat.reshape(3, 3)
    rot_body_from_world = rot_world_from_body.T

    base_lin_vel_world = np.asarray(data.qvel[0:3], dtype=np.float64)
    base_ang_vel_world = np.asarray(data.qvel[3:6], dtype=np.float64)
    gravity_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    base_lin_vel_body = rot_body_from_world @ base_lin_vel_world
    base_ang_vel_body = rot_body_from_world @ base_ang_vel_world
    projected_gravity = rot_body_from_world @ gravity_world

    dof_pos = np.zeros(joint_map.action_dim, dtype=np.float32)
    dof_vel = np.zeros(joint_map.action_dim, dtype=np.float32)
    for i, name in enumerate(joint_map.joint_names):
        qpos_idx = joint_map.qpos_indices.get(name)
        qvel_idx = joint_map.qvel_indices.get(name)
        if qpos_idx is not None:
            dof_pos[i] = float(data.qpos[qpos_idx]) - float(joint_map.default_joint_pos[i])
        if qvel_idx is not None:
            dof_vel[i] = float(data.qvel[qvel_idx])

    obs = np.concatenate(
        [
            base_lin_vel_body.astype(np.float32),
            base_ang_vel_body.astype(np.float32),
            projected_gravity.astype(np.float32),
            cmd_arr[:2],
            np.asarray([cmd_arr[2]], dtype=np.float32),
            dof_pos,
            dof_vel,
            act_arr,
        ],
        axis=0,
    ).astype(np.float32)
    expected_dim = 12 + 3 * int(joint_map.action_dim)
    if obs.shape[0] != expected_dim:
        raise ValueError(f"HumanoidVerse actor obs dim mismatch: expected {expected_dim}, got {obs.shape[0]}")
    return obs

