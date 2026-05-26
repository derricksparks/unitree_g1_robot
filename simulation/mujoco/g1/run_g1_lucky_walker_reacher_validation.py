#!/usr/bin/env python3
"""Validate Lucky walker + right reacher without arm/walker motor conflicts."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

from lucky_bridge.lucky_joint_map import (  # noqa: E402
    LuckyJointMap,
    action_to_joint_targets,
    build_lucky_joint_map,
)
from lucky_bridge.lucky_paths import (  # noqa: E402
    LUCKY_MODEL_CONFIG,
    LUCKY_RIGHT_REACHER_ONNX,
    LUCKY_SCENE_XML,
    LUCKY_WALKER_ONNX,
    missing_lucky_assets,
)
from lucky_bridge.lucky_policy_loader import (  # noqa: E402
    LuckyPolicyLoadError,
    load_lucky_config,
    load_right_reacher_policy,
    load_walker_policy,
)
from run_g1_lucky_locomotion_validation import (  # noqa: E402
    PELVIS_DROP_M,
    ROLL_FALL_RAD,
    TORSO_FALL_RAD,
    VISIBLE_FOOT_LIFT_M,
    VISIBLE_FORWARD_SWING_M,
    _disable_lucky_front_obstacles,
    _floor_contact_detected,
    _foot_center_xyz,
    _make_observation,
    _obstacle_contact_count,
    _quat_apply_inverse,
)
from run_g1_posture_hold import pelvis_roll_pitch_deg  # noqa: E402


RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_TOKEN = "left_"
RIGHT_ARM_MOTION_THRESHOLD_RAD = 0.025


def _site_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))


def _palm_pos_in_pelvis(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sid = _site_id(model, "right_palm")
    if sid < 0:
        return np.zeros(3, dtype=np.float32)
    palm_world = np.asarray(data.site_xpos[sid], dtype=float)
    pelvis_pos = np.asarray(data.qpos[:3], dtype=float)
    pelvis_quat = np.asarray(data.qpos[3:7], dtype=float)
    return _quat_apply_inverse(pelvis_quat, palm_world - pelvis_pos).astype(np.float32)


def _palm_orientation_in_pelvis(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sid = _site_id(model, "right_palm")
    if sid < 0:
        return np.zeros(3, dtype=np.float32)
    mat = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
    palm_q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(palm_q, mat.reshape(-1))
    pelvis_q = np.asarray(data.qpos[3:7], dtype=float)
    pinv = np.array([pelvis_q[0], -pelvis_q[1], -pelvis_q[2], -pelvis_q[3]], dtype=float)
    w1, x1, y1, z1 = pinv
    w2, x2, y2, z2 = palm_q
    rel = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )
    w, x, y, z = rel
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _right_arm_indices(joint_map: LuckyJointMap) -> tuple[int, ...]:
    return tuple(joint_map.joint_names.index(name) for name in RIGHT_ARM_JOINT_NAMES)


def _right_arm_positions(data: mujoco.MjData, joint_map: LuckyJointMap) -> np.ndarray:
    vals: list[float] = []
    for idx in _right_arm_indices(joint_map):
        name = joint_map.joint_names[idx]
        qadr = joint_map.qpos_indices[name]
        vals.append(float(data.qpos[qadr]) - float(joint_map.default_joint_pos[idx]))
    return np.asarray(vals, dtype=np.float32)


def _right_arm_velocities(data: mujoco.MjData, joint_map: LuckyJointMap) -> np.ndarray:
    vals: list[float] = []
    for idx in _right_arm_indices(joint_map):
        name = joint_map.joint_names[idx]
        vadr = joint_map.qvel_indices[name]
        vals.append(float(data.qvel[vadr]))
    return np.asarray(vals, dtype=np.float32)


def _right_arm_default(joint_map: LuckyJointMap) -> np.ndarray:
    return np.asarray([joint_map.default_joint_pos[idx] for idx in _right_arm_indices(joint_map)], dtype=np.float32)


def _right_arm_scales(joint_map: LuckyJointMap) -> np.ndarray:
    return np.asarray([joint_map.action_scales[idx] for idx in _right_arm_indices(joint_map)], dtype=np.float32)


def _make_reacher_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    last_arm_action: np.ndarray,
    reach_target: np.ndarray,
    reach_orientation: np.ndarray,
) -> np.ndarray:
    proj_gravity = _quat_apply_inverse(np.asarray(data.qpos[3:7], dtype=float), np.array([0.0, 0.0, -1.0]))
    return np.concatenate(
        [
            np.asarray(reach_target, dtype=np.float32),
            np.asarray(reach_orientation, dtype=np.float32),
            _palm_pos_in_pelvis(model, data),
            _palm_orientation_in_pelvis(model, data),
            _right_arm_positions(data, joint_map),
            _right_arm_velocities(data, joint_map),
            np.asarray(last_arm_action, dtype=np.float32),
            proj_gravity.astype(np.float32),
        ]
    ).astype(np.float32)


def _apply_walker_reacher_targets(
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    walker_targets: np.ndarray,
    right_arm_targets: np.ndarray,
) -> None:
    for idx in joint_map.controlled_indices:
        name = joint_map.joint_names[idx]
        aid = joint_map.actuator_ids.get(name, -1)
        if aid >= 0:
            data.ctrl[int(aid)] = float(walker_targets[idx])
    right_arm_idx = _right_arm_indices(joint_map)
    right_set = set(right_arm_idx)
    for idx in joint_map.arm_indices:
        name = joint_map.joint_names[idx]
        aid = joint_map.actuator_ids.get(name, -1)
        if aid < 0:
            continue
        if idx in right_set:
            arm_i = right_arm_idx.index(idx)
            data.ctrl[int(aid)] = float(right_arm_targets[arm_i])
        elif name.startswith(LEFT_ARM_TOKEN):
            data.ctrl[int(aid)] = float(joint_map.default_joint_pos[idx])


def run_g1_lucky_walker_reacher_validation(
    *,
    headless: bool = False,
    timeout: float = 10.0,
    viewer_speed: float = 1.0,
    cmd_x: float = 1.0,
    cmd_y: float = 0.0,
    cmd_yaw: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    missing = missing_lucky_assets()
    if missing:
        return {
            "walker_policy_loaded": False,
            "right_reacher_policy_loaded": False,
            "success": False,
            "failure_reason": "missing_lucky_assets:" + ",".join(missing),
        }

    config = load_lucky_config(LUCKY_MODEL_CONFIG)
    model = mujoco.MjModel.from_xml_path(str(LUCKY_SCENE_XML))
    obstacle_info = _disable_lucky_front_obstacles(model)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(config["joint_names"]):
        data.qpos[7 + i] = float(config["default_joint_pos"].get(name, 0.0))
    mujoco.mj_forward(model, data)

    try:
        walker = load_walker_policy(LUCKY_WALKER_ONNX)
        right_reacher = load_right_reacher_policy(LUCKY_RIGHT_REACHER_ONNX)
    except LuckyPolicyLoadError as exc:
        return {
            "walker_policy_loaded": False,
            "right_reacher_policy_loaded": False,
            "success": False,
            "failure_reason": str(exc),
        }

    joint_map = build_lucky_joint_map(model, config, control_waist=True)
    right_arm_idx = _right_arm_indices(joint_map)
    right_arm_default = _right_arm_default(joint_map)
    right_arm_scales = _right_arm_scales(joint_map)
    left_start = _foot_center_xyz(model, data, "left")
    right_start = _foot_center_xyz(model, data, "right")
    initial_pelvis = np.asarray(data.qpos[:3], dtype=float).copy()
    initial_arm = np.asarray([data.qpos[joint_map.qpos_indices[joint_map.joint_names[idx]]] for idx in right_arm_idx], dtype=float)

    cmd = np.array([float(cmd_x), float(cmd_y), float(cmd_yaw)], dtype=np.float32)
    last_action = np.zeros(joint_map.action_dim, dtype=np.float32)
    last_arm_action = np.zeros(len(right_arm_idx), dtype=np.float32)
    walker_targets = joint_map.default_joint_pos.copy()
    right_arm_targets = right_arm_default.copy()
    reach_orientation = np.zeros(3, dtype=np.float32)

    control_step = 0
    left_step_start = left_start.copy()
    right_step_start = right_start.copy()
    left_prev_lift = False
    right_prev_lift = False
    left_peak_lift = 0.0
    right_peak_lift = 0.0
    left_peak_forward = 0.0
    right_peak_forward = 0.0
    max_left_lift = 0.0
    max_right_lift = 0.0
    actual_visible_step_count = 0
    max_torso_pitch_rad = 0.0
    max_torso_roll_rad = 0.0
    max_right_arm_motion_rad = 0.0
    robot_fell = False
    floor_contact_seen = False
    max_obstacle_contacts = 0
    start_wall = time.time()

    def step_once() -> bool:
        nonlocal control_step, last_action, last_arm_action, walker_targets, right_arm_targets
        nonlocal left_step_start, right_step_start, left_prev_lift, right_prev_lift
        nonlocal left_peak_lift, right_peak_lift, left_peak_forward, right_peak_forward
        nonlocal max_left_lift, max_right_lift, actual_visible_step_count
        nonlocal max_torso_pitch_rad, max_torso_roll_rad, max_right_arm_motion_rad
        nonlocal robot_fell, floor_contact_seen, max_obstacle_contacts

        if float(data.time) >= float(timeout):
            return False
        if control_step % 4 == 0:
            obs = _make_observation(data, joint_map, last_action, cmd)
            last_action = np.asarray(walker(obs), dtype=np.float32).reshape(joint_map.action_dim)
            walker_targets = action_to_joint_targets(last_action, joint_map)

            t = float(data.time)
            reach_target = np.array(
                [
                    0.34 + 0.10 * np.sin(0.9 * t),
                    -0.26 + 0.05 * np.sin(0.6 * t),
                    0.20 + 0.07 * np.cos(0.7 * t),
                ],
                dtype=np.float32,
            )
            reach_obs = _make_reacher_observation(
                model,
                data,
                joint_map,
                last_arm_action,
                reach_target,
                reach_orientation,
            )
            last_arm_action = np.asarray(right_reacher(reach_obs), dtype=np.float32).reshape(len(right_arm_idx))
            arm_target = right_arm_default + last_arm_action * right_arm_scales
            max_delta = 0.012
            right_arm_targets = right_arm_targets + np.clip(arm_target - right_arm_targets, -max_delta, max_delta)

        _apply_walker_reacher_targets(data, joint_map, walker_targets, right_arm_targets)
        mujoco.mj_step(model, data)
        control_step += 1

        for side, cur, start, prev_lift in (
            ("left", _foot_center_xyz(model, data, "left"), left_step_start, left_prev_lift),
            ("right", _foot_center_xyz(model, data, "right"), right_step_start, right_prev_lift),
        ):
            lift = max(0.0, float(cur[2] - start[2]))
            fwd = max(0.0, float(cur[0] - start[0]))
            is_lift = lift >= VISIBLE_FOOT_LIFT_M
            if side == "left":
                left_peak_lift = max(left_peak_lift, lift)
                left_peak_forward = max(left_peak_forward, fwd)
                max_left_lift = max(max_left_lift, lift)
                if prev_lift and not is_lift and left_peak_lift >= VISIBLE_FOOT_LIFT_M and left_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    left_step_start = cur.copy()
                    left_peak_lift = 0.0
                    left_peak_forward = 0.0
                left_prev_lift = is_lift
            else:
                right_peak_lift = max(right_peak_lift, lift)
                right_peak_forward = max(right_peak_forward, fwd)
                max_right_lift = max(max_right_lift, lift)
                if prev_lift and not is_lift and right_peak_lift >= VISIBLE_FOOT_LIFT_M and right_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    right_step_start = cur.copy()
                    right_peak_lift = 0.0
                    right_peak_forward = 0.0
                right_prev_lift = is_lift

        current_arm = np.asarray([data.qpos[joint_map.qpos_indices[joint_map.joint_names[idx]]] for idx in right_arm_idx], dtype=float)
        max_right_arm_motion_rad = max(max_right_arm_motion_rad, float(np.max(np.abs(current_arm - initial_arm))))
        roll, pitch = pelvis_roll_pitch_deg(data, model)
        if pitch is not None:
            max_torso_pitch_rad = max(max_torso_pitch_rad, abs(float(pitch)))
        if roll is not None:
            max_torso_roll_rad = max(max_torso_roll_rad, abs(float(roll)))
        pelvis_drop = float(initial_pelvis[2]) - float(data.qpos[2])
        if max_torso_pitch_rad > TORSO_FALL_RAD or max_torso_roll_rad > ROLL_FALL_RAD or pelvis_drop > PELVIS_DROP_M:
            robot_fell = True
        floor_contact_seen = floor_contact_seen or _floor_contact_detected(model, data)
        max_obstacle_contacts = max(max_obstacle_contacts, _obstacle_contact_count(model, data))
        return True

    if headless:
        while step_once():
            pass
    else:
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running() and step_once():
                viewer.sync()
                time.sleep(max(0.001, float(model.opt.timestep) / max(float(viewer_speed), 1e-3)))

    pelvis_forward = float(data.qpos[0] - initial_pelvis[0])
    avg_speed = pelvis_forward / max(float(data.time), 1e-6)
    walker_controls_legs_waist_only = set(joint_map.controlled_indices).isdisjoint(set(joint_map.arm_indices))
    reacher_controls_right_arm_only = set(right_arm_idx).issubset(set(joint_map.arm_indices))
    walking_preserved = bool(actual_visible_step_count >= 1 and avg_speed > 0.05 and floor_contact_seen)
    right_arm_motion_detected = bool(max_right_arm_motion_rad >= RIGHT_ARM_MOTION_THRESHOLD_RAD)
    failure_reasons: list[str] = []
    if walker.output_dim != joint_map.action_dim or not joint_map.mapping_ok:
        failure_reasons.append("walker_joint_mapping_invalid")
    if right_reacher.output_dim != len(right_arm_idx):
        failure_reasons.append("right_reacher_output_dim_invalid")
    if not walker_controls_legs_waist_only:
        failure_reasons.append("walker_arm_conflict")
    if not reacher_controls_right_arm_only:
        failure_reasons.append("reacher_joint_ownership_invalid")
    if robot_fell:
        failure_reasons.append("robot_fell")
    if not walking_preserved:
        failure_reasons.append("walking_not_preserved")
    if not right_arm_motion_detected:
        failure_reasons.append("right_arm_motion_not_detected")
    if max_obstacle_contacts > 0:
        failure_reasons.append("obstacle_contact_detected")

    success = not failure_reasons
    out: dict[str, Any] = {
        "walker_policy_loaded": True,
        "right_reacher_policy_loaded": True,
        "walker_policy_path": str(LUCKY_WALKER_ONNX),
        "right_reacher_policy_path": str(LUCKY_RIGHT_REACHER_ONNX),
        "walker_observation_dim": int(walker.input_dim),
        "walker_action_dim": int(walker.output_dim),
        "right_reacher_observation_dim": int(right_reacher.input_dim),
        "right_reacher_action_dim": int(right_reacher.output_dim),
        "walker_controls_legs_waist_only": bool(walker_controls_legs_waist_only),
        "reacher_controls_right_arm_only": bool(reacher_controls_right_arm_only),
        "walker_arm_outputs_masked": True,
        "right_arm_reacher_active": True,
        "hand_control_separate": True,
        "robot_fell": bool(robot_fell),
        "walking_preserved": bool(walking_preserved),
        "right_arm_motion_detected": bool(right_arm_motion_detected),
        "actual_visible_step_count": int(actual_visible_step_count),
        "pelvis_forward_progress_m": float(pelvis_forward),
        "average_forward_speed_mps": float(avg_speed),
        "actual_floor_contact_detected": bool(floor_contact_seen),
        "max_torso_pitch_rad": float(max_torso_pitch_rad),
        "max_torso_roll_rad": float(max_torso_roll_rad),
        "left_actual_foot_lift_m": float(max_left_lift),
        "right_actual_foot_lift_m": float(max_right_lift),
        "max_right_arm_motion_rad": float(max_right_arm_motion_rad),
        "front_obstacles_disabled": bool(obstacle_info["front_obstacles_disabled"]),
        "obstacle_contact_count": int(max_obstacle_contacts),
        "success": bool(success),
        "failure_reason": None if success else "; ".join(failure_reasons),
        "wall_time_s": float(time.time() - start_wall),
        "sim_time_s": float(data.time),
    }
    if verbose:
        print("----- lucky_walker_reacher_validation -----")
        for key, value in out.items():
            print(f"{key}: {value}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Lucky walker + right reacher in MuJoCo.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--viewer-speed", type=float, default=1.0)
    ap.add_argument("--cmd-x", type=float, default=1.0)
    ap.add_argument("--cmd-y", type=float, default=0.0)
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ns = ap.parse_args()
    run_g1_lucky_walker_reacher_validation(
        headless=bool(ns.headless),
        timeout=float(ns.timeout),
        viewer_speed=float(ns.viewer_speed),
        cmd_x=float(ns.cmd_x),
        cmd_y=float(ns.cmd_y),
        cmd_yaw=float(ns.cmd_yaw),
    )


if __name__ == "__main__":
    main()
