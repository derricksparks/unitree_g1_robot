#!/usr/bin/env python3
"""Validate Lucky walking with arms locked in a compact no-payload carry pose."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

from g1_dex3_finger_control import Dex3FingerController  # noqa: E402
from lucky_bridge.lucky_joint_map import action_to_joint_targets, build_lucky_joint_map  # noqa: E402
from lucky_bridge.lucky_paths import LUCKY_MODEL_CONFIG, LUCKY_SCENE_XML, LUCKY_WALKER_ONNX, missing_lucky_assets  # noqa: E402
from lucky_bridge.lucky_policy_loader import load_lucky_config, load_walker_policy  # noqa: E402
from run_g1_lucky_locomotion_validation import (  # noqa: E402
    PELVIS_DROP_M,
    ROLL_FALL_RAD,
    TORSO_FALL_RAD,
    VISIBLE_FOOT_LIFT_M,
    VISIBLE_FORWARD_SWING_M,
    _apply_targets,
    _disable_lucky_front_obstacles,
    _floor_contact_detected,
    _foot_center_xyz,
    _make_observation,
)
from run_g1_lucky_pick_transport_state_machine import _actuator_id, _site_pos, _solve_arm_targets  # noqa: E402
from run_g1_posture_hold import pelvis_roll_pitch_deg  # noqa: E402


COMPACT_CARRY_ARM_JOINTS = (
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


def _compact_carry_candidates(defaults: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
    """Postures with elbows bent/tucked and palms close to the torso."""
    base = {name: float(defaults.get(name, 0.0)) for name in COMPACT_CARRY_ARM_JOINTS}
    return [
        (
            "default_walker_arm_carry",
            {
                **base,
                "left_elbow_joint": 0.95,
                "right_elbow_joint": 0.95,
                "left_shoulder_roll_joint": 0.16,
                "right_shoulder_roll_joint": -0.16,
            },
        ),
        (
            "tight_chest_carry",
            {
                **base,
                "left_shoulder_pitch_joint": 0.05,
                "right_shoulder_pitch_joint": 0.05,
                "left_shoulder_roll_joint": 0.12,
                "right_shoulder_roll_joint": -0.12,
                "left_shoulder_yaw_joint": -0.25,
                "right_shoulder_yaw_joint": 0.25,
                "left_elbow_joint": 1.25,
                "right_elbow_joint": 1.25,
                "left_wrist_pitch_joint": -0.08,
                "right_wrist_pitch_joint": -0.08,
            },
        ),
        (
            "elbows_tucked_back_carry",
            {
                **base,
                "left_shoulder_pitch_joint": -0.05,
                "right_shoulder_pitch_joint": -0.05,
                "left_shoulder_roll_joint": 0.10,
                "right_shoulder_roll_joint": -0.10,
                "left_shoulder_yaw_joint": -0.40,
                "right_shoulder_yaw_joint": 0.40,
                "left_elbow_joint": 1.35,
                "right_elbow_joint": 1.35,
                "left_wrist_pitch_joint": -0.10,
                "right_wrist_pitch_joint": -0.10,
            },
        ),
        (
            "front_center_chest_carry",
            {
                **base,
                "left_shoulder_pitch_joint": 0.60,
                "right_shoulder_pitch_joint": 0.60,
                "left_shoulder_roll_joint": 0.12,
                "right_shoulder_roll_joint": -0.12,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": -1.0472,
                "right_elbow_joint": -1.0472,
                "left_wrist_pitch_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
            },
        ),
        (
            "raised_chest_elbows_back_carry",
            {
                **base,
                "left_shoulder_pitch_joint": 1.20,
                "right_shoulder_pitch_joint": 1.20,
                "left_shoulder_roll_joint": 0.30,
                "right_shoulder_roll_joint": -0.30,
                "left_shoulder_yaw_joint": -0.40,
                "right_shoulder_yaw_joint": 0.40,
                "left_elbow_joint": 2.20,
                "right_elbow_joint": 2.20,
                "left_wrist_pitch_joint": -0.10,
                "right_wrist_pitch_joint": -0.10,
            },
        ),
        (
            "high_front_chest_carry",
            {
                **base,
                "left_shoulder_pitch_joint": -0.80,
                "right_shoulder_pitch_joint": -0.80,
                "left_shoulder_roll_joint": 0.30,
                "right_shoulder_roll_joint": -0.30,
                "left_shoulder_yaw_joint": 0.40,
                "right_shoulder_yaw_joint": -0.40,
                "left_elbow_joint": 0.60,
                "right_elbow_joint": 0.60,
                "left_wrist_pitch_joint": -0.10,
                "right_wrist_pitch_joint": -0.10,
            },
        ),
    ]


def _root_yaw_rad(data: mujoco.MjData) -> float:
    w, x, y, z = [float(v) for v in data.qpos[3:7]]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _apply_arm_targets(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, value in targets.items():
        aid = _actuator_id(model, name)
        if aid >= 0:
            data.ctrl[aid] = float(value)


def _apply_finger_targets(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, value in targets.items():
        aid = _actuator_id(model, name)
        if aid >= 0:
            data.ctrl[aid] = float(value)


def _set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, value in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def _configure_attached_carry_box(
    model: mujoco.MjModel,
    *,
    mass_kg: float,
    size_xyz: tuple[float, float, float],
) -> bool:
    """Reuse Lucky's red block as a light wide box carried between both palms."""
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    main_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_cylinder")
    if box_bid < 0 or main_gid < 0:
        return False
    half = 0.5 * np.asarray(size_xyz, dtype=float).reshape(3)
    model.geom_type[main_gid] = int(mujoco.mjtGeom.mjGEOM_BOX)
    model.geom_size[main_gid, :3] = half
    model.geom_pos[main_gid, :3] = 0.0
    model.geom_contype[main_gid] = 1
    model.geom_conaffinity[main_gid] = 1
    model.geom_friction[main_gid, :3] = [3.0, 0.2, 0.02]
    model.body_mass[box_bid] = max(float(mass_kg), 1e-5)
    model.body_inertia[box_bid, :3] = np.maximum(model.body_inertia[box_bid, :3], 1e-5)
    for cap_name in ("red_cap_top", "red_cap_bot"):
        cap_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cap_name)
        if cap_gid >= 0:
            model.geom_contype[cap_gid] = 0
            model.geom_conaffinity[cap_gid] = 0
            model.geom_size[cap_gid, :3] = 1e-4
    return True


def _attach_carry_box_between_palms(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Place the free box body at the palm midpoint and zero its free-joint velocity."""
    box_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    left_palm = _site_pos(model, data, "left_palm")
    right_palm = _site_pos(model, data, "right_palm")
    center = 0.5 * (left_palm + right_palm)
    if box_jid >= 0:
        qadr = int(model.jnt_qposadr[box_jid])
        data.qpos[qadr : qadr + 7] = [float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0]
        vadr = int(model.jnt_dofadr[box_jid])
        data.qvel[vadr : vadr + 6] = 0.0
    return center


def _hand_box_contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    box_geom_ids = {
        gid
        for name in ("red_cylinder", "red_cap_top", "red_cap_bot")
        if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
    }
    contact_count = 0
    max_penetration = 0.0
    min_signed_distance = float("inf")
    for cid in range(data.ncon):
        con = data.contact[cid]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        if g1 not in box_geom_ids and g2 not in box_geom_ids:
            continue
        other_gid = g2 if g1 in box_geom_ids else g1
        other_body = int(model.geom_bodyid[other_gid])
        other_body_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or "")
        if "hand" not in other_body_name and "wrist" not in other_body_name:
            continue
        contact_count += 1
        signed_distance = float(con.dist)
        min_signed_distance = min(min_signed_distance, signed_distance)
        max_penetration = max(max_penetration, max(0.0, -signed_distance))
    return {
        "hand_box_contact_count": int(contact_count),
        "max_hand_box_penetration_m": float(max_penetration),
        "min_hand_box_signed_distance_m": float(min_signed_distance if np.isfinite(min_signed_distance) else 0.0),
        "hand_box_collision_active": bool(contact_count > 0),
    }


def _posture_metrics(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> dict[str, Any]:
    torso = np.asarray(data.qpos[:3], dtype=float)
    left_palm = _site_pos(model, data, "left_palm")
    right_palm = _site_pos(model, data, "right_palm")
    left_dist = float(np.linalg.norm(left_palm[:2] - torso[:2]))
    right_dist = float(np.linalg.norm(right_palm[:2] - torso[:2]))
    avg_dist = 0.5 * (left_dist + right_dist)
    avg_forward = float(0.5 * ((left_palm[0] - torso[0]) + (right_palm[0] - torso[0])))
    palm_lateral_center = float(0.5 * ((left_palm[1] - torso[1]) + (right_palm[1] - torso[1])))
    palm_pair_width = float(abs(left_palm[1] - right_palm[1]))
    avg_height = float(0.5 * (left_palm[2] + right_palm[2]))
    avg_height_above_pelvis = float(avg_height - torso[2])
    left_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    palms_parallel_score = 0.0
    palms_front_score = 0.0
    if left_sid >= 0 and right_sid >= 0:
        left_mat = np.asarray(data.site_xmat[left_sid], dtype=float).reshape(3, 3)
        right_mat = np.asarray(data.site_xmat[right_sid], dtype=float).reshape(3, 3)
        palms_parallel_score = max(abs(float(np.dot(left_mat[:, i], right_mat[:, i]))) for i in range(3))
        palms_front_score = 0.5 * (
            max(abs(float(np.dot(left_mat[:, i], np.array([1.0, 0.0, 0.0])))) for i in range(3))
            + max(abs(float(np.dot(right_mat[:, i], np.array([1.0, 0.0, 0.0])))) for i in range(3))
        )
    return {
        "left_palm_torso_distance_m": left_dist,
        "right_palm_torso_distance_m": right_dist,
        "avg_palm_torso_distance_m": avg_dist,
        "avg_palm_forward_offset_m": avg_forward,
        "palm_lateral_center_offset_m": palm_lateral_center,
        "palm_pair_width_m": palm_pair_width,
        "left_palm_height_m": float(left_palm[2]),
        "right_palm_height_m": float(right_palm[2]),
        "avg_palm_height_m": avg_height,
        "avg_palm_height_above_pelvis_m": avg_height_above_pelvis,
        "palms_parallel_score": float(palms_parallel_score),
        "palms_front_facing_score": float(palms_front_score),
        "left_elbow_bent_rad": float(targets.get("left_elbow_joint", 0.0)),
        "right_elbow_bent_rad": float(targets.get("right_elbow_joint", 0.0)),
        "elbows_tucked_back_commanded": bool(
            targets.get("left_shoulder_yaw_joint", 0.0) < 0.0 and targets.get("right_shoulder_yaw_joint", 0.0) > 0.0
        ),
        "palms_at_chest_height": bool(avg_height_above_pelvis >= 0.18),
        "palms_front_centered": bool(0.08 <= avg_forward <= 0.28 and abs(palm_lateral_center) <= 0.06 and palm_pair_width <= 0.45),
        "palms_parallel": bool(palms_parallel_score >= 0.95),
        "palms_face_front": bool(palms_front_score >= 0.89),
        "hands_not_folded_inward": bool(abs(float(targets.get("left_shoulder_yaw_joint", 0.0))) <= 0.20 and abs(float(targets.get("right_shoulder_yaw_joint", 0.0))) <= 0.20),
        "compact_carry_posture_valid": bool(
            avg_dist <= 0.35
            and 0.08 <= avg_forward <= 0.28
            and abs(palm_lateral_center) <= 0.06
            and palm_pair_width <= 0.45
            and avg_height_above_pelvis >= 0.18
            and palms_parallel_score >= 0.95
            and palms_front_score >= 0.89
        ),
    }


def _select_compact_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    defaults: dict[str, float],
) -> tuple[str, dict[str, float], dict[str, Any], int]:
    best_name = "none"
    best_targets: dict[str, float] = {}
    best_metrics: dict[str, Any] = {}
    best_score = float("inf")
    snapshot = np.asarray(data.qpos, dtype=float).copy()
    candidates = _compact_carry_candidates(defaults)
    pelvis = np.asarray(data.qpos[:3], dtype=float)
    for z_offset in (0.20, 0.24, 0.28, 0.32):
        for forward in (0.06, 0.10, 0.14):
            for side in (0.10, 0.13, 0.16):
                left_target = pelvis + np.array([forward, side, z_offset], dtype=float)
                right_target = pelvis + np.array([forward, -side, z_offset], dtype=float)
                ik_targets, left_err, right_err = _solve_arm_targets(model, data, left_target=left_target, right_target=right_target)
                if max(float(left_err), float(right_err)) <= 0.08:
                    ik_targets = {
                        **ik_targets,
                        "left_elbow_joint": max(float(ik_targets.get("left_elbow_joint", 0.0)), 0.95),
                        "right_elbow_joint": max(float(ik_targets.get("right_elbow_joint", 0.0)), 0.95),
                    }
                    candidates.append((f"ik_chest_carry_z{z_offset:.2f}_f{forward:.2f}_s{side:.2f}", ik_targets))
    for name, targets in candidates:
        data.qpos[:] = snapshot
        _set_arm_qpos(model, data, targets)
        mujoco.mj_forward(model, data)
        metrics = _posture_metrics(model, data, targets)
        height_error = max(0.0, 0.24 - float(metrics["avg_palm_height_above_pelvis_m"]))
        front_error = abs(float(metrics["avg_palm_forward_offset_m"]) - 0.16)
        center_error = abs(float(metrics["palm_lateral_center_offset_m"]))
        width_error = abs(float(metrics["palm_pair_width_m"]) - 0.24)
        score = float(metrics["avg_palm_torso_distance_m"]) + 2.0 * front_error + 2.0 * center_error + 0.5 * width_error + 2.5 * height_error
        score += 0.3 * (1.0 - float(metrics["palms_parallel_score"]))
        score += 0.3 * (1.0 - float(metrics["palms_front_facing_score"]))
        if score < best_score:
            best_score = score
            best_name = name
            best_targets = targets
            best_metrics = metrics
    data.qpos[:] = snapshot
    _set_arm_qpos(model, data, best_targets)
    mujoco.mj_forward(model, data)
    return best_name, best_targets, best_metrics, len(candidates)


def run_g1_lucky_compact_carry_posture_validation(
    *,
    headless: bool = False,
    timeout: float = 10.0,
    viewer_speed: float = 1.0,
    cmd_x: float = 0.8,
    cmd_y: float = 0.0,
    cmd_yaw: float = 0.0,
    attach_box: bool = False,
    box_mass_kg: float = 0.01,
    box_size_x: float = 0.12,
    box_size_y: float = 0.338,
    box_size_z: float = 0.10,
    attached_box_finger_mode: str = "light_side_prepare",
    verbose: bool = True,
) -> dict[str, Any]:
    missing = missing_lucky_assets()
    if missing:
        return {"success": False, "failure_reason": "missing_lucky_assets:" + ",".join(missing)}

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

    walker = load_walker_policy(LUCKY_WALKER_ONNX)
    joint_map = build_lucky_joint_map(model, config, control_waist=True)
    posture_name, arm_targets, posture_metrics, candidates_tested = _select_compact_posture(
        model, data, dict(config["default_joint_pos"])
    )
    box_configured = False
    attached_box_center = np.zeros(3, dtype=float)
    if attach_box:
        box_configured = _configure_attached_carry_box(
            model,
            mass_kg=float(box_mass_kg),
            size_xyz=(float(box_size_x), float(box_size_y), float(box_size_z)),
        )
        if box_configured:
            attached_box_center = _attach_carry_box_between_palms(model, data)
            mujoco.mj_forward(model, data)
    finger = Dex3FingerController()
    finger_mode = str(attached_box_finger_mode if box_configured else "box_stable_support_grasp")
    finger_targets = finger.targets_for_mode(finger_mode)
    finger_actuator_count = sum(1 for name in finger_targets if _actuator_id(model, name) >= 0)

    cmd = np.array([float(cmd_x), float(cmd_y), float(cmd_yaw)], dtype=np.float32)
    last_action = np.zeros(joint_map.action_dim, dtype=np.float32)
    target_pos = joint_map.default_joint_pos.copy()
    initial_pelvis = np.asarray(data.qpos[:3], dtype=float).copy()
    initial_yaw = _root_yaw_rad(data)
    left_start = _foot_center_xyz(model, data, "left")
    right_start = _foot_center_xyz(model, data, "right")
    left_prev_lift = False
    right_prev_lift = False
    left_step_start = left_start.copy()
    right_step_start = right_start.copy()
    left_peak_lift = 0.0
    right_peak_lift = 0.0
    left_peak_forward = 0.0
    right_peak_forward = 0.0
    max_left_lift_ever = 0.0
    max_right_lift_ever = 0.0
    actual_visible_step_count = 0
    max_stance_foot_slip = 0.0
    real_foot_placement_count = 0
    max_left_real_placement = 0.0
    max_right_real_placement = 0.0
    control_step = 0
    max_torso_pitch_rad = 0.0
    max_torso_roll_rad = 0.0
    robot_fell = False
    floor_contact_seen = False
    max_hand_box_penetration = 0.0
    max_hand_box_contact_count = 0
    hand_box_collision_seen = False
    start_wall = time.time()

    def step_once() -> bool:
        nonlocal control_step, target_pos, last_action
        nonlocal left_prev_lift, right_prev_lift, left_step_start, right_step_start
        nonlocal left_peak_lift, right_peak_lift, left_peak_forward, right_peak_forward
        nonlocal max_left_lift_ever, max_right_lift_ever, actual_visible_step_count
        nonlocal max_stance_foot_slip, real_foot_placement_count, max_left_real_placement, max_right_real_placement
        nonlocal max_torso_pitch_rad, max_torso_roll_rad, robot_fell, floor_contact_seen
        nonlocal max_hand_box_penetration, max_hand_box_contact_count, hand_box_collision_seen
        if float(data.time) >= float(timeout):
            return False
        left_before = _foot_center_xyz(model, data, "left")
        right_before = _foot_center_xyz(model, data, "right")
        if control_step % 4 == 0:
            obs = _make_observation(data, joint_map, last_action, cmd)
            last_action = np.asarray(walker(obs), dtype=np.float32).reshape(joint_map.action_dim)
            target_pos = action_to_joint_targets(last_action, joint_map)
        _apply_targets(model, data, joint_map, target_pos, hold_masked_arms_at_default=False)
        _apply_arm_targets(model, data, arm_targets)
        _apply_finger_targets(model, data, finger_targets)
        if box_configured:
            _attach_carry_box_between_palms(model, data)
            contact_metrics = _hand_box_contact_metrics(model, data)
            max_hand_box_penetration = max(max_hand_box_penetration, float(contact_metrics["max_hand_box_penetration_m"]))
            max_hand_box_contact_count = max(max_hand_box_contact_count, int(contact_metrics["hand_box_contact_count"]))
            hand_box_collision_seen = hand_box_collision_seen or bool(contact_metrics["hand_box_collision_active"])
        mujoco.mj_step(model, data)
        if box_configured:
            _attach_carry_box_between_palms(model, data)
            mujoco.mj_forward(model, data)
            contact_metrics = _hand_box_contact_metrics(model, data)
            max_hand_box_penetration = max(max_hand_box_penetration, float(contact_metrics["max_hand_box_penetration_m"]))
            max_hand_box_contact_count = max(max_hand_box_contact_count, int(contact_metrics["hand_box_contact_count"]))
            hand_box_collision_seen = hand_box_collision_seen or bool(contact_metrics["hand_box_collision_active"])
        control_step += 1

        left = _foot_center_xyz(model, data, "left")
        right = _foot_center_xyz(model, data, "right")
        left_move = float(np.linalg.norm(left[:2] - left_before[:2]))
        right_move = float(np.linalg.norm(right[:2] - right_before[:2]))
        for side, cur, start, prev_lift, step_move, stance_move in (
            ("left", left, left_step_start, left_prev_lift, left_move, right_move),
            ("right", right, right_step_start, right_prev_lift, right_move, left_move),
        ):
            lift = max(0.0, float(cur[2] - start[2]))
            fwd = max(0.0, float(cur[0] - start[0]))
            is_lift = lift >= VISIBLE_FOOT_LIFT_M
            if side == "left":
                left_peak_lift = max(left_peak_lift, lift)
                left_peak_forward = max(left_peak_forward, fwd)
                max_left_lift_ever = max(max_left_lift_ever, lift)
                max_left_real_placement = max(max_left_real_placement, step_move)
                if prev_lift and not is_lift and left_peak_lift >= VISIBLE_FOOT_LIFT_M and left_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    real_foot_placement_count += int(step_move >= 0.004)
                    max_stance_foot_slip = max(max_stance_foot_slip, stance_move)
                    left_step_start = cur.copy()
                    left_peak_lift = 0.0
                    left_peak_forward = 0.0
                left_prev_lift = is_lift
            else:
                right_peak_lift = max(right_peak_lift, lift)
                right_peak_forward = max(right_peak_forward, fwd)
                max_right_lift_ever = max(max_right_lift_ever, lift)
                max_right_real_placement = max(max_right_real_placement, step_move)
                if prev_lift and not is_lift and right_peak_lift >= VISIBLE_FOOT_LIFT_M and right_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    real_foot_placement_count += int(step_move >= 0.004)
                    max_stance_foot_slip = max(max_stance_foot_slip, stance_move)
                    right_step_start = cur.copy()
                    right_peak_lift = 0.0
                    right_peak_forward = 0.0
                right_prev_lift = is_lift

        roll, pitch = pelvis_roll_pitch_deg(data, model)
        max_torso_pitch_rad = max(max_torso_pitch_rad, abs(float(pitch or 0.0)))
        max_torso_roll_rad = max(max_torso_roll_rad, abs(float(roll or 0.0)))
        pelvis_drop = float(initial_pelvis[2]) - float(data.qpos[2])
        robot_fell = robot_fell or bool(
            max_torso_pitch_rad > TORSO_FALL_RAD or max_torso_roll_rad > ROLL_FALL_RAD or pelvis_drop > PELVIS_DROP_M
        )
        floor_contact_seen = floor_contact_seen or _floor_contact_detected(model, data)
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
    pelvis_lateral = float(data.qpos[1] - initial_pelvis[1])
    pelvis_yaw_change = _wrap_pi(_root_yaw_rad(data) - initial_yaw)
    if box_configured:
        attached_box_center = _attach_carry_box_between_palms(model, data)
        mujoco.mj_forward(model, data)
        contact_metrics = _hand_box_contact_metrics(model, data)
        max_hand_box_penetration = max(max_hand_box_penetration, float(contact_metrics["max_hand_box_penetration_m"]))
        max_hand_box_contact_count = max(max_hand_box_contact_count, int(contact_metrics["hand_box_contact_count"]))
        hand_box_collision_seen = hand_box_collision_seen or bool(contact_metrics["hand_box_collision_active"])
    max_left_lift = max(max_left_lift_ever, max(0.0, float(_foot_center_xyz(model, data, "left")[2] - left_start[2])))
    max_right_lift = max(max_right_lift_ever, max(0.0, float(_foot_center_xyz(model, data, "right")[2] - right_start[2])))
    success = bool(
        posture_metrics["compact_carry_posture_valid"]
        and not robot_fell
        and actual_visible_step_count >= 2
        and max_left_lift >= VISIBLE_FOOT_LIFT_M
        and max_right_lift >= VISIBLE_FOOT_LIFT_M
        and abs(pelvis_forward) >= 0.5
        and floor_contact_seen
    )
    reasons: list[str] = []
    if not success:
        if not posture_metrics["compact_carry_posture_valid"]:
            reasons.append("compact_carry_posture_invalid")
        if robot_fell:
            reasons.append("robot_fell")
        if actual_visible_step_count < 2:
            reasons.append("too_few_visible_steps")
        if abs(pelvis_forward) < 0.5:
            reasons.append("insufficient_walker_progress")
        if not floor_contact_seen:
            reasons.append("no_floor_contact")

    out: dict[str, Any] = {
        "compact_carry_posture_validation": True,
        "policy_loaded": True,
        "walker_policy_path": str(LUCKY_WALKER_ONNX),
        "selected_compact_carry_posture": posture_name,
        "compact_carry_candidates_tested": int(candidates_tested),
        "arms_locked_during_walking": True,
        "finger_hold_commanded": True,
        "finger_hold_mode": finger_mode,
        "finger_hold_actuator_count": int(finger_actuator_count),
        "payload_present": bool(box_configured),
        "payload_attach_mode": "fixed_to_hands_collision_enabled" if box_configured else "none",
        "payload_mass_kg": float(box_mass_kg if box_configured else 0.0),
        "payload_minimum_weight_trial": bool(box_configured and float(box_mass_kg) <= 0.01),
        "attached_box_size_xyz_m": [float(box_size_x), float(box_size_y), float(box_size_z)] if box_configured else None,
        "attached_box_center_world": [float(x) for x in attached_box_center] if box_configured else None,
        "attached_box_configured": bool(box_configured),
        "attached_box_collision_enabled": bool(box_configured),
        "attached_box_fixed_to_hands": bool(box_configured),
        "attached_box_side_clearance_m": float(
            max(0.0, 0.5 * (float(posture_metrics["palm_pair_width_m"]) - float(box_size_y))) if box_configured else 0.0
        ),
        "attached_box_fits_between_palms": bool(
            box_configured and float(box_size_y) <= float(posture_metrics["palm_pair_width_m"]) + 1e-6
        ),
        "hand_box_collision_active": bool(hand_box_collision_seen),
        "max_hand_box_contact_count": int(max_hand_box_contact_count),
        "max_hand_box_penetration_m": float(max_hand_box_penetration),
        "hand_box_impenetrable_visual": bool(box_configured and max_hand_box_penetration <= 0.003),
        **posture_metrics,
        "robot_fell": bool(robot_fell),
        "actual_visible_step_count": int(actual_visible_step_count),
        "real_foot_placement_count": int(real_foot_placement_count),
        "pelvis_forward_progress_m": float(pelvis_forward),
        "pelvis_lateral_progress_m": float(pelvis_lateral),
        "pelvis_yaw_change_rad": float(pelvis_yaw_change),
        "pelvis_yaw_change_deg": float(np.degrees(pelvis_yaw_change)),
        "average_forward_speed_mps": float(pelvis_forward / max(float(data.time), 1e-6)),
        "left_actual_foot_lift_m": float(max_left_lift),
        "right_actual_foot_lift_m": float(max_right_lift),
        "max_left_real_foot_placement_m": float(max_left_real_placement),
        "max_right_real_foot_placement_m": float(max_right_real_placement),
        "max_stance_foot_slip_m": float(max_stance_foot_slip),
        "max_torso_pitch_rad": float(max_torso_pitch_rad),
        "max_torso_roll_rad": float(max_torso_roll_rad),
        "actual_floor_contact_detected": bool(floor_contact_seen),
        "front_obstacles_disabled": bool(obstacle_info["front_obstacles_disabled"]),
        "sim_time_s": float(data.time),
        "wall_time_s": float(time.time() - start_wall),
        "success": bool(success),
        "failure_reason": None if success else "; ".join(reasons),
    }
    if verbose:
        print("----- lucky_compact_carry_posture_validation -----")
        for key, value in out.items():
            print(f"{key}: {value}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Lucky walking with compact carry arm posture.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--viewer-speed", type=float, default=1.0)
    ap.add_argument("--cmd-x", type=float, default=0.8)
    ap.add_argument("--cmd-y", type=float, default=0.0)
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ap.add_argument("--attach-box", action="store_true", help="Attach a light wide box between the locked palms.")
    ap.add_argument("--box-mass-kg", type=float, default=0.01, help="Attached box mass for reporting/minimum-weight trials.")
    ap.add_argument("--box-size-x", type=float, default=0.12)
    ap.add_argument("--box-size-y", type=float, default=0.338)
    ap.add_argument("--box-size-z", type=float, default=0.10)
    ap.add_argument("--attached-box-finger-mode", default="light_side_prepare")
    ns = ap.parse_args()
    run_g1_lucky_compact_carry_posture_validation(
        headless=bool(ns.headless),
        timeout=float(ns.timeout),
        viewer_speed=float(ns.viewer_speed),
        cmd_x=float(ns.cmd_x),
        cmd_y=float(ns.cmd_y),
        cmd_yaw=float(ns.cmd_yaw),
        attach_box=bool(ns.attach_box),
        box_mass_kg=float(ns.box_mass_kg),
        box_size_x=float(ns.box_size_x),
        box_size_y=float(ns.box_size_y),
        box_size_z=float(ns.box_size_z),
        attached_box_finger_mode=str(ns.attached_box_finger_mode),
    )
    if not ns.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
