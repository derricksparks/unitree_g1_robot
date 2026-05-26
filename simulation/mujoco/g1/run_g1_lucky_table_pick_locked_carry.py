#!/usr/bin/env python3
"""Pick the light box, carry it through a right-turn route, and place it."""

from __future__ import annotations

import argparse
import heapq
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
from g1_payload_mpc_wbc import PayloadAwareMPCWBC, command_to_metrics  # noqa: E402
from humanoidverse_bridge.humanoidverse_g1_locomotion import (  # noqa: E402
    make_humanoidverse_actor_obs,
    make_humanoidverse_12dof_policy_config,
    make_humanoidverse_29dof_policy_config,
)
from lucky_bridge.lucky_joint_map import action_to_joint_targets, build_lucky_joint_map  # noqa: E402
from lucky_bridge.lucky_paths import LUCKY_MODEL_CONFIG, LUCKY_SCENE_XML, LUCKY_WALKER_ONNX, missing_lucky_assets  # noqa: E402
from lucky_bridge.lucky_policy_loader import load_lucky_config, load_walker_policy  # noqa: E402
from lucky_bridge.lucky_scene_setup import compute_pick_point, detect_box_ground_truth, setup_lucky_pick_scene  # noqa: E402
from run_g1_lucky_compact_carry_posture_validation import (  # noqa: E402
    _attach_carry_box_between_palms,
    _configure_attached_carry_box,
    _hand_box_contact_metrics,
    _posture_metrics,
)
from run_g1_lucky_locomotion_validation import (  # noqa: E402
    PELVIS_DROP_M,
    ROLL_FALL_RAD,
    TORSO_FALL_RAD,
    _apply_targets,
    _foot_center_xyz,
    _floor_contact_detected,
    _make_observation,
)
from run_g1_lucky_pick_transport_state_machine import _actuator_id, _site_pos, _solve_arm_targets  # noqa: E402
from run_g1_posture_hold import pelvis_roll_pitch_deg  # noqa: E402
from vision.red_box_detector import RedBoxCameraDetector, RedBoxDetectionResult  # noqa: E402


LOCKED_CARRY_ARM_TARGETS: dict[str, float] = {
    "left_shoulder_pitch_joint": 0.60,
    "right_shoulder_pitch_joint": 0.60,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -1.0472,
    "right_elbow_joint": -1.0472,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
LOCKED_CARRY_UPPER_BODY_TARGETS: dict[str, float] = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    **LOCKED_CARRY_ARM_TARGETS,
}
LOCKED_PALM_FORWARD_OFFSET_M = 0.14283506116030464
LOCKED_PALM_LATERAL_OFFSET_M = 0.0
PICK_STANDOFF_FROM_TABLE_EDGE_M = 0.14
ARM_RAISE_STANDOFF_FROM_TABLE_EDGE_M = 0.40
PICK_STAND_TOLERANCE_M = 0.08
BOX_MOTION_BEFORE_GRIP_TOLERANCE_M = 0.005
APPROACH_DISTANCE_FROM_TABLE_M = 0.5
ARM_RAISE_DURATION_S = 1.4
REACH_DURATION_S = 1.6
GRIP_DURATION_S = 1.0
CHEST_HOLD_DURATION_S = 3.0
PICK_TABLE_TURN_CLEARANCE_M = 0.45
PICK_TABLE_CLEARANCE_DURATION_S = 1.2
RIGHT_TURN_DURATION_S = 1.2
RIGHT_TURN_YAW_CMD = -0.45
RIGHT_TURN_BACKUP_CMD_X = 0.0
PRE_TURN_BACKUP_DISTANCE_M = 0.5
PRE_TURN_BACKUP_CMD_X = -0.18
PRE_TURN_BACKUP_MAX_DURATION_S = 6.0
POST_TURN_CLEARANCE_DURATION_S = 1.2
POST_TURN_CLEARANCE_CMD_X = 0.14
SECOND_TABLE_APPROACH_DURATION_S = 4.0
SECOND_TABLE_DISTANCE_M = 2.0
RIGHT_TURN_TARGET_YAW_RAD = -0.95
PLACE_STANDOFF_TOLERANCE_M = 0.22
PLACE_IK_TOLERANCE_M = 0.22
PLACE_FORCE_READY_MAX_WAIT_LOOPS = 6
PLACE_APPROACH_MAX_WAIT_LOOPS = 48
PLACE_ANYWHERE_STANDOFF_M = 0.20
PLACE_ANYWHERE_STANDOFF_TOLERANCE_M = 0.12
PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M = 0.38
PLACE_POINT_AHEAD_OF_ROBOT_M = 0.24
PLACE_RELEASE_GATE_HOLD_S = 2.0
PLACE_LOWER_DURATION_S = 2.0
RELEASE_DURATION_S = 2.0
POST_DONE_HOLD_S = 6.0
UPPER_BODY_STATIC_DRIFT_TOLERANCE_RAD = 0.08
TABLE_SAFE_CLEARANCE_M = 0.16
TABLE_HARD_STOP_CLEARANCE_M = 0.08
STAGGER_STANCE_TARGETS: dict[str, float] = {
    "left_hip_pitch_joint": -0.50,
    "left_knee_joint": 0.95,
    "left_ankle_pitch_joint": -0.42,
    "right_hip_pitch_joint": 0.00,
    "right_knee_joint": 0.10,
    "right_ankle_pitch_joint": 0.00,
}


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, value in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def _apply_joint_ctrl(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, value in targets.items():
        aid = _actuator_id(model, name)
        if aid >= 0:
            data.ctrl[aid] = float(value)


def _snapshot(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in targets:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            out[name] = float(data.qpos[int(model.jnt_qposadr[jid])])
    return out


def _interpolate_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start: dict[str, float],
    target: dict[str, float],
    u: float,
) -> None:
    uu = float(np.clip(u, 0.0, 1.0))
    for name, target_value in target.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            adr = int(model.jnt_qposadr[jid])
            data.qpos[adr] = float(start.get(name, data.qpos[adr]) + uu * (float(target_value) - start.get(name, data.qpos[adr])))


def _box_freejoint_address(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    return int(model.jnt_qposadr[jid]) if jid >= 0 else -1


def _set_box_pose(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray) -> None:
    qadr = _box_freejoint_address(model)
    if qadr < 0:
        return
    data.qpos[qadr : qadr + 7] = [float(pos[0]), float(pos[1]), float(pos[2]), 1.0, 0.0, 0.0, 0.0]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    if jid >= 0:
        data.qvel[int(model.jnt_dofadr[jid]) : int(model.jnt_dofadr[jid]) + 6] = 0.0


def _carry_frame_quat(data: mujoco.MjData, lateral_axis: np.ndarray) -> np.ndarray:
    y_axis = np.asarray(lateral_axis, dtype=float)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        y_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    else:
        y_axis = y_axis / y_norm
    heading = _heading_from_yaw(_root_yaw_rad(data))
    x_seed = np.asarray([heading[0], heading[1], 0.0], dtype=float)
    x_axis = x_seed - float(np.dot(x_seed, y_axis)) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    else:
        z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(1e-6, float(np.linalg.norm(y_axis)))
    mat = np.array(
        [
            float(x_axis[0]),
            float(y_axis[0]),
            float(z_axis[0]),
            float(x_axis[1]),
            float(y_axis[1]),
            float(z_axis[1]),
            float(x_axis[2]),
            float(y_axis[2]),
            float(z_axis[2]),
        ],
        dtype=np.float64,
    )
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _set_box_pose_from_carry_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    lateral_axis: np.ndarray,
) -> None:
    qadr = _box_freejoint_address(model)
    if qadr < 0:
        return
    quat = _carry_frame_quat(data, lateral_axis)
    data.qpos[qadr : qadr + 7] = [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    if jid >= 0:
        data.qvel[int(model.jnt_dofadr[jid]) : int(model.jnt_dofadr[jid]) + 6] = 0.0


def _root_yaw_rad(data: mujoco.MjData) -> float:
    w, x, y, z = [float(v) for v in data.qpos[3:7]]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _imu_torso_yaw_rad(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Yaw estimate from torso orientation (IMU-frame proxy)."""
    torso_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link_rev_1_0"))
    if torso_bid < 0:
        torso_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"))
    if torso_bid < 0:
        return _root_yaw_rad(data)
    mat = np.asarray(data.xmat[torso_bid, :9], dtype=float).reshape(3, 3)
    forward = mat[:, 0]
    return float(np.arctan2(float(forward[1]), float(forward[0])))


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _set_root_yaw(data: mujoco.MjData, yaw: float) -> None:
    half = 0.5 * float(yaw)
    data.qpos[3:7] = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
    data.qvel[3:6] = 0.0


def _heading_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([np.cos(float(yaw)), np.sin(float(yaw))], dtype=float)


def _robot_fell(model: mujoco.MjModel, data: mujoco.MjData, initial_pelvis_z: float) -> bool:
    roll, pitch = pelvis_roll_pitch_deg(data, model)
    pelvis_drop = float(initial_pelvis_z) - float(data.qpos[2])
    return bool(abs(float(pitch or 0.0)) > TORSO_FALL_RAD or abs(float(roll or 0.0)) > ROLL_FALL_RAD or pelvis_drop > PELVIS_DROP_M)


def _contact_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    token_a: tuple[str, ...],
    token_b: tuple[str, ...],
) -> dict[str, float | int | bool]:
    count = 0
    max_penetration = 0.0
    for cid in range(data.ncon):
        con = data.contact[cid]
        names: list[str] = []
        for gid in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[gid])
            geom_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
            body_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
            names.append(f"{geom_name} {body_name}".lower())
        a, b = names
        hit = (any(tok in a for tok in token_a) and any(tok in b for tok in token_b)) or (
            any(tok in b for tok in token_a) and any(tok in a for tok in token_b)
        )
        if hit:
            count += 1
            max_penetration = max(max_penetration, max(0.0, -float(con.dist)))
    return {
        "contact_count": int(count),
        "max_penetration_m": float(max_penetration),
        "contact_active": bool(count > 0),
    }


def _palm_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_palm = _site_pos(model, data, "left_palm")
    right_palm = _site_pos(model, data, "right_palm")
    return 0.5 * (left_palm + right_palm)


def _palm_lateral_axis(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_palm = _site_pos(model, data, "left_palm")
    right_palm = _site_pos(model, data, "right_palm")
    axis = left_palm - right_palm
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return np.asarray([0.0, 1.0, 0.0], dtype=float)
    return axis / norm


def _finger_box_contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    roles = ("thumb", "index", "middle")
    by_digit = {f"{side}_{role}": 0 for side in ("left", "right") for role in roles}
    for cid in range(data.ncon):
        con = data.contact[cid]
        names: list[str] = []
        for gid in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[gid])
            geom_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            body_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            names.append(f"{geom_name} {body_name}")
        a, b = names
        if "red" not in a and "red" not in b:
            continue
        other = b if "red" in a else a
        for side in ("left", "right"):
            for role in roles:
                if f"{side}_hand_{role}" in other:
                    by_digit[f"{side}_{role}"] += 1
    per_role = {role: by_digit[f"left_{role}"] + by_digit[f"right_{role}"] for role in roles}
    return {
        "finger_box_contact_count_by_digit": by_digit,
        "finger_box_contact_count_by_role": per_role,
        "all_three_finger_roles_contact_box_before_lift": bool(all(per_role[role] > 0 for role in roles)),
    }


def _stabilize_manipulation_base(data: mujoco.MjData, *, pelvis_z: float) -> None:
    """Hold the root level while animating the arms for the pickup baseline."""
    data.qpos[2] = float(pelvis_z)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[0:6] = 0.0
    if data.qacc.size >= 6:
        data.qacc[0:6] = 0.0


def _align_base_xy_for_pick(data: mujoco.MjData, target_xy: np.ndarray) -> None:
    data.qpos[0] = float(target_xy[0])
    data.qpos[1] = float(target_xy[1])
    data.qvel[0:2] = 0.0
    if data.qacc.size >= 2:
        data.qacc[0:2] = 0.0


def _set_base_xy(data: mujoco.MjData, xy: np.ndarray) -> None:
    data.qpos[0] = float(xy[0])
    data.qpos[1] = float(xy[1])
    data.qvel[0:2] = 0.0
    if data.qacc.size >= 2:
        data.qacc[0:2] = 0.0


def _estimate_com_margin_m(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    left = _foot_center_xyz(model, data, "left")[:2]
    right = _foot_center_xyz(model, data, "right")[:2]
    pelvis_xy = np.asarray(data.qpos[:2], dtype=float)
    center = 0.5 * (left + right)
    half_span = 0.5 * float(np.linalg.norm(left - right))
    # Conservative margin estimate against lateral COM shift.
    return float(half_span - np.linalg.norm(pelvis_xy - center))


def _place_box_on_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_mass_kg: float,
    box_size_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    scene = setup_lucky_pick_scene(
        model,
        data,
        table_x=0.351,
        box_y=0.0,
        box_size_xyz=box_size_xyz,
        box_edge_offset=0.09,
    )
    _configure_attached_carry_box(model, mass_kg=box_mass_kg, size_xyz=box_size_xyz)
    for gid in range(model.ngeom):
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if name in {"table_leg_2", "table_leg_4"}:
            model.geom_pos[gid, 0] = -0.12
    mujoco.mj_forward(model, data)
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    if box_bid >= 0:
        scene["box_mass_kg"] = float(model.body_mass[box_bid])
    return scene


def _configure_second_table(model: mujoco.MjModel, *, pick_table_xy: np.ndarray) -> dict[str, Any]:
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_white")
    top_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_white_top")
    heading = _heading_from_yaw(RIGHT_TURN_TARGET_YAW_RAD)
    center_xy = np.asarray(pick_table_xy, dtype=float) + heading * SECOND_TABLE_DISTANCE_M
    if table_bid >= 0:
        model.body_pos[table_bid, 0] = float(center_xy[0])
        model.body_pos[table_bid, 1] = float(center_xy[1])
        _set_root_yaw_for_body(model, table_bid, RIGHT_TURN_TARGET_YAW_RAD)
    if top_gid >= 0:
        model.geom_contype[top_gid] = 1
        model.geom_conaffinity[top_gid] = 1
        model.geom_friction[top_gid, :3] = [1.6, 0.2, 0.02]
    for gid in range(model.ngeom):
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if name.startswith("table_white_leg_"):
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1
            if name in {"table_white_leg_2", "table_white_leg_4"}:
                model.geom_pos[gid, 0] = -0.10
    surface_z = 0.633
    if table_bid >= 0 and top_gid >= 0:
        surface_z = float(model.body_pos[table_bid, 2] + model.geom_pos[top_gid, 2] + model.geom_size[top_gid, 2])
    return {
        "second_table_xy": [float(center_xy[0]), float(center_xy[1])],
        "second_table_distance_m": float(np.linalg.norm(center_xy - np.asarray(pick_table_xy, dtype=float))),
        "second_table_yaw_rad": float(RIGHT_TURN_TARGET_YAW_RAD),
        "second_table_surface_z": float(surface_z),
        "second_table_configured": bool(table_bid >= 0 and top_gid >= 0),
    }


def _set_root_yaw_for_body(model: mujoco.MjModel, body_id: int, yaw: float) -> None:
    half = 0.5 * float(yaw)
    model.body_quat[int(body_id), :4] = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _distance_to_oriented_rect(point_xy: np.ndarray, center_xy: np.ndarray, yaw_rad: float, half_x: float, half_y: float) -> float:
    p = np.asarray(point_xy, dtype=float) - np.asarray(center_xy, dtype=float)
    c = float(np.cos(-float(yaw_rad)))
    s = float(np.sin(-float(yaw_rad)))
    lx = c * float(p[0]) - s * float(p[1])
    ly = s * float(p[0]) + c * float(p[1])
    dx = abs(lx) - float(half_x)
    dy = abs(ly) - float(half_y)
    ox = max(dx, 0.0)
    oy = max(dy, 0.0)
    outside = float(np.hypot(ox, oy))
    if dx <= 0.0 and dy <= 0.0:
        return -float(min(-dx, -dy))
    return outside


def _clamp_xy_to_oriented_rect(point_xy: np.ndarray, center_xy: np.ndarray, yaw_rad: float, half_x: float, half_y: float) -> np.ndarray:
    p = np.asarray(point_xy, dtype=float) - np.asarray(center_xy, dtype=float)
    c = float(np.cos(-float(yaw_rad)))
    s = float(np.sin(-float(yaw_rad)))
    lx = c * float(p[0]) - s * float(p[1])
    ly = s * float(p[0]) + c * float(p[1])
    clamped_lx = float(np.clip(lx, -float(half_x), float(half_x)))
    clamped_ly = float(np.clip(ly, -float(half_y), float(half_y)))
    cw = float(np.cos(float(yaw_rad)))
    sw = float(np.sin(float(yaw_rad)))
    wx = cw * clamped_lx - sw * clamped_ly
    wy = sw * clamped_lx + cw * clamped_ly
    return np.asarray([float(center_xy[0] + wx), float(center_xy[1] + wy)], dtype=float)


def _closest_xy_on_oriented_rect_boundary(point_xy: np.ndarray, center_xy: np.ndarray, yaw_rad: float, half_x: float, half_y: float) -> np.ndarray:
    p = np.asarray(point_xy, dtype=float) - np.asarray(center_xy, dtype=float)
    c = float(np.cos(-float(yaw_rad)))
    s = float(np.sin(-float(yaw_rad)))
    lx = c * float(p[0]) - s * float(p[1])
    ly = s * float(p[0]) + c * float(p[1])
    clx = float(np.clip(lx, -float(half_x), float(half_x)))
    cly = float(np.clip(ly, -float(half_y), float(half_y)))
    inside = bool(abs(lx) <= float(half_x) and abs(ly) <= float(half_y))
    if inside:
        dx = float(half_x) - abs(lx)
        dy = float(half_y) - abs(ly)
        if dx <= dy:
            clx = float(np.sign(lx) * float(half_x)) if abs(lx) > 1e-9 else float(half_x)
            cly = float(ly)
        else:
            clx = float(lx)
            cly = float(np.sign(ly) * float(half_y)) if abs(ly) > 1e-9 else float(half_y)
    cw = float(np.cos(float(yaw_rad)))
    sw = float(np.sin(float(yaw_rad)))
    wx = cw * clx - sw * cly
    wy = sw * clx + cw * cly
    return np.asarray([float(center_xy[0] + wx), float(center_xy[1] + wy)], dtype=float)


def _set_goal_marker_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    *,
    ready_to_release: bool = False,
    release_countdown_s: float | None = None,
) -> bool:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "goal_marker_joint")
    if jid < 0:
        return False
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [float(pos[0]), float(pos[1]), float(pos[2]), 1.0, 0.0, 0.0, 0.0]
    vadr = int(model.jnt_dofadr[jid])
    data.qvel[vadr : vadr + 6] = 0.0
    marker_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker_geom"))
    if marker_gid >= 0:
        if ready_to_release:
            alpha = 0.9
            if release_countdown_s is not None:
                # Blink while counting down the 2s release gate hold.
                phase = float(data.time) * 6.0
                alpha = float(0.45 + 0.45 * (0.5 + 0.5 * np.sin(phase)))
            model.geom_rgba[marker_gid, :4] = [0.1, 0.95, 0.1, alpha]
        else:
            model.geom_rgba[marker_gid, :4] = [0.95, 0.8, 0.1, 0.9]
    mujoco.mj_forward(model, data)
    return True


def _plan_shortest_path_astar(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    obstacles: list[tuple[np.ndarray, float, float, float]],
    clearance_m: float,
    resolution_m: float = 0.12,
) -> list[np.ndarray]:
    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    margin = 1.0
    xs = [float(start[0]), float(goal[0])]
    ys = [float(start[1]), float(goal[1])]
    for center, _, hx, hy in obstacles:
        xs.extend([float(center[0]) - hx - clearance_m, float(center[0]) + hx + clearance_m])
        ys.extend([float(center[1]) - hy - clearance_m, float(center[1]) + hy + clearance_m])
    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    def to_grid(p: np.ndarray) -> tuple[int, int]:
        return (
            int(round((float(p[0]) - min_x) / resolution_m)),
            int(round((float(p[1]) - min_y) / resolution_m)),
        )

    def to_world(ix: int, iy: int) -> np.ndarray:
        return np.asarray([min_x + ix * resolution_m, min_y + iy * resolution_m], dtype=float)

    nx = int(max(4, round((max_x - min_x) / resolution_m)))
    ny = int(max(4, round((max_y - min_y) / resolution_m)))
    start_n = to_grid(start)
    goal_n = to_grid(goal)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def blocked(ix: int, iy: int) -> bool:
        if ix < 0 or iy < 0 or ix > nx or iy > ny:
            return True
        p = to_world(ix, iy)
        for center, yaw, hx, hy in obstacles:
            if _distance_to_oriented_rect(p, center, yaw, hx + clearance_m, hy + clearance_m) <= 0.0:
                return True
        return False

    if blocked(*start_n) or blocked(*goal_n):
        return [goal]

    pq: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(pq, (0.0, start_n))
    g_cost: dict[tuple[int, int], float] = {start_n: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    found = False
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal_n:
            found = True
            break
        for dx, dy in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if blocked(*nxt):
                continue
            step = float(np.hypot(dx, dy)) * resolution_m
            ng = g_cost[cur] + step
            if ng >= g_cost.get(nxt, 1e18):
                continue
            g_cost[nxt] = ng
            parent[nxt] = cur
            h = float(np.hypot(nxt[0] - goal_n[0], nxt[1] - goal_n[1])) * resolution_m
            heapq.heappush(pq, (ng + h, nxt))

    if not found:
        return [goal]

    nodes = [goal_n]
    while nodes[-1] != start_n:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    world_nodes = [to_world(ix, iy) for ix, iy in nodes]
    if len(world_nodes) <= 2:
        return [goal]
    # Light simplification: keep every ~4th node and final.
    out = [world_nodes[i] for i in range(3, len(world_nodes), 4)]
    if len(out) == 0 or float(np.linalg.norm(out[-1] - goal)) > 1e-6:
        out.append(goal)
    return out


def run_g1_lucky_table_pick_locked_carry(
    *,
    headless: bool = False,
    timeout: float = 45.0,
    viewer_speed: float = 2.0,
    box_mass_kg: float = 0.01,
    box_size_x: float = 0.12,
    box_size_y: float = 0.33867925908248114,
    box_size_z: float = 0.10,
    lift_duration_s: float = 3.0,
    locomotion_backend: str = "lucky",
    humanoidverse_policy_onnx: str | None = None,
    turn_step_count: int = 5,
    turn_duration_s: float = 4.0,
    turn_yaw_cmd: float = -0.40,
    turn_forward_cmd: float = 0.0,
    assist_final_yaw_correction: bool = False,
    assist_final_alignment: bool = False,
    post_done_hold: float = POST_DONE_HOLD_S,
    demo_fast: bool = False,
    freeze_upper_during_transport: bool = True,
    stop_after_turning_point: bool = False,
    vision_enabled: bool = True,
    save_debug_image: bool = False,
    show_vision_window: bool = False,
    show_depth_window: bool = False,
    show_route_map: bool = False,
    vision_camera_mode: str = "robot_mounted",
    vision_window_fps: float = 10.0,
    lidar_num_rays: int = 72,
    lidar_max_range_m: float = 3.0,
    verbose: bool = True,
) -> dict[str, Any]:
    missing = missing_lucky_assets()
    if missing:
        return {"success": False, "failure_reason": "missing_lucky_assets:" + ",".join(missing)}

    config = load_lucky_config(LUCKY_MODEL_CONFIG)
    model = mujoco.MjModel.from_xml_path(str(LUCKY_SCENE_XML))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(config["joint_names"]):
        data.qpos[7 + i] = float(config["default_joint_pos"].get(name, 0.0))
    scene_metrics = _place_box_on_table(
        model,
        data,
        box_mass_kg=float(box_mass_kg),
        box_size_xyz=(float(box_size_x), float(box_size_y), float(box_size_z)),
    )
    table_front_edge_x = float(scene_metrics["table_front_edge_x"])
    pick_table_xy = np.asarray([0.351, 0.0], dtype=float)
    second_table_metrics = _configure_second_table(model, pick_table_xy=pick_table_xy)
    pick_table_top_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    second_table_top_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_white_top")
    pick_table_half_x = float(model.geom_size[pick_table_top_gid, 0]) if pick_table_top_gid >= 0 else 0.35
    pick_table_half_y = float(model.geom_size[pick_table_top_gid, 1]) if pick_table_top_gid >= 0 else 0.30
    second_table_half_x = float(model.geom_size[second_table_top_gid, 0]) if second_table_top_gid >= 0 else 0.35
    second_table_half_y = float(model.geom_size[second_table_top_gid, 1]) if second_table_top_gid >= 0 else 0.30
    left_hip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_roll_link")
    right_hip_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hip_roll_link")
    data.qpos[0] = table_front_edge_x - APPROACH_DISTANCE_FROM_TABLE_M
    data.qpos[1] = 0.0
    mujoco.mj_forward(model, data)
    vision_detector: RedBoxCameraDetector | None = None
    show_vision_window_active = bool(show_vision_window and not headless)
    show_depth_window_active = bool(show_depth_window and not headless)
    show_route_map_active = bool(show_route_map and not headless)
    cv2_available = False
    if show_vision_window_active or show_depth_window_active or show_route_map_active:
        try:
            import cv2  # type: ignore  # noqa: F401

            cv2_available = True
        except Exception:
            show_vision_window_active = False
            show_depth_window_active = False
            show_route_map_active = False
            print("VISION_WINDOW: OpenCV not available; live camera window disabled.")
    if vision_enabled:
        vision_detector = RedBoxCameraDetector(
            model,
            width=640,
            height=480,
            show_window=show_vision_window_active,
            show_depth_window=show_depth_window_active,
            camera_mode=str(vision_camera_mode),
            window_name="G1 Vision Detection",
            depth_window_name="G1 Depth Perception",
        )
        vision_result = vision_detector.detect(
            data,
            save_debug_image=bool(save_debug_image),
            debug_image_path="/tmp/g1_red_box_detection.png" if save_debug_image else None,
            show_window=show_vision_window_active,
            show_depth_window=show_depth_window_active,
        )
    else:
        gt = detect_box_ground_truth(model, data)
        vision_result = RedBoxDetectionResult(
            vision_enabled=False,
            detection_mode="ground_truth_fallback",
            pose_source="mujoco_ground_truth_only",
            box_detected=bool(gt.box_detected),
            detection_success=bool(gt.box_detected),
            box_bbox_px=None,
            box_center_px=None,
            box_position_world=tuple(float(x) for x in gt.box_position_world),
            frame_processing_time_s=0.0,
            frame_processing_time_ok=True,
            camera_source="ground_truth_only",
            distance_to_box_m=None,
            detection_confidence=0.0,
            obstacle_detected=False,
            obstacle_distance_m=None,
            obstacle_lateral_bias=0.0,
            rgb_fps=None,
            depth_fps=None,
            debug_image_path=None,
            error=None,
        )
    if vision_result.box_position_world is None:
        initial_box = np.asarray(detect_box_ground_truth(model, data).box_position_world, dtype=float)
    else:
        initial_box = np.asarray(vision_result.box_position_world, dtype=float)
    initial_pelvis_z = float(data.qpos[2])

    backend = str(locomotion_backend).strip().lower()
    if backend not in {"lucky", "humanoidverse"}:
        raise ValueError(f"Unsupported locomotion backend: {locomotion_backend}")
    if backend == "humanoidverse":
        if not humanoidverse_policy_onnx:
            raise ValueError("humanoidverse_policy_onnx is required when locomotion_backend='humanoidverse'")
        locomotion_policy = load_walker_policy(humanoidverse_policy_onnx)
        if int(locomotion_policy.output_dim) == 12:
            locomotion_cfg = make_humanoidverse_12dof_policy_config()
            locomotion_joint_map = build_lucky_joint_map(model, locomotion_cfg, control_waist=False)
        elif int(locomotion_policy.output_dim) == 29:
            locomotion_cfg = make_humanoidverse_29dof_policy_config()
            # Keep locomotion ownership on legs + waist only.
            locomotion_joint_map = build_lucky_joint_map(model, locomotion_cfg, control_waist=True)
        else:
            raise ValueError(
                f"Unsupported HumanoidVerse ONNX output dim: {locomotion_policy.output_dim} (expected 12 or 29)"
            )
    else:
        locomotion_policy = load_walker_policy(LUCKY_WALKER_ONNX)
        locomotion_cfg = config
        locomotion_joint_map = build_lucky_joint_map(model, config, control_waist=True)

    payload_mpc = PayloadAwareMPCWBC()
    last_action = np.zeros(locomotion_joint_map.action_dim, dtype=np.float32)
    target_pos = locomotion_joint_map.default_joint_pos.copy()
    last_payload_mpc_metrics: dict[str, Any] = {}
    control_step = 0
    steps_per_window_update = max(1, int(round(1.0 / max(1e-6, float(model.opt.timestep) * max(1.0, float(vision_window_fps))))))
    pick = np.asarray([table_front_edge_x - PICK_STANDOFF_FROM_TABLE_EDGE_M, float(initial_box[1])], dtype=float)
    phases: list[str] = ["INIT_TABLE_BOX"]
    floor_contact_seen = False
    robot_fell = False
    sync_after_step: Any = None
    if demo_fast:
        arm_raise_duration_s = 1.0
        reach_duration_s = 1.2
        grip_duration_s = 0.7
        chest_hold_duration_s = 1.2
        pick_table_clearance_duration_s = max(0.9, 0.75 * PICK_TABLE_CLEARANCE_DURATION_S)
        place_lower_duration_s = 1.2
        release_duration_s = max(2.0, 0.75 * RELEASE_DURATION_S)
        post_turn_clearance_duration_s = 0.9
        viewer_speed = max(float(viewer_speed), 2.0)
    else:
        arm_raise_duration_s = ARM_RAISE_DURATION_S
        reach_duration_s = REACH_DURATION_S
        grip_duration_s = GRIP_DURATION_S
        chest_hold_duration_s = CHEST_HOLD_DURATION_S
        pick_table_clearance_duration_s = PICK_TABLE_CLEARANCE_DURATION_S
        place_lower_duration_s = PLACE_LOWER_DURATION_S
        release_duration_s = RELEASE_DURATION_S
        post_turn_clearance_duration_s = POST_TURN_CLEARANCE_DURATION_S

    turn_uses_walker_steps = True
    turn_root_teleport_used = False
    transport_uses_walker_steps = True
    transport_root_teleporting_used = False
    turn_visible_step_count = 0
    transport_visible_step_count = 0
    carry_turn_actual_step_count = 0
    carry_transport_actual_step_count = 0
    total_carry_actual_step_count = 0
    max_upper_body_joint_drift_rad = 0.0
    max_box_attach_error_m = 0.0
    min_pick_table_clearance_m = 1e9
    min_second_table_clearance_m = 1e9
    safety_cmd_clamps_count = 0
    collision_recovery_count = 0
    obstacle_clearance_target_m = 0.5
    goal_table_approach_clearance_m = 0.12
    obstacle_avoidance_trigger_count = 0
    min_obstacle_distance_m = 1e9
    last_obstacle_lateral_bias = 0.0
    min_pelvis_surrounding_clearance_m = 1e9
    lidar_replan_count = 0
    transport_stall_recovery_count = 0
    goal_marker_active = False
    planned_route_waypoints_count = 0
    planned_route_length_m = 0.0

    def enter_phase(name: str) -> None:
        print(f"PHASE: {name}")
        phases.append(name)

    pelvis_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    lidar_geomgroup = np.ones(6, dtype=np.uint8)
    robot_body_ids: set[int] = set()
    if pelvis_bid >= 0:
        stack = [int(pelvis_bid)]
        while stack:
            bid = int(stack.pop())
            if bid in robot_body_ids:
                continue
            robot_body_ids.add(bid)
            for cid in range(model.nbody):
                if int(model.body_parentid[cid]) == bid:
                    stack.append(int(cid))

    def _scan_pelvis_lidar(*, rays: int, max_range_m: float) -> tuple[np.ndarray, np.ndarray]:
        root_xy = np.asarray(data.qpos[:2], dtype=float)
        if pelvis_bid >= 0:
            origin = np.asarray(data.xpos[pelvis_bid, :3], dtype=np.float64).copy()
        else:
            origin = np.asarray([root_xy[0], root_xy[1], float(data.qpos[2])], dtype=np.float64)
        origin[2] = max(0.60, float(origin[2]) - 0.18)
        yaw = _root_yaw_rad(data)
        dists = np.full((int(rays),), float(max_range_m), dtype=np.float64)
        pts = np.zeros((int(rays), 2), dtype=np.float64)
        geomid = np.zeros((1,), dtype=np.int32)
        for i in range(int(rays)):
            ang = float(yaw + (2.0 * np.pi * i) / max(1, int(rays)))
            vec = np.asarray([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            dist = float(
                mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    vec,
                    lidar_geomgroup,
                    1,
                    int(pelvis_bid),
                    geomid,
                    None,
                )
            )
            if dist < 0.0:
                dist = float(max_range_m)
            if geomid[0] >= 0:
                gid = int(geomid[0])
                gname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
                gbody = int(model.geom_bodyid[gid]) if gid >= 0 else -1
                if gbody in robot_body_ids or gname.startswith("red_"):
                    dist = float(max_range_m)
            dists[i] = float(np.clip(dist, 0.0, float(max_range_m)))
            pts[i, :] = origin[:2] + dists[i] * vec[:2]
        return dists, pts

    def _render_route_map(
        *,
        robot_xy: np.ndarray,
        robot_yaw: float,
        goal_xy: np.ndarray,
        lidar_points_xy: np.ndarray,
        path_points_xy: list[np.ndarray],
    ) -> None:
        if not (show_route_map_active and cv2_available):
            return
        import cv2  # type: ignore

        size = 640
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :] = (20, 20, 20)
        table_color = (100, 160, 220)
        map_center = np.asarray(robot_xy, dtype=float)
        scale = 90.0  # px/m

        def w2p(p: np.ndarray) -> tuple[int, int]:
            rel = np.asarray(p, dtype=float) - map_center
            x = int(size * 0.5 + rel[0] * scale)
            y = int(size * 0.5 - rel[1] * scale)
            return x, y

        def draw_rect(center: np.ndarray, yaw: float, hx: float, hy: float, color: tuple[int, int, int]) -> None:
            c = np.asarray(center, dtype=float)
            rot = np.asarray(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
                dtype=float,
            )
            corners_local = np.asarray([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]], dtype=float)
            pts = (corners_local @ rot.T) + c
            pts_px = np.asarray([w2p(p) for p in pts], dtype=np.int32)
            cv2.polylines(img, [pts_px], True, color, 2, cv2.LINE_AA)

        draw_rect(pick_table_xy, 0.0, pick_table_half_x + obstacle_clearance_target_m, pick_table_half_y + obstacle_clearance_target_m, table_color)
        draw_rect(np.asarray(second_table_metrics["second_table_xy"], dtype=float), float(second_table_metrics["second_table_yaw_rad"]), second_table_half_x + obstacle_clearance_target_m, second_table_half_y + obstacle_clearance_target_m, table_color)

        for pt in lidar_points_xy:
            cv2.circle(img, w2p(np.asarray(pt, dtype=float)), 1, (80, 80, 255), -1)

        if len(path_points_xy) > 0:
            chain = [np.asarray(robot_xy, dtype=float)] + [np.asarray(p, dtype=float) for p in path_points_xy]
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(img, w2p(a), w2p(b), (80, 255, 80), 3, cv2.LINE_AA)
            for p in path_points_xy:
                cv2.circle(img, w2p(np.asarray(p, dtype=float)), 4, (0, 220, 0), -1)

        cv2.circle(img, w2p(np.asarray(goal_xy, dtype=float)), 6, (0, 255, 255), 2)
        robot_p = np.asarray(robot_xy, dtype=float)
        nose = robot_p + 0.12 * _heading_from_yaw(float(robot_yaw))
        left = robot_p + 0.07 * _heading_from_yaw(float(robot_yaw) + 2.45)
        right = robot_p + 0.07 * _heading_from_yaw(float(robot_yaw) - 2.45)
        tri = np.asarray([w2p(nose), w2p(left), w2p(right)], dtype=np.int32)
        cv2.fillPoly(img, [tri], (255, 255, 255))
        cv2.putText(img, "route map (pelvis frame center)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imshow("G1 Route Map", img)
        cv2.waitKey(1)

    step_state: dict[str, dict[str, float | bool] | None] = {
        "left": None,
        "right": None,
    }

    def _count_visible_steps(phase_name: str) -> int:
        nonlocal turn_visible_step_count, transport_visible_step_count
        nonlocal carry_turn_actual_step_count, carry_transport_actual_step_count, total_carry_actual_step_count
        new_steps = 0
        for side in ("left", "right"):
            foot = _foot_center_xyz(model, data, side)
            x = float(foot[0])
            y = float(foot[1])
            z = float(foot[2])
            state = step_state.get(side)
            if not isinstance(state, dict):
                step_state[side] = {
                    "baseline_z": z,
                    "airborne": False,
                    "lift_x": x,
                    "lift_y": y,
                }
                continue
            baseline_z = min(float(state["baseline_z"]), z)
            airborne = bool(state["airborne"])
            lift_x = float(state["lift_x"])
            lift_y = float(state["lift_y"])
            up_thresh = baseline_z + 0.018
            down_thresh = baseline_z + 0.010
            if (not airborne) and z > up_thresh:
                airborne = True
                lift_x = x
                lift_y = y
            elif airborne and z <= down_thresh:
                step_disp = float(np.linalg.norm(np.asarray([x - lift_x, y - lift_y], dtype=float)))
                if step_disp >= 0.02:
                    new_steps += 1
                airborne = False
            step_state[side] = {
                "baseline_z": baseline_z,
                "airborne": airborne,
                "lift_x": lift_x,
                "lift_y": lift_y,
            }
        if new_steps > 0:
            if phase_name == "RIGHT_TURN_WITH_BOX_STEP_BASED":
                turn_visible_step_count += int(new_steps)
                carry_turn_actual_step_count += int(new_steps)
                total_carry_actual_step_count += int(new_steps)
            elif phase_name == "WALK_TO_SECOND_TABLE_WITH_BOX":
                transport_visible_step_count += int(new_steps)
                carry_transport_actual_step_count += int(new_steps)
                total_carry_actual_step_count += int(new_steps)
        return int(new_steps)

    def _record_upper_body_drift(targets: dict[str, float]) -> None:
        nonlocal max_upper_body_joint_drift_rad
        for name, target in targets.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            qadr = int(model.jnt_qposadr[jid])
            drift = abs(float(data.qpos[qadr]) - float(target))
            max_upper_body_joint_drift_rad = max(max_upper_body_joint_drift_rad, float(drift))

    def _record_box_attach_error(offset: np.ndarray) -> None:
        nonlocal max_box_attach_error_m
        expected = _palm_center(model, data) + offset
        measured = np.asarray(detect_box_ground_truth(model, data).box_position_world, dtype=float)
        max_box_attach_error_m = max(max_box_attach_error_m, float(np.linalg.norm(measured - expected)))

    def sync() -> None:
        if sync_after_step is not None:
            sync_after_step()

    def _robot_table_clearance(table_xy: np.ndarray, table_yaw: float, half_x: float, half_y: float) -> float:
        pts: list[np.ndarray] = [np.asarray(data.qpos[:2], dtype=float)]
        if left_hip_bid >= 0:
            pts.append(np.asarray(data.xpos[left_hip_bid][:2], dtype=float))
        if right_hip_bid >= 0:
            pts.append(np.asarray(data.xpos[right_hip_bid][:2], dtype=float))
        dists = [
            _distance_to_oriented_rect(p, table_xy, table_yaw, half_x, half_y)
            for p in pts
        ]
        return float(min(dists))

    def _apply_table_safety(cmd: np.ndarray, *, phase_name: str) -> np.ndarray:
        nonlocal min_pick_table_clearance_m, min_second_table_clearance_m
        nonlocal safety_cmd_clamps_count, collision_recovery_count
        out = np.asarray(cmd, dtype=np.float32).copy()
        heading = _heading_from_yaw(_root_yaw_rad(data))
        lateral = np.asarray([-heading[1], heading[0]], dtype=float)
        pick_clear = _robot_table_clearance(
            table_xy=pick_table_xy,
            table_yaw=0.0,
            half_x=pick_table_half_x,
            half_y=pick_table_half_y,
        )
        second_clear = _robot_table_clearance(
            table_xy=np.asarray(second_table_metrics["second_table_xy"], dtype=float),
            table_yaw=float(second_table_metrics["second_table_yaw_rad"]),
            half_x=second_table_half_x,
            half_y=second_table_half_y,
        )
        min_pick_table_clearance_m = min(min_pick_table_clearance_m, float(pick_clear))
        min_second_table_clearance_m = min(min_second_table_clearance_m, float(second_clear))

        world_cmd = heading * float(out[0]) + lateral * float(out[1])
        closest_table_toward: np.ndarray | None = None
        closest_clear = 1e9

        for clear, table_xy in (
            (pick_clear, pick_table_xy),
            (second_clear, np.asarray(second_table_metrics["second_table_xy"], dtype=float)),
        ):
            to_table = np.asarray(table_xy, dtype=float) - np.asarray(data.qpos[:2], dtype=float)
            n = float(np.linalg.norm(to_table))
            if n < 1e-6:
                continue
            toward = to_table / n
            toward_speed = float(np.dot(world_cmd, toward))
            if clear < closest_clear:
                closest_clear = float(clear)
                closest_table_toward = np.asarray(toward, dtype=float)
            if clear < TABLE_SAFE_CLEARANCE_M and toward_speed > 0.0:
                world_cmd = world_cmd - toward * toward_speed
                out[2] = float(np.clip(float(out[2]), -0.25, 0.25))
                safety_cmd_clamps_count += 1
            if clear < TABLE_HARD_STOP_CLEARANCE_M:
                # Recovery: prioritize turning away instead of freezing.
                world_cmd = np.zeros(2, dtype=float)
                turn_sign = float(np.sign(heading[0] * toward[1] - heading[1] * toward[0]))
                if abs(turn_sign) < 1e-6:
                    turn_sign = 1.0
                out[2] = float(np.clip(0.35 * turn_sign, -0.45, 0.45))
                collision_recovery_count += 1

        out[0] = float(np.dot(world_cmd, heading))
        out[1] = float(np.dot(world_cmd, lateral))
        out[0] = float(np.clip(float(out[0]), -0.25, 0.55))
        out[1] = float(np.clip(float(out[1]), -0.18, 0.18))
        if phase_name == "WALK_TO_SECOND_TABLE_WITH_BOX":
            # Enforce forward-only navigation along planned path.
            out[0] = float(max(0.0, float(out[0])))
            out[1] = 0.0
            if closest_clear < TABLE_HARD_STOP_CLEARANCE_M and closest_table_toward is not None:
                # Hard-stop escape: turn away, then move forward away.
                turn_sign = float(np.sign(heading[0] * closest_table_toward[1] - heading[1] * closest_table_toward[0]))
                if abs(turn_sign) < 1e-6:
                    turn_sign = 1.0
                out[2] = float(np.clip(0.35 * turn_sign, -0.45, 0.45))
                away = -np.asarray(closest_table_toward, dtype=float)
                forward_away = float(np.dot(heading, away))
                if forward_away > 0.35:
                    out[0] = max(float(out[0]), float(np.clip(0.20 * forward_away, 0.08, 0.20)))
                else:
                    out[0] = 0.0
        return out

    def step_walk(
        cmd: np.ndarray,
        *,
        hold_masked_arms_at_default: bool = True,
        carried_box_offset: np.ndarray | None = None,
        pin_carried_box: bool = True,
        phase_name: str = "",
        freeze_upper_targets: dict[str, float] | None = None,
        freeze_finger_targets: dict[str, float] | None = None,
        allow_payload_mpc_waist: bool = True,
    ) -> None:
        nonlocal last_action, target_pos, control_step, floor_contact_seen, robot_fell, last_payload_mpc_metrics
        cmd_arr = np.asarray(cmd, dtype=np.float32).reshape(3)
        if carried_box_offset is not None and phase_name in {
            "BACKWARD_CLEAR_BEFORE_TURN",
            "RIGHT_TURN_WITH_BOX_STEP_BASED",
            "POST_TURN_CLEAR_FROM_PICK_TABLE",
            "WALK_TO_SECOND_TABLE_WITH_BOX",
        }:
            cmd_arr = _apply_table_safety(cmd_arr, phase_name=phase_name)
        if control_step % 4 == 0:
            if backend == "humanoidverse":
                obs = make_humanoidverse_actor_obs(model, data, locomotion_joint_map, cmd_arr, last_action)
            else:
                obs = _make_observation(data, locomotion_joint_map, last_action, cmd_arr)
            last_action = np.asarray(locomotion_policy(obs), dtype=np.float32).reshape(locomotion_joint_map.action_dim)
            target_pos = action_to_joint_targets(last_action, locomotion_joint_map)
        _apply_targets(model, data, locomotion_joint_map, target_pos, hold_masked_arms_at_default=hold_masked_arms_at_default)
        if not hold_masked_arms_at_default:
            if freeze_upper_targets is not None:
                _set_joint_qpos(model, data, freeze_upper_targets)
                _apply_joint_ctrl(model, data, freeze_upper_targets)
                if freeze_finger_targets is not None:
                    _apply_joint_ctrl(model, data, freeze_finger_targets)
            elif carried_box_offset is not None and allow_payload_mpc_waist:
                payload_xyz = _palm_center(model, data) + carried_box_offset
                mpc_cmd = payload_mpc.solve(
                    pelvis_xy=np.asarray(data.qpos[:2], dtype=float),
                    payload_xyz=payload_xyz,
                    com_margin_m=_estimate_com_margin_m(model, data),
                )
                mpc_targets = dict(LOCKED_CARRY_UPPER_BODY_TARGETS)
                mpc_targets["waist_pitch_joint"] = float(mpc_cmd.torso_pitch_bias_rad)
                mpc_targets["waist_roll_joint"] = float(mpc_cmd.torso_roll_bias_rad)
                _apply_joint_ctrl(model, data, mpc_targets)
                last_payload_mpc_metrics = command_to_metrics(mpc_cmd)
            elif carried_box_offset is not None:
                _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
                if freeze_finger_targets is not None:
                    _apply_joint_ctrl(model, data, freeze_finger_targets)
            else:
                _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        if carried_box_offset is None:
            _set_box_pose(model, data, initial_box)
        elif pin_carried_box:
            _set_box_pose_from_carry_frame(
                model,
                data,
                _palm_center(model, data) + carried_box_offset,
                _palm_lateral_axis(model, data),
            )
        mujoco.mj_step(model, data)
        if carried_box_offset is None:
            _set_box_pose(model, data, initial_box)
        elif pin_carried_box:
            _set_box_pose_from_carry_frame(
                model,
                data,
                _palm_center(model, data) + carried_box_offset,
                _palm_lateral_axis(model, data),
            )
            _record_box_attach_error(carried_box_offset)
        mujoco.mj_forward(model, data)
        if freeze_upper_targets is not None:
            _record_upper_body_drift(freeze_upper_targets)
        floor_contact_seen = floor_contact_seen or _floor_contact_detected(model, data)
        robot_fell = robot_fell or _robot_fell(model, data, initial_pelvis_z)
        if phase_name:
            _count_visible_steps(phase_name)
        if vision_detector is not None and (show_vision_window_active or show_depth_window_active) and (control_step % steps_per_window_update == 0):
            vision_detector.detect(data, show_window=show_vision_window_active, show_depth_window=show_depth_window_active)
        control_step += 1
        sync()

    def run_step_turn_in_place(
        *,
        yaw_cmd: float,
        forward_cmd: float,
        duration_s: float,
        target_step_count: int,
        carried_box_offset: np.ndarray,
        contact_cb: Any,
        freeze_upper_targets: dict[str, float] | None,
        freeze_finger_targets: dict[str, float] | None,
    ) -> tuple[float, float, int, bool]:
        start_yaw = _root_yaw_rad(data)
        start_t = float(data.time)
        start_steps = int(turn_visible_step_count)
        while float(data.time) - start_t < float(duration_s):
            step_phase = int((float(data.time) - start_t) / 0.40) % 2
            step_bias_x = 0.04 if step_phase == 0 else -0.02
            step_walk(
                np.array([float(forward_cmd) + step_bias_x, 0.0, float(yaw_cmd)], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carried_box_offset,
                phase_name="RIGHT_TURN_WITH_BOX_STEP_BASED",
                freeze_upper_targets=freeze_upper_targets,
                freeze_finger_targets=freeze_finger_targets,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            contact_cb("RIGHT_TURN_WITH_BOX_STEP_BASED")
        yaw_change = _wrap_pi(_root_yaw_rad(data) - start_yaw)
        elapsed = max(1e-6, float(data.time) - start_t)
        steps = int(turn_visible_step_count) - start_steps
        turn_ok = bool(abs(float(yaw_change)) >= 0.75 and steps >= 4)
        return float(yaw_change), float(elapsed), int(steps), bool(turn_ok)

    def walk_to_waypoint_with_steps(
        *,
        target_xy: np.ndarray,
        target_yaw: float,
        max_duration: float,
        carried_box_offset: np.ndarray,
        contact_cb: Any,
        freeze_upper_targets: dict[str, float] | None,
        freeze_finger_targets: dict[str, float] | None,
    ) -> tuple[float, float, bool]:
        nonlocal obstacle_avoidance_trigger_count, min_obstacle_distance_m, last_obstacle_lateral_bias
        nonlocal min_pelvis_surrounding_clearance_m, lidar_replan_count, transport_stall_recovery_count
        start_t = float(data.time)
        reached = False
        best_goal_dist = float(np.linalg.norm(np.asarray(target_xy, dtype=float) - np.asarray(data.qpos[:2], dtype=float)))
        stall_since = float(data.time)
        last_nav_log_t = float(data.time)
        dynamic_route: list[np.ndarray] = _plan_shortest_path_astar(
            start_xy=np.asarray(data.qpos[:2], dtype=float),
            goal_xy=np.asarray(target_xy, dtype=float),
            obstacles=[
                (pick_table_xy, 0.0, pick_table_half_x, pick_table_half_y),
                (
                    np.asarray(second_table_metrics["second_table_xy"], dtype=float),
                    float(second_table_metrics["second_table_yaw_rad"]),
                    second_table_half_x + goal_table_approach_clearance_m,
                    second_table_half_y + goal_table_approach_clearance_m,
                ),
            ],
            clearance_m=float(obstacle_clearance_target_m),
            resolution_m=0.12,
        )
        last_replan_t = -1e9
        while float(data.time) - start_t < float(max_duration):
            root_xy = np.asarray(data.qpos[:2], dtype=float)
            goal_dist_now = float(np.linalg.norm(np.asarray(target_xy, dtype=float) - root_xy))
            if goal_dist_now < best_goal_dist - 0.01:
                best_goal_dist = goal_dist_now
                stall_since = float(data.time)
            if len(dynamic_route) > 0:
                wp = np.asarray(dynamic_route[0], dtype=float)
                if float(np.linalg.norm(wp - root_xy)) < 0.12:
                    dynamic_route = dynamic_route[1:]
            nav_target_xy = np.asarray(dynamic_route[0], dtype=float) if len(dynamic_route) > 0 else np.asarray(target_xy, dtype=float)
            err_xy = nav_target_xy - root_xy
            dist = float(np.linalg.norm(err_xy))
            if dist < 0.20:
                if len(dynamic_route) == 0:
                    reached = True
                    break
                dynamic_route = dynamic_route[1:]
                continue
            if robot_fell or dist > 5.0:
                break
            yaw = _root_yaw_rad(data)
            heading = _heading_from_yaw(yaw)
            lateral = np.asarray([-heading[1], heading[0]], dtype=float)
            nav_heading_yaw = float(np.arctan2(float(err_xy[1]), float(err_xy[0])))
            align_target_yaw = float(target_yaw) if dist < 0.25 and len(dynamic_route) <= 1 else nav_heading_yaw
            yaw_err = _wrap_pi(float(align_target_yaw) - yaw)
            cmd_y = 0.0  # forward-only navigation (no sideways movement)
            align_scale = float(max(0.0, np.cos(float(yaw_err))))
            cmd_x = float(np.clip(0.70 * dist * align_scale, 0.0, 0.26))
            if abs(float(yaw_err)) > 0.65:
                # Keep small forward cadence while turning to avoid standstill.
                cmd_x = 0.04
            cmd_yaw = float(np.clip(1.1 * yaw_err, -0.35, 0.35))
            dists, pts_xy = _scan_pelvis_lidar(rays=int(lidar_num_rays), max_range_m=float(lidar_max_range_m))
            nearest = float(np.min(dists)) if dists.size > 0 else float(lidar_max_range_m)
            min_pelvis_surrounding_clearance_m = min(min_pelvis_surrounding_clearance_m, float(nearest))

            rel = np.asarray(pts_xy, dtype=float) - np.asarray(root_xy, dtype=float)
            fwd = rel @ heading
            lat = rel @ lateral
            front_mask = (fwd > 0.05) & (np.abs(lat) < 0.9)
            forward_d = float(np.min(np.linalg.norm(rel[front_mask], axis=1))) if np.any(front_mask) else float(lidar_max_range_m)
            min_obstacle_distance_m = min(min_obstacle_distance_m, float(forward_d))
            close_mask = (fwd > 0.05) & (np.linalg.norm(rel, axis=1) < 1.2)
            if np.any(close_mask):
                inv = 1.0 / np.maximum(0.05, np.linalg.norm(rel[close_mask], axis=1))
                bias = float(np.sum(lat[close_mask] * inv) / np.sum(inv))
                last_obstacle_lateral_bias = float(np.clip(bias, -1.0, 1.0))
            else:
                last_obstacle_lateral_bias = 0.0

            dist_to_goal = float(np.linalg.norm(np.asarray(target_xy, dtype=float) - root_xy))
            effective_clearance = float(obstacle_clearance_target_m) if dist_to_goal > 0.9 else float(goal_table_approach_clearance_m)
            if forward_d < effective_clearance:
                obstacle_avoidance_trigger_count += 1
                severity = float(np.clip((effective_clearance - float(forward_d)) / max(0.05, effective_clearance), 0.0, 1.0))
                cmd_x = min(cmd_x, float(0.16 * (1.0 - severity)))
                steer = -float(np.sign(last_obstacle_lateral_bias)) if abs(last_obstacle_lateral_bias) > 1e-3 else 0.0
                if abs(steer) < 1e-3:
                    steer = 1.0 if yaw_err >= 0.0 else -1.0
                cmd_yaw = float(np.clip(cmd_yaw + 0.38 * severity * steer, -0.45, 0.45))

            if float(data.time) - stall_since > 2.0 and goal_dist_now > 0.25:
                transport_stall_recovery_count += 1
                # Hard recovery: rotate to reacquire heading, then re-engage forward.
                cmd_x = 0.0
                cmd_y = 0.0
                cmd_yaw = float(np.clip(0.45 * np.sign(yaw_err if abs(yaw_err) > 1e-3 else 1.0), -0.45, 0.45))
                stall_since = float(data.time)

            if float(data.time) - last_nav_log_t > 1.0:
                print(
                    "NAV:",
                    f"goal_dist={goal_dist_now:.3f}",
                    f"nearest_clear={nearest:.3f}",
                    f"forward_clear={forward_d:.3f}",
                    f"cmd_x={cmd_x:.3f}",
                    f"cmd_yaw={cmd_yaw:.3f}",
                    f"replans={lidar_replan_count}",
                    f"recoveries={transport_stall_recovery_count}",
                )
                last_nav_log_t = float(data.time)

            if (
                forward_d < 0.85
                and float(data.time) - float(last_replan_t) > 0.8
                and float(np.linalg.norm(np.asarray(target_xy, dtype=float) - root_xy)) > 1.0
            ):
                lidar_obs: list[tuple[np.ndarray, float, float, float]] = []
                for p, dd in zip(pts_xy, dists):
                    if float(dd) >= min(1.4, float(lidar_max_range_m)):
                        continue
                    lidar_obs.append((np.asarray(p, dtype=float), 0.0, 0.08, 0.08))
                dynamic_obs = [
                    (pick_table_xy, 0.0, pick_table_half_x, pick_table_half_y),
                    (
                        np.asarray(second_table_metrics["second_table_xy"], dtype=float),
                        float(second_table_metrics["second_table_yaw_rad"]),
                        second_table_half_x + goal_table_approach_clearance_m,
                        second_table_half_y + goal_table_approach_clearance_m,
                    ),
                    *lidar_obs,
                ]
                dynamic_route = _plan_shortest_path_astar(
                    start_xy=root_xy,
                    goal_xy=np.asarray(target_xy, dtype=float),
                    obstacles=dynamic_obs,
                    clearance_m=float(obstacle_clearance_target_m),
                    resolution_m=0.12,
                )
                lidar_replan_count += 1
                last_replan_t = float(data.time)

            _render_route_map(
                robot_xy=root_xy,
                robot_yaw=float(yaw),
                goal_xy=np.asarray(target_xy, dtype=float),
                lidar_points_xy=np.asarray(pts_xy, dtype=float),
                path_points_xy=dynamic_route,
            )
            step_walk(
                np.array([cmd_x, cmd_y, cmd_yaw], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carried_box_offset,
                phase_name="WALK_TO_SECOND_TABLE_WITH_BOX",
                freeze_upper_targets=freeze_upper_targets,
                freeze_finger_targets=freeze_finger_targets,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            contact_cb("WALK_TO_SECOND_TABLE_WITH_BOX")
        progress = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - np.asarray(target_xy, dtype=float)))
        elapsed = max(1e-6, float(data.time) - start_t)
        return float(progress), float(elapsed), bool(reached)

    def run_sequence() -> dict[str, Any]:
        nonlocal robot_fell, transport_root_teleporting_used
        nonlocal goal_marker_active, planned_route_waypoints_count, planned_route_length_m
        nonlocal min_pelvis_surrounding_clearance_m, lidar_replan_count
        nonlocal transport_stall_recovery_count
        enter_phase("DETECT_BOX")
        print(
            "VISION:",
            f"mode={vision_result.detection_mode}",
            f"camera_source={vision_result.camera_source}",
            f"box_detected={vision_result.box_detected}",
            f"detection_confidence={vision_result.detection_confidence:.2f}",
            f"distance_to_box_m={vision_result.distance_to_box_m}",
            f"obstacle_distance_m={vision_result.obstacle_distance_m}",
            f"frame_processing_time_s={vision_result.frame_processing_time_s:.4f}",
            f"frame_processing_time_ok={vision_result.frame_processing_time_ok}",
        )
        # Walk to the explicit pre-pick stop point at 0.4 m from table.
        arm_raise_xy = np.asarray(
            [table_front_edge_x - ARM_RAISE_STANDOFF_FROM_TABLE_EDGE_M, float(initial_box[1])],
            dtype=float,
        )
        start = float(data.time)
        while float(data.time) - start < min(float(timeout), 22.0):
            delta_xy = arm_raise_xy - np.asarray(data.qpos[:2], dtype=float)
            if float(np.linalg.norm(delta_xy)) <= PICK_STAND_TOLERANCE_M:
                break
            cmd_x = float(np.clip(1.2 * float(delta_xy[0]), 0.06, 0.60))
            step_walk(np.array([cmd_x, float(np.clip(1.5 * float(delta_xy[1]), -0.20, 0.20)), 0.0], dtype=np.float32))
        enter_phase("WALK_TO_PICK_TABLE")
        distance_to_arm_raise = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - arm_raise_xy))
        robot_table_standoff_m = float(table_front_edge_x - float(data.qpos[0]))
        walk_success = bool(distance_to_arm_raise <= 0.20 and not robot_fell)

        enter_phase("STOP_AT_TABLE")
        left_foot = _foot_center_xyz(model, data, "left")
        right_foot = _foot_center_xyz(model, data, "right")
        stance_foot_lead_m = float(abs(float(left_foot[0]) - float(right_foot[0])))
        one_foot_ahead_for_reach = bool(stance_foot_lead_m >= 0.015)
        enter_phase("STAGGER_STANCE_FOR_REACH")

        finger = Dex3FingerController(lp_alpha=1.0, max_delta_rad=10.0)
        finger.reset({name: 0.0 for name in finger.targets_for_mode("open").keys()})
        open_targets = finger.targets_for_mode("pregrasp_spread")
        finger_targets = finger.targets_for_mode("side_support_grasp")
        max_hand_box_penetration = 0.0
        max_hand_table_penetration = 0.0
        max_hand_table_contact_count = 0
        max_box_motion_before_grip_m = 0.0
        hand_table_contact_count_by_phase: dict[str, int] = {}
        max_robot_table_penetration = 0.0
        max_robot_table_contact_count = 0
        robot_table_contact_count_by_phase: dict[str, int] = {}
        robot_table_contact_pairs_by_phase: dict[str, list[str]] = {}

        def record_hand_table_contact(phase: str) -> None:
            nonlocal max_hand_table_penetration, max_hand_table_contact_count
            nonlocal max_robot_table_penetration, max_robot_table_contact_count
            table_contact = _contact_metrics(
                model,
                data,
                token_a=("hand", "palm", "thumb", "index", "middle", "wrist"),
                token_b=("table",),
            )
            count = int(table_contact["contact_count"])
            hand_table_contact_count_by_phase[phase] = max(
                hand_table_contact_count_by_phase.get(phase, 0),
                count,
            )
            max_hand_table_penetration = max(max_hand_table_penetration, float(table_contact["max_penetration_m"]))
            max_hand_table_contact_count = max(max_hand_table_contact_count, count)
            robot_table_contact = _contact_metrics(
                model,
                data,
                token_a=(
                    "pelvis",
                    "torso",
                    "waist",
                    "hip",
                    "knee",
                    "ankle",
                    "foot",
                    "shoulder",
                    "elbow",
                    "wrist",
                    "hand",
                    "palm",
                    "thumb",
                    "index",
                    "middle",
                ),
                token_b=("table",),
            )
            robot_count = int(robot_table_contact["contact_count"])
            if robot_count > 0 and phase not in robot_table_contact_pairs_by_phase:
                pairs: list[str] = []
                for cid in range(data.ncon):
                    con = data.contact[cid]
                    names = []
                    for gid in (int(con.geom1), int(con.geom2)):
                        bid = int(model.geom_bodyid[gid])
                        geom_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
                        body_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
                        names.append(f"{geom_name}/{body_name}")
                    joined = " <-> ".join(names)
                    if "table" in joined.lower():
                        pairs.append(joined)
                robot_table_contact_pairs_by_phase[phase] = pairs[:6]
            robot_table_contact_count_by_phase[phase] = max(
                robot_table_contact_count_by_phase.get(phase, 0),
                robot_count,
            )
            max_robot_table_penetration = max(
                max_robot_table_penetration,
                float(robot_table_contact["max_penetration_m"]),
            )
            max_robot_table_contact_count = max(max_robot_table_contact_count, robot_count)

        arm_raise_start_snapshot = _snapshot(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        arm_raise_start = float(data.time)
        while float(data.time) - arm_raise_start < arm_raise_duration_s:
            raw_u = (float(data.time) - arm_raise_start) / max(arm_raise_duration_s, 1e-6)
            u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
            _interpolate_qpos(model, data, arm_raise_start_snapshot, LOCKED_CARRY_UPPER_BODY_TARGETS, u)
            _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
            _apply_joint_ctrl(model, data, open_targets)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            _set_box_pose(model, data, initial_box)
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            record_hand_table_contact("RAISE_HANDS_CLEAR_OF_TABLE")
            robot_fell = robot_fell or _robot_fell(model, data, initial_pelvis_z)
            sync()
        enter_phase("RAISE_HANDS_CLEAR_OF_TABLE")

        if APPROACH_DISTANCE_FROM_TABLE_M > 0.6:
            final_approach_start = float(data.time)
            while float(data.time) - final_approach_start < 4.0:
                delta_xy = pick - np.asarray(data.qpos[:2], dtype=float)
                if float(np.linalg.norm(delta_xy)) <= PICK_STAND_TOLERANCE_M or float(data.qpos[0]) >= float(pick[0]) - 0.02:
                    break
                cmd_x = float(np.clip(1.0 * float(delta_xy[0]), 0.04, 0.20))
                step_walk(
                    np.array([cmd_x, float(np.clip(1.2 * float(delta_xy[1]), -0.12, 0.12)), 0.0], dtype=np.float32),
                    hold_masked_arms_at_default=False,
                )
                record_hand_table_contact("FINAL_STANDOFF_APPROACH")
            enter_phase("FINAL_STANDOFF_APPROACH")
            final_stop_start = float(data.time)
            while float(data.time) - final_stop_start < 0.5:
                step_walk(np.zeros(3, dtype=np.float32), hold_masked_arms_at_default=False)
                record_hand_table_contact("FINAL_STANDOFF_APPROACH")
        else:
            enter_phase("FINAL_STANDOFF_APPROACH")
        pre_standoff_alignment_error = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - pick))
        # Avoid hard base snapping unless we're significantly off target.
        if pre_standoff_alignment_error > 0.12:
            _align_base_xy_for_pick(data, pick)
        _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
        _set_joint_qpos(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        mujoco.mj_forward(model, data)
        distance_to_pick = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - pick))
        robot_table_standoff_m = float(table_front_edge_x - float(data.qpos[0]))
        walk_success = bool(
            walk_success
            and distance_to_pick <= PICK_STAND_TOLERANCE_M
            and robot_table_standoff_m >= PICK_STANDOFF_FROM_TABLE_EDGE_M - 0.02
            and not robot_fell
        )

        half_y = 0.5 * float(box_size_y)
        reach_z = float(initial_box[2] + 0.02)
        reach_left = np.array([initial_box[0], initial_box[1] + half_y, reach_z], dtype=float)
        reach_right = np.array([initial_box[0], initial_box[1] - half_y, reach_z], dtype=float)
        reach_targets, reach_left_err, reach_right_err = _solve_arm_targets(
            model,
            data,
            left_target=reach_left,
            right_target=reach_right,
        )
        reach_success = bool(max(float(reach_left_err), float(reach_right_err)) <= 0.10)
        arm_start = _snapshot(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        reach_start = float(data.time)
        while float(data.time) - reach_start < reach_duration_s:
            raw_u = (float(data.time) - reach_start) / max(reach_duration_s, 1e-6)
            u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
            _interpolate_qpos(model, data, arm_start, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **reach_targets}, u)
            _apply_joint_ctrl(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **reach_targets})
            _apply_joint_ctrl(model, data, open_targets)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            _set_box_pose(model, data, initial_box)
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            max_box_motion_before_grip_m = max(
                max_box_motion_before_grip_m,
                float(np.linalg.norm(np.asarray(detect_box_ground_truth(model, data).box_position_world, dtype=float) - initial_box)),
            )
            record_hand_table_contact("REACH_BOX_FROM_STANDOFF")
            robot_fell = robot_fell or _robot_fell(model, data, initial_pelvis_z)
            sync()
        enter_phase("REACH_BOX_FROM_STANDOFF")

        grip_start = float(data.time)
        closing_targets = finger.targets_for_mode("side_support_grasp")
        while float(data.time) - grip_start < grip_duration_s:
            close_u = (float(data.time) - grip_start) / max(grip_duration_s, 1e-6)
            _set_joint_qpos(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **reach_targets})
            _apply_joint_ctrl(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **reach_targets})
            _apply_joint_ctrl(model, data, finger.step("side_support_grasp"))
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            _set_box_pose(model, data, initial_box)
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            max_box_motion_before_grip_m = max(
                max_box_motion_before_grip_m,
                float(np.linalg.norm(np.asarray(detect_box_ground_truth(model, data).box_position_world, dtype=float) - initial_box)),
            )
            record_hand_table_contact("GRIP_BOX_ON_TABLE")
            sync()
        enter_phase("GRIP_BOX_ON_TABLE")
        mujoco.mj_forward(model, data)
        grip_palm_center = _palm_center(model, data)
        carry_offset_from_hands = initial_box - grip_palm_center
        grip_posture_metrics = _posture_metrics(model, data, reach_targets)
        hand_box_side_gap_at_grip_m = max(0.0, 0.5 * (float(grip_posture_metrics["palm_pair_width_m"]) - float(box_size_y)))
        hands_touch_box_before_lift = bool(hand_box_side_gap_at_grip_m <= 1e-4)
        finger_contact = _finger_box_contact_summary(model, data)
        all_three_fingers_commanded_to_touch = bool(
            np.isclose(close_u, 1.0, atol=0.02) or all(abs(float(data.ctrl[_actuator_id(model, jn)]) - float(closing_targets[jn])) < 0.05 for jn in closing_targets if _actuator_id(model, jn) >= 0)
        )
        all_three_fingers_touch_box_before_lift = bool(
            hands_touch_box_before_lift
            and all_three_fingers_commanded_to_touch
            and (
                bool(finger_contact["all_three_finger_roles_contact_box_before_lift"])
                or float(hand_box_side_gap_at_grip_m) <= 1e-4
            )
        )

        lift_start = float(data.time)
        attached_box_center = initial_box.copy()
        while float(data.time) - lift_start < float(lift_duration_s):
            raw_u = (float(data.time) - lift_start) / max(float(lift_duration_s), 1e-6)
            u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
            _interpolate_qpos(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **reach_targets}, LOCKED_CARRY_UPPER_BODY_TARGETS, u)
            _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
            _apply_joint_ctrl(model, data, finger_targets)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            mujoco.mj_forward(model, data)
            palm_center = _palm_center(model, data)
            box_pos = palm_center + carry_offset_from_hands
            _set_box_pose_from_carry_frame(model, data, box_pos, _palm_lateral_axis(model, data))
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            contact = _hand_box_contact_metrics(model, data)
            max_hand_box_penetration = max(max_hand_box_penetration, float(contact["max_hand_box_penetration_m"]))
            record_hand_table_contact("LIFT_TO_LOCKED_CARRY")
            attached_box_center = box_pos.copy()
            robot_fell = robot_fell or _robot_fell(model, data, initial_pelvis_z)
            sync()
        enter_phase("LIFT_TO_LOCKED_CARRY")

        hold_start = float(data.time)
        while float(data.time) - hold_start < chest_hold_duration_s:
            _set_joint_qpos(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
            _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
            _apply_joint_ctrl(model, data, finger_targets)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            attached_box_center = _palm_center(model, data) + carry_offset_from_hands
            _set_box_pose_from_carry_frame(model, data, attached_box_center, _palm_lateral_axis(model, data))
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            contact = _hand_box_contact_metrics(model, data)
            max_hand_box_penetration = max(max_hand_box_penetration, float(contact["max_hand_box_penetration_m"]))
            record_hand_table_contact("HOLD_LOCKED_CARRY")
            sync()
        enter_phase("HOLD_LOCKED_CARRY")
        chest_box_center = attached_box_center.copy()
        carry_posture_metrics_at_chest = _posture_metrics(model, data, LOCKED_CARRY_ARM_TARGETS)

        turn_clearance_m = float(table_front_edge_x - float(data.qpos[0]))
        enter_phase("BACKWARD_CLEAR_BEFORE_TURN")
        pre_turn_start_xy = np.asarray(data.qpos[:2], dtype=float).copy()
        pre_turn_start_t = float(data.time)
        while True:
            moved_m = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - pre_turn_start_xy))
            if moved_m >= PRE_TURN_BACKUP_DISTANCE_M:
                break
            if float(data.time) - pre_turn_start_t >= PRE_TURN_BACKUP_MAX_DURATION_S:
                break
            step_walk(
                np.array([PRE_TURN_BACKUP_CMD_X, 0.0, 0.0], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                phase_name="BACKWARD_CLEAR_BEFORE_TURN",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("BACKWARD_CLEAR_BEFORE_TURN")
            attached_box_center = _palm_center(model, data) + carry_offset_from_hands

        right_turn_start_yaw = _root_yaw_rad(data)
        turn_start_xy = np.asarray(data.qpos[:2], dtype=float).copy()
        enter_phase("RIGHT_TURN_WITH_BOX_STEP_BASED")
        right_turn_yaw_change, right_turn_elapsed_s, _, right_turn_success = run_step_turn_in_place(
            yaw_cmd=float(turn_yaw_cmd),
            forward_cmd=float(turn_forward_cmd),
            duration_s=float(turn_duration_s),
            target_step_count=int(turn_step_count),
            carried_box_offset=carry_offset_from_hands,
            contact_cb=record_hand_table_contact,
            freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
            freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
        )
        if assist_final_yaw_correction:
            target_yaw = float(second_table_metrics["second_table_yaw_rad"])
            current_yaw = _root_yaw_rad(data)
            yaw_err = _wrap_pi(target_yaw - current_yaw)
            if abs(yaw_err) <= 0.12:
                _set_root_yaw(data, target_yaw)
        _set_joint_qpos(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        _set_box_pose_from_carry_frame(
            model,
            data,
            _palm_center(model, data) + carry_offset_from_hands,
            _palm_lateral_axis(model, data),
        )
        mujoco.mj_forward(model, data)
        right_turn_xy_drift_m = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - turn_start_xy))
        right_turn_success = bool(right_turn_success and right_turn_yaw_change <= -1.20 and turn_visible_step_count >= 4)
        turn_success = bool(turn_visible_step_count >= 4 and abs(float(right_turn_yaw_change)) >= 1.20)

        enter_phase("POST_TURN_CLEAR_FROM_PICK_TABLE")
        post_turn_clearance_start = float(data.time)
        while float(data.time) - post_turn_clearance_start < post_turn_clearance_duration_s:
            step_walk(
                np.array([POST_TURN_CLEARANCE_CMD_X, 0.0, 0.0], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                phase_name="WALK_TO_SECOND_TABLE_WITH_BOX",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("POST_TURN_CLEAR_FROM_PICK_TABLE")
            attached_box_center = _palm_center(model, data) + carry_offset_from_hands

        if stop_after_turning_point:
            turn_stop_hold_start = float(data.time)
            while float(data.time) - turn_stop_hold_start < 1.0:
                _set_joint_qpos(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
                _apply_joint_ctrl(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
                _apply_joint_ctrl(model, data, finger_targets)
                _set_box_pose_from_carry_frame(
                    model,
                    data,
                    _palm_center(model, data) + carry_offset_from_hands,
                    _palm_lateral_axis(model, data),
                )
                data.time += float(model.opt.timestep)
                mujoco.mj_forward(model, data)
                record_hand_table_contact("TURNING_POINT_HOLD")
                sync()
            enter_phase("TURNING_POINT_HOLD")
            posture_metrics = _posture_metrics(model, data, LOCKED_CARRY_ARM_TARGETS)
            turn_stage_success = bool(
                walk_success
                and reach_success
                and one_foot_ahead_for_reach
                and turn_success
                and not robot_fell
                and max_hand_table_contact_count == 0
                and max_robot_table_contact_count == 0
                and floor_contact_seen
            )
            return {
                "table_pick_locked_carry": True,
                "phases": phases,
                "policy_loaded": True,
                "locomotion_backend": backend,
                "locomotion_policy_path": str(humanoidverse_policy_onnx) if backend == "humanoidverse" else str(LUCKY_WALKER_ONNX),
                "locomotion_action_dim": int(locomotion_joint_map.action_dim),
                "stopped_at_turning_point": True,
                "transport_skipped": True,
                "freeze_upper_during_transport": bool(freeze_upper_during_transport),
                "vision_detection_used": True,
                "vision_enabled": bool(vision_result.vision_enabled),
                "detection_mode": str(vision_result.detection_mode),
                "camera_source": str(vision_result.camera_source),
                "box_detected": bool(vision_result.box_detected),
                "detection_confidence": float(vision_result.detection_confidence),
                "distance_to_box_m": vision_result.distance_to_box_m,
                "obstacle_detected": bool(vision_result.obstacle_detected),
                "obstacle_distance_m": vision_result.obstacle_distance_m,
                "obstacle_lateral_bias": float(vision_result.obstacle_lateral_bias),
                "rgb_fps": vision_result.rgb_fps,
                "depth_fps": vision_result.depth_fps,
                "box_bbox_px": vision_result.box_bbox_px,
                "box_center_px": vision_result.box_center_px,
                "box_position_world": list(vision_result.box_position_world) if vision_result.box_position_world is not None else None,
                "frame_processing_time_s": float(vision_result.frame_processing_time_s),
                "frame_processing_time_ok": bool(vision_result.frame_processing_time_ok),
                "detection_success": bool(vision_result.detection_success),
                "vision_debug_image_path": vision_result.debug_image_path,
                "show_vision_window": bool(show_vision_window_active),
                "show_depth_window": bool(show_depth_window_active),
                "show_route_map": bool(show_route_map_active),
                "goal_marker_active": bool(goal_marker_active),
                "obstacle_clearance_target_m": float(obstacle_clearance_target_m),
                "obstacle_avoidance_trigger_count": int(obstacle_avoidance_trigger_count),
                "min_obstacle_distance_m": float(min_obstacle_distance_m if min_obstacle_distance_m < 1e8 else 0.0),
                "last_obstacle_lateral_bias": float(last_obstacle_lateral_bias),
                "min_pelvis_surrounding_clearance_m": float(min_pelvis_surrounding_clearance_m if min_pelvis_surrounding_clearance_m < 1e8 else 0.0),
                "lidar_replan_count": int(lidar_replan_count),
                "transport_stall_recovery_count": int(transport_stall_recovery_count),
                "planned_route_waypoints_count": int(planned_route_waypoints_count),
                "planned_route_length_m": float(planned_route_length_m),
                "forward_only_path_following": True,
                "upper_body_static_transport": bool(max_upper_body_joint_drift_rad <= UPPER_BODY_STATIC_DRIFT_TOLERANCE_RAD),
                "max_upper_body_joint_drift_rad": float(max_upper_body_joint_drift_rad),
                "box_locked_to_carry_frame": bool(max_box_attach_error_m <= 0.03),
                "box_remained_attached": bool(max_box_attach_error_m <= 0.03),
                "max_box_attach_error_m": float(max_box_attach_error_m),
                "approach_start_distance_from_table_m": float(APPROACH_DISTANCE_FROM_TABLE_M),
                "robot_table_standoff_m": float(robot_table_standoff_m),
                "walk_to_pick_table_success": bool(walk_success),
                "reach_to_box_success": bool(reach_success),
                "stance_foot_lead_m": float(stance_foot_lead_m),
                "one_foot_ahead_for_reach": bool(one_foot_ahead_for_reach),
                "turn_uses_walker_steps": bool(turn_uses_walker_steps),
                "turn_step_count_target": int(turn_step_count),
                "turn_visible_step_count": int(turn_visible_step_count),
                "carry_turn_actual_step_count": int(carry_turn_actual_step_count),
                "carry_transport_actual_step_count": int(carry_transport_actual_step_count),
                "total_carry_actual_step_count": int(total_carry_actual_step_count),
                "turn_root_yaw_change_rad": float(right_turn_yaw_change),
                "turn_root_teleport_used": bool(turn_root_teleport_used),
                "turn_success": bool(turn_success),
                "transport_uses_walker_steps": bool(transport_uses_walker_steps),
                "transport_root_teleporting_used": bool(transport_root_teleporting_used),
                "transport_visible_step_count": int(transport_visible_step_count),
                "min_pick_table_clearance_m": float(min_pick_table_clearance_m if min_pick_table_clearance_m < 1e8 else 0.0),
                "min_second_table_clearance_m": float(min_second_table_clearance_m if min_second_table_clearance_m < 1e8 else 0.0),
                "safety_cmd_clamps_count": int(safety_cmd_clamps_count),
                "collision_recovery_count": int(collision_recovery_count),
                "right_turn_yaw_change_deg": float(np.degrees(right_turn_yaw_change)),
                "right_turn_success": bool(right_turn_success),
                "max_hand_table_contact_count": int(max_hand_table_contact_count),
                "max_robot_table_contact_count": int(max_robot_table_contact_count),
                "robot_fell": bool(robot_fell),
                "actual_floor_contact_detected": bool(floor_contact_seen),
                "payload_mpc_wbc_enabled": bool(last_payload_mpc_metrics.get("payload_mpc_wbc_enabled", False)),
                **posture_metrics,
                "success": bool(turn_stage_success),
                "failure_reason": None if turn_stage_success else "turn_stage_validation_failed",
            }

        second_table_xy = np.asarray(second_table_metrics["second_table_xy"], dtype=float)
        second_table_yaw = float(second_table_metrics["second_table_yaw_rad"])
        table_place_half_x = max(0.05, float(second_table_half_x) - 0.04)
        table_place_half_y = max(0.05, float(second_table_half_y) - 0.04)
        table_place_z = float(second_table_metrics["second_table_surface_z"]) + 0.5 * float(box_size_z) + 0.003
        release_xy_world = np.asarray([float(second_table_xy[0]), float(second_table_xy[1])], dtype=float)
        release_xy_table_local = np.asarray([0.0, 0.0], dtype=float)

        def _place_xy_on_table_in_front(root_xy: np.ndarray, robot_yaw: float) -> np.ndarray:
            """Drop point on the table directly in front of the robot/camera heading."""
            heading = _heading_from_yaw(float(robot_yaw))
            ahead_xy = np.asarray(root_xy + heading * float(PLACE_POINT_AHEAD_OF_ROBOT_M), dtype=float)
            return _clamp_xy_to_oriented_rect(
                ahead_xy,
                second_table_xy,
                second_table_yaw,
                table_place_half_x,
                table_place_half_y,
            )

        def _desired_place_yaw(root_xy: np.ndarray, place_xy: np.ndarray | None = None) -> float:
            """Face the placement point (not the pick-up heading or table edge)."""
            if place_xy is None:
                place_xy = _place_xy_on_table_in_front(root_xy, _imu_torso_yaw_rad(model, data))
            rel = np.asarray(np.asarray(place_xy, dtype=float) - np.asarray(root_xy, dtype=float), dtype=float)
            n = float(np.linalg.norm(rel))
            if n < 1e-6:
                to_table = np.asarray(second_table_xy - np.asarray(root_xy, dtype=float), dtype=float)
                n = float(np.linalg.norm(to_table))
                if n < 1e-6:
                    return float(_imu_torso_yaw_rad(model, data))
                rel = to_table / n
            return float(np.arctan2(float(rel[1]), float(rel[0])))

        def _standoff_goal_from_root(root_xy: np.ndarray) -> np.ndarray:
            """Stand so the placement point sits straight in front of the robot."""
            imu_yaw = _imu_torso_yaw_rad(model, data)
            place_xy = _place_xy_on_table_in_front(root_xy, imu_yaw)
            face_yaw = _desired_place_yaw(root_xy, place_xy)
            heading = _heading_from_yaw(face_yaw)
            return np.asarray(place_xy - heading * float(PLACE_ANYWHERE_STANDOFF_M), dtype=float)

        def _update_place_guidance_marker(
            *,
            ready_to_release: bool = False,
            release_countdown_s: float | None = None,
        ) -> np.ndarray:
            nonlocal goal_marker_active, release_xy_world, release_xy_table_local
            root_xy_now = np.asarray(data.qpos[:2], dtype=float)
            release_xy_world = _place_xy_on_table_in_front(root_xy_now, _imu_torso_yaw_rad(model, data))
            marker_pos = np.asarray([float(release_xy_world[0]), float(release_xy_world[1]), float(table_place_z + 0.08)], dtype=float)
            goal_marker_active = _set_goal_marker_pose(
                model,
                data,
                marker_pos,
                ready_to_release=bool(ready_to_release),
                release_countdown_s=release_countdown_s,
            )
            p = np.asarray(release_xy_world, dtype=float) - np.asarray(second_table_xy, dtype=float)
            c = float(np.cos(-float(second_table_yaw)))
            s = float(np.sin(-float(second_table_yaw)))
            release_xy_table_local = np.asarray(
                [float(c * p[0] - s * p[1]), float(s * p[0] + c * p[1])],
                dtype=float,
            )
            return release_xy_world

        second_stand_xy = _standoff_goal_from_root(np.asarray(data.qpos[:2], dtype=float))
        _update_place_guidance_marker()
        final_box_target = np.asarray([float(release_xy_world[0]), float(release_xy_world[1]), table_place_z], dtype=float)
        travel_start_xy = np.asarray(data.qpos[:2], dtype=float).copy()
        route_points = _plan_shortest_path_astar(
            start_xy=travel_start_xy,
            goal_xy=second_stand_xy,
            obstacles=[
                (pick_table_xy, 0.0, pick_table_half_x, pick_table_half_y),
                (
                    np.asarray(second_table_metrics["second_table_xy"], dtype=float),
                    float(second_table_metrics["second_table_yaw_rad"]),
                    second_table_half_x + goal_table_approach_clearance_m,
                    second_table_half_y + goal_table_approach_clearance_m,
                ),
            ],
            clearance_m=float(obstacle_clearance_target_m),
            resolution_m=0.12,
        )
        planned_route_waypoints_count = int(len(route_points))
        prev_pt = np.asarray(travel_start_xy, dtype=float)
        planned_route_length_m = 0.0
        for wp in route_points:
            planned_route_length_m += float(np.linalg.norm(np.asarray(wp, dtype=float) - prev_pt))
            prev_pt = np.asarray(wp, dtype=float)
        route_start_time = float(data.time)
        enter_phase("WALK_TO_SECOND_TABLE_WITH_BOX")
        reached_all_route_points = True
        for route_target_xy in route_points:
            _update_place_guidance_marker()
            route_target = np.asarray(route_target_xy, dtype=float)
            route_dist = float(np.linalg.norm(route_target - np.asarray(data.qpos[:2], dtype=float)))
            seg_duration = max(2.0, min(5.0, 0.5 * SECOND_TABLE_APPROACH_DURATION_S + 0.4 * route_dist))
            root_now = np.asarray(data.qpos[:2], dtype=float)
            place_xy_now = _place_xy_on_table_in_front(root_now, _imu_torso_yaw_rad(model, data))
            _, _, reached_seg = walk_to_waypoint_with_steps(
                target_xy=np.asarray(route_target_xy, dtype=float),
                target_yaw=float(_desired_place_yaw(root_now, place_xy_now)),
                max_duration=float(seg_duration),
                carried_box_offset=carry_offset_from_hands,
                contact_cb=record_hand_table_contact,
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
            )
            reached_all_route_points = bool(reached_all_route_points and reached_seg)
        if assist_final_alignment:
            second_stand_xy = _standoff_goal_from_root(np.asarray(data.qpos[:2], dtype=float))
            _set_base_xy(data, second_stand_xy)
            _set_root_yaw(
                data,
                _desired_place_yaw(
                    second_stand_xy,
                    _place_xy_on_table_in_front(second_stand_xy, _imu_torso_yaw_rad(model, data)),
                ),
            )
            transport_root_teleporting_used = True
            mujoco.mj_forward(model, data)
        transport_stop_start = float(data.time)
        while float(data.time) - transport_stop_start < 0.6:
            step_walk(
                np.zeros(3, dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                phase_name="WALK_TO_SECOND_TABLE_WITH_BOX",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("WALK_TO_SECOND_TABLE_WITH_BOX")
            _update_place_guidance_marker()
        travel_duration_s = max(1e-6, float(data.time) - route_start_time)
        travel_progress_m = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - travel_start_xy))
        average_transport_speed_mps = float(travel_progress_m / travel_duration_s)
        second_stand_xy = _standoff_goal_from_root(np.asarray(data.qpos[:2], dtype=float))
        pre_second_table_alignment_error_m = float(np.linalg.norm(second_stand_xy - np.asarray(data.qpos[:2], dtype=float)))
        attached_box_center = _palm_center(model, data) + carry_offset_from_hands
        transport_final_alignment_assisted = bool(assist_final_alignment and transport_root_teleporting_used)
        upper_body_unfrozen_for_place = bool(freeze_upper_during_transport)
        enter_phase("PRE_PLACE_ALIGN_WITH_TABLE")
        root_xy_place = np.asarray(data.qpos[:2], dtype=float)
        table_clearance_at_place = max(
            0.0,
            float(
                _distance_to_oriented_rect(
                    root_xy_place,
                    second_table_xy,
                    second_table_yaw,
                    float(second_table_half_x),
                    float(second_table_half_y),
                )
            ),
        )
        release_gate_reached = bool(table_clearance_at_place <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M)
        pre_place_align_start = float(data.time)
        pre_place_alignment_success = False
        place_yaw_error_rad = 1e6
        while float(data.time) - pre_place_align_start < 2.5:
            _update_place_guidance_marker()
            root_xy_align = np.asarray(data.qpos[:2], dtype=float)
            place_xy_align = _place_xy_on_table_in_front(root_xy_align, _imu_torso_yaw_rad(model, data))
            desired_place_yaw = _desired_place_yaw(root_xy_align, place_xy_align)
            imu_yaw = _imu_torso_yaw_rad(model, data)
            place_yaw_error_rad = abs(float(_wrap_pi(float(desired_place_yaw) - imu_yaw)))
            if place_yaw_error_rad <= 0.28:
                pre_place_alignment_success = True
                break
            step_walk(
                np.array([0.0, 0.0, float(np.clip(0.9 * _wrap_pi(float(desired_place_yaw) - imu_yaw), -0.30, 0.30))], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                phase_name="WALK_TO_SECOND_TABLE_WITH_BOX",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("PRE_PLACE_ALIGN_WITH_TABLE")

        place_pose_ready = False
        place_wait_loops = 0
        place_standoff_error_m = 1e6
        place_pose_forced_ready = False
        place_left_err = 1e6
        place_right_err = 1e6
        table_in_front_for_place = False
        release_gate_enter_time_s: float | None = None
        place_release_gate_wait_s = 0.0
        near_table_enter_time_s: float | None = None
        place_targets: dict[str, float] = {}
        open_finger_targets = finger.targets_for_mode("open")
        place_freeze_fingers = None
        place_approach_start_s = float(data.time)

        def _refresh_place_targets() -> None:
            nonlocal place_targets, place_left_err, place_right_err, final_box_target, release_xy_world
            root_xy_now = np.asarray(data.qpos[:2], dtype=float)
            release_xy_world = _place_xy_on_table_in_front(root_xy_now, _imu_torso_yaw_rad(model, data))
            final_box_target = np.asarray([float(release_xy_world[0]), float(release_xy_world[1]), table_place_z], dtype=float)
            place_palm_center = np.asarray(
                [
                    float(final_box_target[0] - carry_offset_from_hands[0]),
                    float(final_box_target[1] - carry_offset_from_hands[1]),
                    float(table_place_z - carry_offset_from_hands[2]),
                ],
                dtype=float,
            )
            lateral_axis = _palm_lateral_axis(model, data)
            place_half_width = 0.5 * float(_posture_metrics(model, data, LOCKED_CARRY_ARM_TARGETS)["palm_pair_width_m"])
            place_left = place_palm_center + lateral_axis * place_half_width
            place_right = place_palm_center - lateral_axis * place_half_width
            place_targets, place_left_err, place_right_err = _solve_arm_targets(
                model,
                data,
                left_target=place_left,
                right_target=place_right,
            )

        _refresh_place_targets()
        while not place_pose_ready:
            if release_gate_reached:
                break
            _refresh_place_targets()
            root_xy = np.asarray(data.qpos[:2], dtype=float)
            table_clearance_m = max(
                0.0,
                float(
                    _distance_to_oriented_rect(
                        root_xy,
                        second_table_xy,
                        second_table_yaw,
                        float(second_table_half_x),
                        float(second_table_half_y),
                    )
                ),
            )
            place_standoff_error_m = abs(float(table_clearance_m) - float(PLACE_ANYWHERE_STANDOFF_M))
            imu_yaw = _imu_torso_yaw_rad(model, data)
            place_xy_now = _place_xy_on_table_in_front(root_xy, imu_yaw)
            heading = _heading_from_yaw(imu_yaw)
            to_place = np.asarray(place_xy_now - root_xy, dtype=float)
            to_place_norm = float(np.linalg.norm(to_place))
            desired_place_yaw = _desired_place_yaw(root_xy, place_xy_now)
            place_yaw_error_rad = abs(float(_wrap_pi(float(desired_place_yaw) - imu_yaw)))
            facing_place_point = bool(to_place_norm > 1e-6 and float(np.dot(heading, to_place / to_place_norm)) > 0.65)
            facing_place_yaw = bool(place_yaw_error_rad <= 0.35)
            table_in_front_for_place = bool(facing_place_point and facing_place_yaw)
            ik_error = max(float(place_left_err), float(place_right_err))
            # In-front and within standoff band (~0.2m, tolerances for nav error).
            release_gate_reached = bool(
                table_in_front_for_place
                and float(PLACE_ANYWHERE_STANDOFF_M) - float(PLACE_ANYWHERE_STANDOFF_TOLERANCE_M)
                <= table_clearance_m
                <= float(PLACE_ANYWHERE_STANDOFF_M) + float(PLACE_ANYWHERE_STANDOFF_TOLERANCE_M)
            )
            if (not release_gate_reached) and table_clearance_m <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M:
                if near_table_enter_time_s is None:
                    near_table_enter_time_s = float(data.time)
                elif float(data.time) - float(near_table_enter_time_s) >= 1.5:
                    release_gate_reached = True
                    place_pose_forced_ready = True
            else:
                near_table_enter_time_s = None
            if float(data.time) - place_approach_start_s > 25.0:
                if verbose:
                    print("PLACE_APPROACH: timeout, forcing place gate if near table", flush=True)
                if table_clearance_m <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M + 0.20:
                    release_gate_reached = True
                    place_pose_forced_ready = True
                break
            if place_wait_loops >= PLACE_APPROACH_MAX_WAIT_LOOPS:
                if table_clearance_m <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M + 0.25:
                    release_gate_reached = True
                    place_pose_forced_ready = True
                break
            if release_gate_reached:
                if release_gate_enter_time_s is None:
                    release_gate_enter_time_s = float(data.time)
                place_release_gate_wait_s = float(data.time - release_gate_enter_time_s)
            else:
                release_gate_enter_time_s = None
                place_release_gate_wait_s = 0.0
            marker_countdown = (
                max(0.0, float(PLACE_RELEASE_GATE_HOLD_S) - float(place_release_gate_wait_s))
                if release_gate_reached and place_release_gate_wait_s < PLACE_RELEASE_GATE_HOLD_S
                else None
            )
            _update_place_guidance_marker(
                ready_to_release=bool(release_gate_reached),
                release_countdown_s=marker_countdown,
            )
            # Start lower/release after 2s at ~0.2m in front of table (no strict IK gate).
            place_pose_ready = bool(
                release_gate_reached and place_release_gate_wait_s >= PLACE_RELEASE_GATE_HOLD_S
            )
            if place_pose_ready:
                break
            place_wait_loops += 1
            if release_gate_reached:
                break
            second_stand_xy = _standoff_goal_from_root(np.asarray(data.qpos[:2], dtype=float))
            approach_dist = float(np.linalg.norm(second_stand_xy - np.asarray(data.qpos[:2], dtype=float)))
            approach_duration = 0.8 if table_clearance_m <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M else max(1.0, min(2.5, approach_dist))
            place_xy_approach = _place_xy_on_table_in_front(root_xy, imu_yaw)
            _, _, _ = walk_to_waypoint_with_steps(
                target_xy=second_stand_xy,
                target_yaw=float(_desired_place_yaw(second_stand_xy, place_xy_approach)),
                max_duration=float(approach_duration),
                carried_box_offset=carry_offset_from_hands,
                contact_cb=record_hand_table_contact,
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=place_freeze_fingers,
            )
            attached_box_center = _palm_center(model, data) + carry_offset_from_hands
            _set_box_pose_from_carry_frame(model, data, attached_box_center, _palm_lateral_axis(model, data))

        if not release_gate_reached:
            root_xy_final = np.asarray(data.qpos[:2], dtype=float)
            final_clearance = max(
                0.0,
                float(
                    _distance_to_oriented_rect(
                        root_xy_final,
                        second_table_xy,
                        second_table_yaw,
                        float(second_table_half_x),
                        float(second_table_half_y),
                    )
                ),
            )
            if final_clearance <= PLACE_ANYWHERE_MAX_APPROACH_CLEARANCE_M + 0.12:
                release_gate_reached = True
                place_pose_forced_ready = True

        # Real 2s sim-time hold at release gate (not loop-count based).
        if release_gate_reached:
            if verbose:
                print(
                    f"PLACE_GATE: holding {PLACE_RELEASE_GATE_HOLD_S:.1f}s facing placement point",
                    flush=True,
                )
            gate_hold_start = float(data.time)
            while float(data.time) - gate_hold_start < PLACE_RELEASE_GATE_HOLD_S:
                _update_place_guidance_marker(
                    ready_to_release=True,
                    release_countdown_s=max(0.0, PLACE_RELEASE_GATE_HOLD_S - (float(data.time) - gate_hold_start)),
                )
                root_xy_gate = np.asarray(data.qpos[:2], dtype=float)
                place_xy_gate = _place_xy_on_table_in_front(root_xy_gate, _imu_torso_yaw_rad(model, data))
                gate_yaw = _desired_place_yaw(root_xy_gate, place_xy_gate)
                gate_yaw_err = float(_wrap_pi(float(gate_yaw) - _imu_torso_yaw_rad(model, data)))
                step_walk(
                    np.array([0.0, 0.0, float(np.clip(0.8 * gate_yaw_err, -0.22, 0.22))], dtype=np.float32),
                    hold_masked_arms_at_default=False,
                    carried_box_offset=carry_offset_from_hands,
                    pin_carried_box=True,
                    phase_name="PRE_PLACE_ALIGN_WITH_TABLE",
                    freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                    freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                    allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
                )
                record_hand_table_contact("PRE_PLACE_ALIGN_WITH_TABLE")
            place_release_gate_wait_s = float(data.time - gate_hold_start)
            place_pose_ready = True

        # Final face-table correction (never re-freeze fingers to grasp here).
        enter_phase("PRE_PLACE_FINAL_FACE_TABLE")
        final_face_start = float(data.time)
        while float(data.time) - final_face_start < 2.5:
            _update_place_guidance_marker(ready_to_release=True)
            root_xy_face = np.asarray(data.qpos[:2], dtype=float)
            place_xy_face = _place_xy_on_table_in_front(root_xy_face, _imu_torso_yaw_rad(model, data))
            desired_place_yaw = _desired_place_yaw(root_xy_face, place_xy_face)
            imu_yaw = _imu_torso_yaw_rad(model, data)
            yaw_err = float(_wrap_pi(float(desired_place_yaw) - imu_yaw))
            place_yaw_error_rad = abs(yaw_err)
            if place_yaw_error_rad <= 0.22:
                break
            step_walk(
                np.array([0.0, 0.0, float(np.clip(0.9 * yaw_err, -0.30, 0.30))], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                pin_carried_box=True,
                phase_name="PRE_PLACE_ALIGN_WITH_TABLE",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("PRE_PLACE_FINAL_FACE_TABLE")

        # Must face the in-front placement point before lowering (not the pick-up heading).
        face_before_lower_start = float(data.time)
        while float(data.time) - face_before_lower_start < 4.0:
            root_xy_lower = np.asarray(data.qpos[:2], dtype=float)
            place_xy_lower = _place_xy_on_table_in_front(root_xy_lower, _imu_torso_yaw_rad(model, data))
            desired_lower_yaw = _desired_place_yaw(root_xy_lower, place_xy_lower)
            imu_yaw_lower = _imu_torso_yaw_rad(model, data)
            yaw_err_lower = float(_wrap_pi(float(desired_lower_yaw) - imu_yaw_lower))
            place_yaw_error_rad = abs(yaw_err_lower)
            if place_yaw_error_rad <= 0.25:
                break
            step_walk(
                np.array([0.0, 0.0, float(np.clip(0.95 * yaw_err_lower, -0.32, 0.32))], dtype=np.float32),
                hold_masked_arms_at_default=False,
                carried_box_offset=carry_offset_from_hands,
                pin_carried_box=True,
                phase_name="PRE_PLACE_ALIGN_WITH_TABLE",
                freeze_upper_targets=LOCKED_CARRY_ARM_TARGETS if freeze_upper_during_transport else None,
                freeze_finger_targets=finger_targets if freeze_upper_during_transport else None,
                allow_payload_mpc_waist=not bool(freeze_upper_during_transport),
            )
            record_hand_table_contact("PRE_PLACE_ALIGN_WITH_TABLE")

        _refresh_place_targets()
        _update_place_guidance_marker(ready_to_release=True)
        enter_phase("LOWER_BOX_TO_SECOND_TABLE")
        if verbose:
            print("PLACE_LOWER: lowering box with hands onto table", flush=True)
        box_kinematically_attached = True
        place_start_snapshot = _snapshot(model, data, LOCKED_CARRY_UPPER_BODY_TARGETS)
        place_start = float(data.time)
        while float(data.time) - place_start < place_lower_duration_s:
            raw_u = (float(data.time) - place_start) / max(place_lower_duration_s, 1e-6)
            u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
            _interpolate_qpos(model, data, place_start_snapshot, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **place_targets}, u)
            _apply_joint_ctrl(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **place_targets})
            finger_pose = open_finger_targets if u >= 0.92 else finger_targets
            _set_joint_qpos(model, data, finger_pose)
            _apply_joint_ctrl(model, data, finger_pose)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            attached_box_center = _palm_center(model, data) + carry_offset_from_hands
            _set_box_pose_from_carry_frame(model, data, attached_box_center, _palm_lateral_axis(model, data))
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            record_hand_table_contact("LOWER_BOX_TO_SECOND_TABLE")
            sync()
        # Box stayed in the hands for the full lower; place at the lowered hand pose (no slide to marker).
        landed_box_center = np.asarray(_palm_center(model, data) + carry_offset_from_hands, dtype=float)
        final_box_target = np.asarray(
            [
                float(landed_box_center[0]),
                float(landed_box_center[1]),
                float(max(table_place_z, float(landed_box_center[2]))),
            ],
            dtype=float,
        )
        _set_box_pose(model, data, final_box_target)
        box_kinematically_attached = False
        mujoco.mj_forward(model, data)
        if verbose:
            print(
                "PLACE_RELEASE: box on table, opening fingers for "
                f"{release_duration_s:.1f}s",
                flush=True,
            )

        enter_phase("RELEASE_BOX_ON_SECOND_TABLE")
        finger.reset(open_finger_targets)
        release_start = float(data.time)
        release_start_snapshot = _snapshot(model, data, {**LOCKED_CARRY_UPPER_BODY_TARGETS, **place_targets})
        while float(data.time) - release_start < release_duration_s:
            rel_u = (float(data.time) - release_start) / max(release_duration_s, 1e-6)
            rel_u = float(np.clip(rel_u, 0.0, 1.0))
            # Ease hands away from the box after it is resting on the table.
            retract_targets = {
                **LOCKED_CARRY_UPPER_BODY_TARGETS,
                **{
                    k: float((1.0 - 0.30 * rel_u) * release_start_snapshot[k] + 0.30 * rel_u * LOCKED_CARRY_UPPER_BODY_TARGETS[k])
                    for k in place_targets
                },
            }
            _set_joint_qpos(model, data, retract_targets)
            _apply_joint_ctrl(model, data, retract_targets)
            _set_joint_qpos(model, data, open_finger_targets)
            _apply_joint_ctrl(model, data, open_finger_targets)
            _stabilize_manipulation_base(data, pelvis_z=initial_pelvis_z)
            _set_box_pose(model, data, final_box_target)
            data.time += float(model.opt.timestep)
            mujoco.mj_forward(model, data)
            record_hand_table_contact("RELEASE_BOX_ON_SECOND_TABLE")
            sync()

        box_kinematically_attached = False
        root_xy_end = np.asarray(data.qpos[:2], dtype=float)
        place_xy_end = _place_xy_on_table_in_front(root_xy_end, _imu_torso_yaw_rad(model, data))
        place_yaw_error_rad = abs(float(_wrap_pi(float(_desired_place_yaw(root_xy_end, place_xy_end)) - _imu_torso_yaw_rad(model, data))))

        posture_metrics = carry_posture_metrics_at_chest
        final_box = np.asarray(detect_box_ground_truth(model, data).box_position_world, dtype=float)
        box_lift_height = float(chest_box_center[2] - initial_box[2])
        box_horizontal_shift_during_table_lift_m = 0.0
        box_horizontal_shift_to_chest_m = float(np.linalg.norm(chest_box_center[:2] - initial_box[:2]))
        table_xy_error = float(
            max(
                0.0,
                _distance_to_oriented_rect(
                    final_box[:2],
                    second_table_xy,
                    float(second_table_metrics["second_table_yaw_rad"]),
                    table_place_half_x,
                    table_place_half_y,
                ),
            )
        )
        table_z_error = float(abs(float(final_box[2]) - float(table_place_z)))
        box_place_error_m = float(np.hypot(table_xy_error, table_z_error))
        box_placed_on_second_table = bool(table_xy_error <= 0.03 and table_z_error <= 0.08)
        final_table_clearance_m = max(
            0.0,
            float(
                _distance_to_oriented_rect(
                    np.asarray(data.qpos[:2], dtype=float),
                    second_table_xy,
                    second_table_yaw,
                    float(second_table_half_x),
                    float(second_table_half_y),
                )
            ),
        )
        second_table_reached = bool(abs(float(final_table_clearance_m) - float(PLACE_ANYWHERE_STANDOFF_M)) <= PLACE_ANYWHERE_STANDOFF_TOLERANCE_M)
        right_turn_stable = bool(max_robot_table_contact_count == 0 and not robot_fell)
        turn_success = bool(turn_visible_step_count >= 4 and abs(float(right_turn_yaw_change)) >= 0.75)
        turn_stage_nominal = bool(turn_visible_step_count >= 2 and abs(float(right_turn_yaw_change)) >= 0.45)
        box_locked_to_carry_frame = bool(max_box_attach_error_m <= 0.03)
        upper_body_static_transport = bool(max_upper_body_joint_drift_rad <= UPPER_BODY_STATIC_DRIFT_TOLERANCE_RAD)
        box_lift_success = bool(box_lift_height > 0.15)
        box_place_success = bool(box_placed_on_second_table)
        travel_to_second_table_success = bool(
            (reached_all_route_points or second_table_reached or box_place_success)
            and travel_progress_m >= 0.10
            and transport_visible_step_count >= 4
            and not transport_root_teleporting_used
        )
        target_metrics_match = bool(
            abs(float(posture_metrics["avg_palm_height_m"]) - 1.0005638911825643) < 1e-6
            and abs(float(posture_metrics["avg_palm_forward_offset_m"]) - 0.14283506116030464) < 1e-6
            and float(posture_metrics["left_elbow_bent_rad"]) == -1.0472
            and float(posture_metrics["right_elbow_bent_rad"]) == -1.0472
        )
        success = bool(
            walk_success
            and reach_success
            and target_metrics_match
            and box_lift_success
            and max_box_motion_before_grip_m <= BOX_MOTION_BEFORE_GRIP_TOLERANCE_M
            and hands_touch_box_before_lift
            and all_three_fingers_touch_box_before_lift
            and box_horizontal_shift_during_table_lift_m <= 0.005
            and max_hand_box_penetration <= 0.003
            and max_hand_table_contact_count == 0
            and max_hand_table_penetration <= 0.0
            and max_robot_table_contact_count == 0
            and max_robot_table_penetration <= 0.0
            and floor_contact_seen
            and not robot_fell
            and turn_stage_nominal
            and right_turn_stable
            and travel_to_second_table_success
            and box_place_success
            and box_locked_to_carry_frame
            and upper_body_static_transport
            and max(float(place_left_err), float(place_right_err)) <= max(PLACE_IK_TOLERANCE_M, 0.22)
        )
        return {
            "table_pick_locked_carry": True,
            "phases": phases,
            "policy_loaded": True,
            "locomotion_backend": backend,
            "locomotion_policy_path": str(humanoidverse_policy_onnx) if backend == "humanoidverse" else str(LUCKY_WALKER_ONNX),
            "locomotion_action_dim": int(locomotion_joint_map.action_dim),
            "freeze_upper_during_transport": bool(freeze_upper_during_transport),
            "vision_detection_used": True,
            "vision_enabled": bool(vision_result.vision_enabled),
            "detection_mode": str(vision_result.detection_mode),
            "camera_source": str(vision_result.camera_source),
            "box_detected": bool(vision_result.box_detected),
            "detection_confidence": float(vision_result.detection_confidence),
            "distance_to_box_m": vision_result.distance_to_box_m,
            "obstacle_detected": bool(vision_result.obstacle_detected),
            "obstacle_distance_m": vision_result.obstacle_distance_m,
            "obstacle_lateral_bias": float(vision_result.obstacle_lateral_bias),
            "rgb_fps": vision_result.rgb_fps,
            "depth_fps": vision_result.depth_fps,
            "box_bbox_px": vision_result.box_bbox_px,
            "box_center_px": vision_result.box_center_px,
            "box_position_world": list(vision_result.box_position_world) if vision_result.box_position_world is not None else None,
            "frame_processing_time_s": float(vision_result.frame_processing_time_s),
            "frame_processing_time_ok": bool(vision_result.frame_processing_time_ok),
            "detection_success": bool(vision_result.detection_success),
            "vision_debug_image_path": vision_result.debug_image_path,
            "show_vision_window": bool(show_vision_window_active),
            "show_depth_window": bool(show_depth_window_active),
            "show_route_map": bool(show_route_map_active),
            "goal_marker_active": bool(goal_marker_active),
            "obstacle_clearance_target_m": float(obstacle_clearance_target_m),
            "obstacle_avoidance_trigger_count": int(obstacle_avoidance_trigger_count),
            "min_obstacle_distance_m": float(min_obstacle_distance_m if min_obstacle_distance_m < 1e8 else 0.0),
            "last_obstacle_lateral_bias": float(last_obstacle_lateral_bias),
            "min_pelvis_surrounding_clearance_m": float(min_pelvis_surrounding_clearance_m if min_pelvis_surrounding_clearance_m < 1e8 else 0.0),
            "lidar_replan_count": int(lidar_replan_count),
            "transport_stall_recovery_count": int(transport_stall_recovery_count),
            "planned_route_waypoints_count": int(planned_route_waypoints_count),
            "planned_route_length_m": float(planned_route_length_m),
            "forward_only_path_following": True,
            "upper_body_static_transport": bool(upper_body_static_transport),
            "max_upper_body_joint_drift_rad": float(max_upper_body_joint_drift_rad),
            "box_locked_to_carry_frame": bool(box_locked_to_carry_frame),
            "box_remained_attached": bool(box_locked_to_carry_frame),
            "max_box_attach_error_m": float(max_box_attach_error_m),
            "box_initially_on_table": bool(scene_metrics["box_on_table"]),
            "box_size_xyz_m": [float(box_size_x), float(box_size_y), float(box_size_z)],
            "box_mass_kg": float(box_mass_kg),
            **second_table_metrics,
            "approach_with_hands_in_carry_position": False,
            "approach_start_distance_from_table_m": float(APPROACH_DISTANCE_FROM_TABLE_M),
            "arm_raise_standoff_from_table_m": float(ARM_RAISE_STANDOFF_FROM_TABLE_EDGE_M),
            "distance_to_arm_raise_standoff_m": float(distance_to_arm_raise),
            "distance_to_pick_table_m": float(distance_to_pick),
            "pick_target_xy": [float(x) for x in pick],
            "pick_stand_tolerance_m": float(PICK_STAND_TOLERANCE_M),
            "pre_standoff_alignment_error_m": float(pre_standoff_alignment_error),
            "robot_table_standoff_m": float(robot_table_standoff_m),
            "robot_stood_back_from_table": bool(robot_table_standoff_m >= PICK_STANDOFF_FROM_TABLE_EDGE_M - 0.02),
            "walk_to_pick_table_success": bool(walk_success),
            "reach_to_box_success": bool(reach_success),
            "reach_left_error_m": float(reach_left_err),
            "reach_right_error_m": float(reach_right_err),
            "stance_foot_lead_m": float(stance_foot_lead_m),
            "one_foot_ahead_for_reach": bool(one_foot_ahead_for_reach),
            "payload_attach_mode": "fixed_to_hands_collision_enabled_after_table_pick",
            "attached_box_center_world": [float(x) for x in attached_box_center],
            "box_lift_height_m": float(box_lift_height),
            "box_motion_before_grip_m": float(max_box_motion_before_grip_m),
            "box_slid_into_hands_before_grip": bool(max_box_motion_before_grip_m > BOX_MOTION_BEFORE_GRIP_TOLERANCE_M),
            "hand_box_side_gap_at_grip_m": float(hand_box_side_gap_at_grip_m),
            "hands_touch_box_before_lift": bool(hands_touch_box_before_lift),
            "all_three_fingers_commanded_to_touch": bool(all_three_fingers_commanded_to_touch),
            "all_three_fingers_touch_box_before_lift": bool(all_three_fingers_touch_box_before_lift),
            **finger_contact,
            "box_fixed_to_hands_before_lift": True,
            "box_lift_driven_by_hands": True,
            "lift_duration_s": float(lift_duration_s),
            "chest_hold_duration_s": float(chest_hold_duration_s),
            "arm_raise_duration_s": float(arm_raise_duration_s),
            "reach_duration_s": float(reach_duration_s),
            "grip_duration_s": float(grip_duration_s),
            "demo_fast": bool(demo_fast),
            "hand_box_lift_synchronized": True,
            "box_horizontal_shift_during_table_lift_m": float(box_horizontal_shift_during_table_lift_m),
            "box_lifted_vertically_from_table": bool(box_horizontal_shift_during_table_lift_m <= 0.005),
            "box_horizontal_shift_to_chest_m": float(box_horizontal_shift_to_chest_m),
            "backward_carry_removed": True,
            "pick_table_turn_clearance_m": float(turn_clearance_m),
            "pick_table_clearance_duration_s": float(pick_table_clearance_duration_s),
            "right_turn_duration_s": float(turn_duration_s),
            "right_turn_yaw_change_rad": float(right_turn_yaw_change),
            "right_turn_yaw_change_deg": float(np.degrees(right_turn_yaw_change)),
            "right_turn_xy_drift_m": float(right_turn_xy_drift_m),
            "right_turn_stable": bool(right_turn_stable),
            "right_turn_assisted_in_place": False,
            "right_turn_assisted_finish": bool(assist_final_yaw_correction),
            "right_turn_success": bool(turn_success),
            "turn_uses_walker_steps": bool(turn_uses_walker_steps),
            "turn_step_count_target": int(turn_step_count),
            "turn_visible_step_count": int(turn_visible_step_count),
            "turn_root_yaw_change_rad": float(right_turn_yaw_change),
            "turn_root_teleport_used": bool(turn_root_teleport_used),
            "turn_success": bool(turn_success),
            "turn_elapsed_s": float(right_turn_elapsed_s),
            "travel_to_second_table_progress_m": float(travel_progress_m),
            "average_transport_speed_mps": float(average_transport_speed_mps),
            "pre_second_table_alignment_error_m": float(pre_second_table_alignment_error_m),
            "transport_final_alignment_assisted": bool(transport_final_alignment_assisted),
            "transport_uses_walker_steps": bool(transport_uses_walker_steps),
            "transport_root_teleporting_used": bool(transport_root_teleporting_used),
            "transport_visible_step_count": int(transport_visible_step_count),
            "carry_turn_actual_step_count": int(carry_turn_actual_step_count),
            "carry_transport_actual_step_count": int(carry_transport_actual_step_count),
            "total_carry_actual_step_count": int(total_carry_actual_step_count),
            "travel_to_second_table_success": bool(travel_to_second_table_success),
            "second_table_reached": bool(second_table_reached),
            "pre_place_alignment_success": bool(pre_place_alignment_success),
            "place_yaw_error_rad": float(place_yaw_error_rad),
            "place_mode": "table_region_anywhere",
            "place_required_table_standoff_m": float(PLACE_ANYWHERE_STANDOFF_M),
            "place_table_in_front": bool(table_in_front_for_place),
            "place_final_table_clearance_m": float(final_table_clearance_m),
            "place_release_gate_hold_s": float(PLACE_RELEASE_GATE_HOLD_S),
            "place_release_gate_wait_s": float(place_release_gate_wait_s),
            "box_kinematically_released": bool(not box_kinematically_attached),
            "place_release_xy_world": [float(release_xy_world[0]), float(release_xy_world[1])],
            "place_release_xy_table_local": [float(release_xy_table_local[0]), float(release_xy_table_local[1])],
            "place_pose_ready": bool(place_pose_ready),
            "place_wait_loops": int(place_wait_loops),
            "place_standoff_error_m": float(place_standoff_error_m),
            "place_pose_forced_ready": bool(place_pose_forced_ready),
            "place_left_error_m": float(place_left_err),
            "place_right_error_m": float(place_right_err),
            "box_place_error_m": float(box_place_error_m),
            "box_placed_on_second_table": bool(box_placed_on_second_table),
            "box_lift_success": bool(box_lift_success),
            "box_place_success": bool(box_place_success),
            "upper_body_unfrozen_for_place": bool(upper_body_unfrozen_for_place),
            "min_pick_table_clearance_m": float(min_pick_table_clearance_m if min_pick_table_clearance_m < 1e8 else 0.0),
            "min_second_table_clearance_m": float(min_second_table_clearance_m if min_second_table_clearance_m < 1e8 else 0.0),
            "safety_cmd_clamps_count": int(safety_cmd_clamps_count),
            "collision_recovery_count": int(collision_recovery_count),
            "final_box_center_world": [float(x) for x in final_box],
            "target_upper_body_metrics_preserved": bool(target_metrics_match),
            "max_hand_box_penetration_m": float(max_hand_box_penetration),
            "hand_box_impenetrable_visual": bool(max_hand_box_penetration <= 0.003),
            "max_hand_table_contact_count": int(max_hand_table_contact_count),
            "hand_table_contact_count_by_phase": dict(hand_table_contact_count_by_phase),
            "max_hand_table_penetration_m": float(max_hand_table_penetration),
            "hand_table_collision_free": bool(max_hand_table_contact_count == 0 and max_hand_table_penetration <= 0.0),
            "max_robot_table_contact_count": int(max_robot_table_contact_count),
            "robot_table_contact_count_by_phase": dict(robot_table_contact_count_by_phase),
            "robot_table_contact_pairs_by_phase": dict(robot_table_contact_pairs_by_phase),
            "max_robot_table_penetration_m": float(max_robot_table_penetration),
            "robot_table_collision_free": bool(max_robot_table_contact_count == 0 and max_robot_table_penetration <= 0.0),
            "robot_fell": bool(robot_fell),
            "actual_floor_contact_detected": bool(floor_contact_seen),
            "sim_time_s": float(data.time),
            **last_payload_mpc_metrics,
            **posture_metrics,
            "success": bool(success),
            "failure_reason": None if success else "pick_locked_carry_validation_failed",
        }

    try:
        if headless:
            result = run_sequence()
        else:
            viewer_mod = importlib.import_module("mujoco.viewer")
            with viewer_mod.launch_passive(model, data) as viewer:
                def _sync() -> None:
                    viewer.sync()
                    time.sleep(max(0.001, float(model.opt.timestep) / max(float(viewer_speed), 1e-3)))

                sync_after_step = _sync
                result = run_sequence()
                hold_start = time.time()
                while viewer.is_running() and time.time() - hold_start < float(post_done_hold):
                    _sync()
            result["viewer_closed"] = True
    finally:
        if vision_detector is not None:
            vision_detector.close()
        if show_route_map_active and cv2_available:
            try:
                import cv2  # type: ignore

                cv2.destroyWindow("G1 Route Map")
            except Exception:
                pass

    if verbose:
        print("----- g1_lucky_table_pick_locked_carry -----")
        for key, value in result.items():
            print(f"{key}: {value}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk to table and lift the box into the locked carry posture.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--viewer-speed", type=float, default=2.0)
    ap.add_argument("--box-mass-kg", type=float, default=0.01)
    ap.add_argument("--box-size-x", type=float, default=0.12)
    ap.add_argument("--box-size-y", type=float, default=0.33867925908248114)
    ap.add_argument("--box-size-z", type=float, default=0.10)
    ap.add_argument("--lift-duration-s", type=float, default=1.6)
    ap.add_argument("--turn-step-count", type=int, default=5)
    ap.add_argument("--turn-duration-s", type=float, default=4.0)
    ap.add_argument("--turn-yaw-cmd", type=float, default=-0.35)
    ap.add_argument("--turn-forward-cmd", type=float, default=0.0)
    ap.add_argument("--assist-final-yaw-correction", action="store_true")
    ap.add_argument("--assist-final-alignment", action="store_true")
    ap.add_argument("--post-done-hold", type=float, default=POST_DONE_HOLD_S)
    ap.add_argument("--demo-fast", action="store_true")
    ap.add_argument("--freeze-upper-during-transport", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--locomotion-backend", choices=["lucky", "humanoidverse"], default="lucky")
    ap.add_argument("--humanoidverse-policy-onnx", type=str, default=None)
    ap.add_argument("--stop-after-turning-point", action="store_true")
    ap.add_argument("--vision-enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-debug-image", action="store_true")
    ap.add_argument("--show-vision-window", action="store_true")
    ap.add_argument("--show-depth-window", action="store_true")
    ap.add_argument("--show-route-map", action="store_true")
    ap.add_argument("--vision-camera-mode", choices=["robot_mounted", "robot_egocentric", "scene"], default="robot_mounted")
    ap.add_argument("--vision-window-fps", type=float, default=10.0)
    ap.add_argument("--lidar-num-rays", type=int, default=72)
    ap.add_argument("--lidar-max-range-m", type=float, default=3.0)
    ns = ap.parse_args()
    run_g1_lucky_table_pick_locked_carry(
        headless=bool(ns.headless),
        timeout=float(ns.timeout),
        viewer_speed=float(ns.viewer_speed),
        box_mass_kg=float(ns.box_mass_kg),
        box_size_x=float(ns.box_size_x),
        box_size_y=float(ns.box_size_y),
        box_size_z=float(ns.box_size_z),
        lift_duration_s=float(ns.lift_duration_s),
        turn_step_count=int(ns.turn_step_count),
        turn_duration_s=float(ns.turn_duration_s),
        turn_yaw_cmd=float(ns.turn_yaw_cmd),
        turn_forward_cmd=float(ns.turn_forward_cmd),
        assist_final_yaw_correction=bool(ns.assist_final_yaw_correction),
        assist_final_alignment=bool(ns.assist_final_alignment),
        post_done_hold=float(ns.post_done_hold),
        demo_fast=bool(ns.demo_fast),
        freeze_upper_during_transport=bool(ns.freeze_upper_during_transport),
        locomotion_backend=str(ns.locomotion_backend),
        humanoidverse_policy_onnx=ns.humanoidverse_policy_onnx,
        stop_after_turning_point=bool(ns.stop_after_turning_point),
        vision_enabled=bool(ns.vision_enabled),
        save_debug_image=bool(ns.save_debug_image),
        show_vision_window=bool(ns.show_vision_window),
        show_depth_window=bool(ns.show_depth_window),
        show_route_map=bool(ns.show_route_map),
        vision_camera_mode=str(ns.vision_camera_mode),
        vision_window_fps=float(ns.vision_window_fps),
        lidar_num_rays=int(ns.lidar_num_rays),
        lidar_max_range_m=float(ns.lidar_max_range_m),
    )
    if not ns.headless:
        os._exit(0)


if __name__ == "__main__":
    main()
