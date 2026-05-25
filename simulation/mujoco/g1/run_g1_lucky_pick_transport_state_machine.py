#!/usr/bin/env python3
"""Lucky walker pick-transport milestone as an explicit stopped-state machine."""

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

from g1_arm_cartesian_reach import build_chain_metadata, frozen_qpos_snapshot, solve_position_only  # noqa: E402
from g1_dex3_finger_control import Dex3FingerController  # noqa: E402
from lucky_bridge.lucky_joint_map import action_to_joint_targets, build_lucky_joint_map  # noqa: E402
from lucky_bridge.lucky_paths import LUCKY_MODEL_CONFIG, LUCKY_SCENE_XML, LUCKY_WALKER_ONNX, missing_lucky_assets  # noqa: E402
from lucky_bridge.lucky_policy_loader import LuckyPolicyLoadError, load_lucky_config, load_walker_policy  # noqa: E402
from lucky_bridge.lucky_scene_setup import compute_pick_point, detect_box_ground_truth, setup_lucky_pick_scene  # noqa: E402
from run_g1_lucky_locomotion_validation import (  # noqa: E402
    PELVIS_DROP_M,
    ROLL_FALL_RAD,
    TORSO_FALL_RAD,
    _floor_contact_detected,
    _foot_center_xyz,
    _make_observation,
)
from run_g1_posture_hold import pelvis_roll_pitch_deg  # noqa: E402


PHASES = (
    "INIT_SCENE",
    "WALK_TO_PICK_POINT",
    "STOP_AND_STABILIZE",
    "ENTER_MANIPULATION_STANCE",
    "DETECT_BOX",
    "TWO_ARM_REACH",
    "GRIP_BOX",
    "DONE",
)
LEFT_ARM_CHAIN = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_CHAIN = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)
FINGER_JOINTS = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)


def _joint_qpos_adrs(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"missing joint {name}")
        out.append(int(model.jnt_qposadr[jid]))
    return out


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def _site_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = _site_id(model, name)
    if sid < 0:
        return np.zeros(3, dtype=float)
    return np.asarray(data.site_xpos[sid, :3], dtype=float).copy()


def _marker(
    scene: Any,
    *,
    geom_type: mujoco.mjtGeom,
    pos: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mat = np.eye(3, dtype=float).reshape(-1)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        int(geom_type),
        np.asarray(size, dtype=float),
        np.asarray(pos, dtype=float),
        mat,
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def _draw_debug_markers(
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    pick: np.ndarray,
    detection: Any | None,
    table_front_edge_x: float,
) -> None:
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    scene.ngeom = 0
    _marker(
        scene,
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=np.array([table_front_edge_x, 0.0, 0.74], dtype=float),
        size=np.array([0.006, 0.35, 0.006], dtype=float),
        rgba=np.array([1.0, 1.0, 0.0, 0.65], dtype=float),
    )
    support = _support_metrics(model, data)
    min_x = float(support["support_min_x"])
    max_x = float(support["support_max_x"])
    min_y = float(support["support_min_y"])
    max_y = float(support["support_max_y"])
    center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, 0.025], dtype=float)
    _marker(
        scene,
        geom_type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=center,
        size=np.array([(max_x - min_x) * 0.5, (max_y - min_y) * 0.5, 0.004], dtype=float),
        rgba=np.array([0.0, 0.8, 1.0, 0.18], dtype=float),
    )
    for pos, rgba in (
        (np.array([pick[0], pick[1], 0.04], dtype=float), np.array([0.0, 0.6, 1.0, 0.9])),
        (np.array([data.qpos[0], data.qpos[1], 0.08], dtype=float), np.array([1.0, 0.0, 1.0, 0.9])),
        (_foot_center_xyz(model, data, "left"), np.array([0.0, 0.4, 1.0, 0.75])),
        (_foot_center_xyz(model, data, "right"), np.array([1.0, 0.0, 0.4, 0.75])),
        (_site_pos(model, data, "left_palm"), np.array([0.0, 1.0, 0.0, 0.85])),
        (_site_pos(model, data, "right_palm"), np.array([1.0, 0.4, 0.0, 0.85])),
    ):
        _marker(
            scene,
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=pos,
            size=np.array([0.025, 0.025, 0.025], dtype=float),
            rgba=rgba,
        )
    if detection is not None and detection.box_detected:
        for pos, rgba in (
            (np.asarray(detection.box_position_world, dtype=float), np.array([1.0, 0.0, 0.0, 0.9])),
            (np.asarray(detection.left_grasp_target_world, dtype=float), np.array([0.0, 1.0, 0.4, 0.9])),
            (np.asarray(detection.right_grasp_target_world, dtype=float), np.array([1.0, 0.6, 0.0, 0.9])),
        ):
            _marker(
                scene,
                geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=pos,
                size=np.array([0.022, 0.022, 0.022], dtype=float),
                rgba=rgba,
            )


def _apply_body_targets(
    data: mujoco.MjData,
    joint_map: Any,
    target_pos: np.ndarray,
    *,
    arm_targets: dict[str, float] | None = None,
) -> None:
    arm_targets = arm_targets or {}
    for idx in joint_map.controlled_indices:
        name = joint_map.joint_names[idx]
        aid = joint_map.actuator_ids.get(name, -1)
        if aid >= 0:
            data.ctrl[int(aid)] = float(target_pos[idx])
    for idx in joint_map.arm_indices:
        name = joint_map.joint_names[idx]
        aid = joint_map.actuator_ids.get(name, -1)
        if aid >= 0:
            data.ctrl[int(aid)] = float(arm_targets.get(name, joint_map.default_joint_pos[idx]))


def _apply_finger_targets(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for name, val in targets.items():
        aid = _actuator_id(model, name)
        if aid >= 0:
            data.ctrl[aid] = float(val)


def _robot_fell(model: mujoco.MjModel, data: mujoco.MjData, initial_z: float) -> tuple[bool, float, float]:
    roll, pitch = pelvis_roll_pitch_deg(data, model)
    pitch_abs = abs(float(pitch)) if pitch is not None else 0.0
    roll_abs = abs(float(roll)) if roll is not None else 0.0
    fell = bool(pitch_abs > TORSO_FALL_RAD or roll_abs > ROLL_FALL_RAD or float(initial_z - data.qpos[2]) > PELVIS_DROP_M)
    return fell, pitch_abs, roll_abs


def _set_joint_target(targets: np.ndarray, joint_map: Any, joint_name: str, delta: float) -> None:
    if joint_name not in joint_map.joint_names:
        return
    idx = joint_map.joint_names.index(joint_name)
    targets[idx] = float(targets[idx] + float(delta))


def _support_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | bool]:
    left = _foot_center_xyz(model, data, "left")
    right = _foot_center_xyz(model, data, "right")
    half_len = 0.08
    half_w = 0.04
    min_x = float(min(left[0], right[0]) - half_len)
    max_x = float(max(left[0], right[0]) + half_len)
    min_y = float(min(left[1], right[1]) - half_w)
    max_y = float(max(left[1], right[1]) + half_w)
    com_xy = np.asarray(data.qpos[:2], dtype=float)
    margins = np.array(
        [
            com_xy[0] - min_x,
            max_x - com_xy[0],
            com_xy[1] - min_y,
            max_y - com_xy[1],
        ],
        dtype=float,
    )
    length = max_x - min_x
    width = max_y - min_y
    return {
        "support_polygon_length_m": float(length),
        "support_polygon_area_m2": float(length * width),
        "com_margin_m": float(np.min(margins)),
        "com_inside_support_polygon": bool(np.min(margins) >= 0.0),
        "support_min_x": float(min_x),
        "support_max_x": float(max_x),
        "support_min_y": float(min_y),
        "support_max_y": float(max_y),
    }


def _make_manipulation_stance_targets(
    joint_map: Any,
    base_targets: np.ndarray,
    *,
    stance_foot: str,
    forward_offset: float,
    width_offset: float,
    knee_bend: float,
    torso_pitch_bias: float = 0.0,
) -> np.ndarray:
    targets = np.asarray(base_targets, dtype=np.float32).copy()
    bend = 0.35 * float(knee_bend)
    fwd = 0.35 * float(forward_offset)
    width = 0.50 * float(width_offset)
    for side in ("left", "right"):
        _set_joint_target(targets, joint_map, f"{side}_knee_joint", bend)
        _set_joint_target(targets, joint_map, f"{side}_ankle_pitch_joint", -0.55 * bend)
        _set_joint_target(targets, joint_map, f"{side}_hip_pitch_joint", -0.45 * bend)
    lead = "left" if stance_foot == "left" else "right"
    trail = "right" if lead == "left" else "left"
    _set_joint_target(targets, joint_map, f"{lead}_hip_pitch_joint", -0.45 * bend - 0.90 * fwd)
    _set_joint_target(targets, joint_map, f"{trail}_hip_pitch_joint", -0.45 * bend + 0.35 * fwd)
    _set_joint_target(targets, joint_map, "left_hip_roll_joint", width)
    _set_joint_target(targets, joint_map, "right_hip_roll_joint", -width)
    if "waist_pitch_joint" in joint_map.joint_names:
        targets[joint_map.joint_names.index("waist_pitch_joint")] = float(torso_pitch_bias)
    if "waist_roll_joint" in joint_map.joint_names:
        targets[joint_map.joint_names.index("waist_roll_joint")] = 0.0
    return targets


def _preview_stance_candidate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: Any,
    *,
    base_targets: np.ndarray,
    stance_foot: str,
    forward_offset: float,
    width_offset: float,
    knee_bend: float,
    torso_pitch_bias: float,
    initial_pelvis_z: float,
    preview_duration: float = 0.45,
) -> tuple[dict[str, float | bool], np.ndarray]:
    saved_qpos = np.asarray(data.qpos, dtype=float).copy()
    saved_qvel = np.asarray(data.qvel, dtype=float).copy()
    saved_time = float(data.time)
    target = _make_manipulation_stance_targets(
        joint_map,
        base_targets,
        stance_foot=stance_foot,
        forward_offset=forward_offset,
        width_offset=width_offset,
        knee_bend=knee_bend,
        torso_pitch_bias=torso_pitch_bias,
    )
    start = float(data.time)
    fell = False
    max_pitch = 0.0
    max_roll = 0.0
    max_backward_pitch = 0.0
    while float(data.time) - start < float(preview_duration):
        _apply_body_targets(data, joint_map, target)
        mujoco.mj_step(model, data)
        step_fell, pitch_abs, roll_abs = _robot_fell(model, data, initial_pelvis_z)
        _, pitch_signed = pelvis_roll_pitch_deg(data, model)
        signed_pitch = float(pitch_signed) if pitch_signed is not None else 0.0
        max_backward_pitch = max(max_backward_pitch, max(0.0, -signed_pitch))
        max_pitch = max(max_pitch, pitch_abs)
        max_roll = max(max_roll, roll_abs)
        fell = fell or step_fell
    support = _support_metrics(model, data)
    pelvis_drop = float(initial_pelvis_z - data.qpos[2])
    np.copyto(data.qpos, saved_qpos)
    np.copyto(data.qvel, saved_qvel)
    data.time = saved_time
    mujoco.mj_forward(model, data)
    return (
        {
            "com_margin_m": float(support["com_margin_m"]),
            "support_polygon_length_m": float(support["support_polygon_length_m"]),
            "support_polygon_area_m2": float(support["support_polygon_area_m2"]),
            "torso_pitch_rad": float(max_pitch),
            "torso_roll_rad": float(max_roll),
            "max_backward_pitch_rad": float(max_backward_pitch),
            "pelvis_drop_m": float(pelvis_drop),
            "fell": bool(fell),
            "com_inside_support_polygon": bool(support["com_inside_support_polygon"]),
        },
        target,
    )


def _select_manipulation_stance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: Any,
    *,
    base_targets: np.ndarray,
    stance_foot: str,
    requested_forward_offset: float,
    requested_width_offset: float,
    requested_knee_bend: float,
    initial_pelvis_z: float,
    margin_before: float,
) -> dict[str, Any]:
    forward_values = sorted({0.0, 0.03, 0.06, 0.08, float(requested_forward_offset)})
    width_values = sorted({0.0, 0.02, 0.04, 0.06, float(requested_width_offset)})
    knee_values = sorted({0.04, 0.07, 0.10, float(requested_knee_bend)})
    pitch_values = (-0.06, -0.03, 0.0, 0.03, 0.06)
    best: dict[str, Any] | None = None
    tested = 0
    for fwd in forward_values:
        for width in width_values:
            for knee in knee_values:
                for pitch_bias in pitch_values:
                    tested += 1
                    metrics, target = _preview_stance_candidate(
                        model,
                        data,
                        joint_map,
                        base_targets=base_targets,
                        stance_foot=stance_foot,
                        forward_offset=float(fwd),
                        width_offset=float(width),
                        knee_bend=float(knee),
                        torso_pitch_bias=float(pitch_bias),
                        initial_pelvis_z=initial_pelvis_z,
                    )
                    rejected = bool(
                        metrics["fell"]
                        or float(metrics["com_margin_m"]) <= 0.01
                        or float(metrics["torso_pitch_rad"]) > 0.35
                        or float(metrics["pelvis_drop_m"]) > 0.08
                    )
                    score = float(metrics["com_margin_m"])
                    if not rejected and (best is None or score > float(best["metrics"]["com_margin_m"])):
                        best = {
                            "forward_offset": float(fwd),
                            "width_offset": float(width),
                            "knee_bend": float(knee),
                            "torso_pitch_bias": float(pitch_bias),
                            "metrics": metrics,
                            "target": target,
                        }
    if best is None or float(best["metrics"]["com_margin_m"]) < float(margin_before) - 0.005:
        return {
            "success": False,
            "fallback": "neutral_stand",
            "candidates_tested": tested,
            "target": np.asarray(base_targets, dtype=np.float32).copy(),
            "selected": {
                "forward_offset": 0.0,
                "width_offset": 0.0,
                "knee_bend": 0.0,
                "torso_pitch_bias": 0.0,
            },
            "metrics": _support_metrics(model, data),
        }
    return {
        "success": True,
        "fallback": "none",
        "candidates_tested": tested,
        "target": best["target"],
        "selected": {
            "forward_offset": best["forward_offset"],
            "width_offset": best["width_offset"],
            "knee_bend": best["knee_bend"],
            "torso_pitch_bias": best["torso_pitch_bias"],
        },
        "metrics": best["metrics"],
    }


def _body_or_geom_contact(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> bool:
    box_tokens = ("red_block", "red_cylinder", "red_cap")
    hand_tokens = (f"{side}_hand", f"{side}_palm")
    for cid in range(data.ncon):
        con = data.contact[cid]
        texts: list[str] = []
        for gid in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[gid])
            gname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            bname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            texts.append(f"{gname} {bname}")
        joined = " | ".join(texts)
        if any(t in joined for t in box_tokens) and any(t in joined for t in hand_tokens):
            return True
    return False


def _table_contact_counts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int | bool]:
    arm_count = 0
    hand_count = 0
    left_hand = False
    right_hand = False
    table_tokens = ("table", "table_top")
    arm_tokens = ("shoulder", "elbow", "wrist")
    hand_tokens = ("hand", "palm")
    for cid in range(data.ncon):
        con = data.contact[cid]
        texts: list[str] = []
        for gid in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[gid])
            gname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            bname = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            texts.append(f"{gname} {bname}")
        joined = " | ".join(texts)
        if not any(tok in joined for tok in table_tokens):
            continue
        if any(tok in joined for tok in arm_tokens):
            arm_count += 1
        if any(tok in joined for tok in hand_tokens):
            hand_count += 1
            left_hand = left_hand or "left_hand" in joined or "left_palm" in joined
            right_hand = right_hand or "right_hand" in joined or "right_palm" in joined
    return {
        "arm_table_contact_count": arm_count,
        "hand_table_contact_count": hand_count,
        "left_hand_table_contact": left_hand,
        "right_hand_table_contact": right_hand,
    }


def _finger_qpos_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            out[name] = float(data.qpos[int(model.jnt_qposadr[jid])])
    return out


REACH_PRESETS: dict[str, dict[str, float]] = {
    "neutral": {},
    "small_visible_reach": {
        "left_shoulder_pitch_joint": 0.1,
        "right_shoulder_pitch_joint": 0.1,
        "left_elbow_joint": 0.75,
        "right_elbow_joint": 0.75,
    },
    "two_arm_forward_carry_grasp": {
        "left_shoulder_pitch_joint": 0.08,
        "right_shoulder_pitch_joint": 0.08,
        "left_shoulder_roll_joint": 0.35,
        "right_shoulder_roll_joint": -0.35,
        "left_shoulder_yaw_joint": -0.05,
        "right_shoulder_yaw_joint": 0.05,
        "left_elbow_joint": 0.9,
        "right_elbow_joint": 0.9,
        "left_wrist_pitch_joint": -0.15,
        "right_wrist_pitch_joint": -0.15,
    },
    "low_forward": {
        "left_shoulder_pitch_joint": 0.8,
        "right_shoulder_pitch_joint": 0.8,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.45,
        "right_shoulder_roll_joint": -0.45,
    },
    "mid_forward": {
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_elbow_joint": 1.0,
        "right_elbow_joint": 1.0,
        "left_shoulder_roll_joint": 0.55,
        "right_shoulder_roll_joint": -0.55,
    },
    "wide_side_grasp": {
        "left_shoulder_pitch_joint": 0.1,
        "right_shoulder_pitch_joint": 0.1,
        "left_elbow_joint": 1.2,
        "right_elbow_joint": 1.2,
        "left_shoulder_roll_joint": 0.85,
        "right_shoulder_roll_joint": -0.85,
        "left_shoulder_yaw_joint": -0.3,
        "right_shoulder_yaw_joint": 0.3,
    },
    "narrow_front_grasp": {
        "left_shoulder_pitch_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "left_elbow_joint": 1.3,
        "right_elbow_joint": 1.3,
        "left_shoulder_roll_joint": 0.25,
        "right_shoulder_roll_joint": -0.25,
        "left_shoulder_yaw_joint": 0.2,
        "right_shoulder_yaw_joint": -0.2,
    },
}


def _preset_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    preset_name: str,
) -> dict[str, float]:
    preset = REACH_PRESETS.get(preset_name, {})
    targets: dict[str, float] = {}
    for name in (*LEFT_ARM_CHAIN, *RIGHT_ARM_CHAIN):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        targets[name] = float(preset.get(name, data.qpos[int(model.jnt_qposadr[jid])]))
    return targets


def _preview_arm_targets_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    arm_targets: dict[str, float],
    detection: Any,
) -> tuple[float, float, float]:
    saved_qpos = np.asarray(data.qpos, dtype=float).copy()
    for name, value in arm_targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(value)
    mujoco.mj_forward(model, data)
    left_err = float(np.linalg.norm(_site_pos(model, data, "left_palm") - np.asarray(detection.left_grasp_target_world, dtype=float)))
    right_err = float(np.linalg.norm(_site_pos(model, data, "right_palm") - np.asarray(detection.right_grasp_target_world, dtype=float)))
    np.copyto(data.qpos, saved_qpos)
    mujoco.mj_forward(model, data)
    return left_err, right_err, max(left_err, right_err)


def _select_reach_preset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    detection: Any,
    requested: str,
) -> tuple[str, dict[str, float], float, int]:
    names = list(REACH_PRESETS) if requested == "auto" else [requested]
    best_name = names[0]
    best_targets = _preset_targets(model, data, best_name)
    best_error = float("inf")
    visible_name: str | None = None
    visible_targets: dict[str, float] | None = None
    visible_error = float("inf")
    for name in names:
        targets = _preset_targets(model, data, name)
        _, _, err = _preview_arm_targets_error(model, data, arm_targets=targets, detection=detection)
        if err < best_error:
            best_name = name
            best_targets = targets
            best_error = err
        if name != "neutral" and err < visible_error:
            visible_name = name
            visible_targets = targets
            visible_error = err
    if requested == "auto" and visible_name is not None and visible_error <= best_error + 0.03:
        return visible_name, visible_targets or best_targets, float(visible_error), len(names)
    return best_name, best_targets, float(best_error), len(names)


def _arm_joint_mapping_diagnostics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    saved = np.asarray(data.qpos, dtype=float).copy()
    baseline_left = _site_pos(model, data, "left_palm")
    baseline_right = _site_pos(model, data, "right_palm")
    for name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] += 0.05
    mujoco.mj_forward(model, data)
    left_after = _site_pos(model, data, "left_palm")
    right_after = _site_pos(model, data, "right_palm")
    np.copyto(data.qpos, saved)
    mujoco.mj_forward(model, data)
    left_forward = bool(np.linalg.norm(left_after - baseline_left) > 1e-4)
    right_forward = bool(np.linalg.norm(right_after - baseline_right) > 1e-4)
    return {
        "arm_joint_mapping_valid": bool(left_forward and right_forward),
        "left_arm_forward_command_increases_reach": left_forward,
        "right_arm_forward_command_increases_reach": right_forward,
    }


def _solve_arm_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    left_target: np.ndarray,
    right_target: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    ik_data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    np.copyto(ik_data.qpos, data.qpos)
    np.copyto(ik_data.qvel, data.qvel)
    mujoco.mj_forward(model, ik_data)

    arm_targets: dict[str, float] = {}
    for chain, site_name, target in (
        (LEFT_ARM_CHAIN, "left_palm", left_target),
        (RIGHT_ARM_CHAIN, "right_palm", right_target),
    ):
        qadrs, q_low, q_high = build_chain_metadata(model, chain)
        q_neutral = np.array([float(ik_data.qpos[adr]) for adr in qadrs], dtype=float)
        frozen = frozen_qpos_snapshot(model, ik_data, active_joint_names=chain)
        q_work, _ = solve_position_only(
            model,
            fd,
            ik_data,
            ik_qpos_adrs=qadrs,
            q_low=q_low,
            q_high=q_high,
            q_neutral=q_neutral,
            body_id=-1,
            site_id=_site_id(model, site_name),
            target_xyz=np.asarray(target, dtype=float),
            frozen_qpos=frozen,
            inner_iters=22,
            max_abs_joint_from_neutral=1.4,
        )
        for name, val in zip(chain, q_work, strict=False):
            arm_targets[name] = float(val)

    mujoco.mj_forward(model, ik_data)
    left_err = float(np.linalg.norm(_site_pos(model, ik_data, "left_palm") - np.asarray(left_target, dtype=float)))
    right_err = float(np.linalg.norm(_site_pos(model, ik_data, "right_palm") - np.asarray(right_target, dtype=float)))
    return arm_targets, left_err, right_err


def run_g1_lucky_pick_transport_state_machine(
    *,
    headless: bool = False,
    timeout: float = 30.0,
    viewer_speed: float = 1.0,
    pick_x: float | None = None,
    pick_y: float | None = None,
    table_x: float = 0.351,
    box_y: float = 0.026,
    box_size_x: float = 0.18,
    box_size_y: float = 0.12,
    box_size_z: float = 0.10,
    box_edge_offset: float = 0.08,
    pick_stand_off: float = 0.35,
    reach_only: bool = False,
    auto_tune_pick_stand_off: bool = False,
    manipulation_stance: bool = False,
    auto_select_manipulation_stance: bool = True,
    stance_foot: str = "left",
    stance_forward_offset: float = 0.08,
    stance_width_offset: float = 0.03,
    manipulation_knee_bend: float = 0.10,
    stance_duration: float = 2.0,
    reach_mode: str = "staged",
    reach_preset: str = "auto",
    use_right_reacher: bool = False,
    left_arm_method: str = "direct_ik",
    reach_duration: float = 5.0,
    stop_stabilize_duration: float = 1.0,
    grip_duration: float = 1.0,
    hold_viewer_after_done: bool | None = None,
    post_done_hold: float = 3.0,
    expected_payload_mass_kg: float = 2.0,
    debug_markers: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    missing = missing_lucky_assets()
    if missing:
        return {"success": False, "failure_reason": "missing_lucky_assets:" + ",".join(missing)}

    config = load_lucky_config(LUCKY_MODEL_CONFIG)
    model = mujoco.MjModel.from_xml_path(str(LUCKY_SCENE_XML))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(config["joint_names"]):
        data.qpos[7 + i] = float(config["default_joint_pos"].get(name, 0.0))
    scene_metrics = setup_lucky_pick_scene(
        model,
        data,
        table_x=table_x,
        box_y=box_y,
        box_size_xyz=(float(box_size_x), float(box_size_y), float(box_size_z)),
        box_edge_offset=float(box_edge_offset),
    )

    try:
        walker = load_walker_policy(LUCKY_WALKER_ONNX)
    except LuckyPolicyLoadError as exc:
        return {"success": False, "failure_reason": str(exc), **scene_metrics}

    joint_map = build_lucky_joint_map(model, config, control_waist=True)
    stance_enabled_config = bool((manipulation_stance or auto_select_manipulation_stance) and not reach_only)
    phases_completed: list[str] = ["INIT_SCENE"]
    initial_pelvis_z = float(data.qpos[2])
    last_action = np.zeros(joint_map.action_dim, dtype=np.float32)
    target_pos = joint_map.default_joint_pos.copy()
    computed_pick = compute_pick_point(model, pick_stand_off=float(pick_stand_off), box_y=float(box_y))
    pick_stand_off_candidates_tested = 0
    best_pick_stand_off_m = float(pick_stand_off)
    best_reach_error_m = float("inf")
    if auto_tune_pick_stand_off:
        saved_qpos = np.asarray(data.qpos, dtype=float).copy()
        for candidate in (0.20, 0.25, 0.30, 0.35, 0.40):
            cand_pick = compute_pick_point(model, pick_stand_off=float(candidate), box_y=float(box_y))
            data.qpos[0] = float(cand_pick[0])
            data.qpos[1] = float(cand_pick[1])
            mujoco.mj_forward(model, data)
            cand_det = detect_box_ground_truth(model, data)
            _, _, cand_err, _ = _select_reach_preset(
                model,
                data,
                detection=cand_det,
                requested=reach_preset,
            )
            pick_stand_off_candidates_tested += 1
            if cand_err < best_reach_error_m:
                best_reach_error_m = float(cand_err)
                best_pick_stand_off_m = float(candidate)
        np.copyto(data.qpos, saved_qpos)
        mujoco.mj_forward(model, data)
        computed_pick = compute_pick_point(model, pick_stand_off=best_pick_stand_off_m, box_y=float(box_y))
    pick = np.array(
        [
            float(computed_pick[0] if pick_x is None else pick_x),
            float(computed_pick[1] if pick_y is None else pick_y),
        ],
        dtype=float,
    )
    if reach_only:
        data.qpos[0] = float(pick[0])
        data.qpos[1] = float(pick[1])
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
    max_pitch = 0.0
    max_roll = 0.0
    robot_fell_during_walk = False
    floor_contact_seen = False
    sync_after_step: Any = None
    stop_pelvis_speed = 0.0
    stop_torso_pitch = 0.0
    stop_torso_roll = 0.0
    max_arm_table_contact_count = 0
    max_hand_table_contact_count = 0
    left_hand_table_contact = False
    right_hand_table_contact = False
    current_detection: Any | None = None
    support_before = _support_metrics(model, data)
    support_after_stop = dict(support_before)
    support_after_stance = dict(support_before)
    enter_manipulation_stance_success = not stance_enabled_config
    robot_stable_in_manipulation_stance = not stance_enabled_config
    manipulation_stance_candidates_tested = 0
    selected_stance_forward_offset_m = 0.0
    selected_stance_width_offset_m = 0.0
    selected_knee_bend_rad = 0.0
    selected_torso_pitch_bias_rad = 0.0
    best_stance_com_margin_m = float(support_before["com_margin_m"])
    stance_selection_success = not stance_enabled_config
    manipulation_stance_fallback = "disabled" if not stance_enabled_config else "none"
    com_margin_improved_by_stance = False
    torso_backward_lean_detected = False
    robot_fell_backward = False
    max_backward_pitch_rad = 0.0
    torso_pitch_after_stance = 0.0
    torso_roll_after_stance = 0.0
    robot_fell_during_stance = False
    stance_hold_completed = not manipulation_stance
    min_com_margin_during_reach = float("inf")
    robot_fell_during_reach = False
    torso_pitch_max_during_reach = 0.0
    reach_paused_for_stability = False
    distance_to_pick_at_stop = float("inf")
    robot_stopped_at_pick_point_metric = False
    base_xy_at_stop = np.asarray(data.qpos[:2], dtype=float).copy()
    selected_reach_preset = "none"
    reach_presets_tested = 0
    best_preset_error_m = float("inf")
    reach_animation_played = False
    reach_animation_duration_s = 0.0
    grip_animation_duration_s = 0.0
    viewer_hold_completed = False

    def step_body(cmd: np.ndarray) -> None:
        nonlocal last_action, target_pos, max_pitch, max_roll, robot_fell_during_walk, floor_contact_seen
        obs = _make_observation(data, joint_map, last_action, cmd)
        last_action = np.asarray(walker(obs), dtype=np.float32).reshape(joint_map.action_dim)
        target_pos = action_to_joint_targets(last_action, joint_map)
        _apply_body_targets(data, joint_map, target_pos)
        mujoco.mj_step(model, data)
        fell, pitch_abs, roll_abs = _robot_fell(model, data, initial_pelvis_z)
        max_pitch = max(max_pitch, pitch_abs)
        max_roll = max(max_roll, roll_abs)
        robot_fell_during_walk = robot_fell_during_walk or fell
        floor_contact_seen = floor_contact_seen or _floor_contact_detected(model, data)
        if sync_after_step is not None:
            sync_after_step()

    def run_headless_phase_loop() -> None:
        nonlocal target_pos
        nonlocal stop_pelvis_speed, stop_torso_pitch, stop_torso_roll
        nonlocal max_arm_table_contact_count, max_hand_table_contact_count
        nonlocal left_hand_table_contact, right_hand_table_contact
        nonlocal current_detection
        nonlocal support_after_stop, support_after_stance, enter_manipulation_stance_success
        nonlocal robot_stable_in_manipulation_stance, torso_pitch_after_stance, torso_roll_after_stance
        nonlocal robot_fell_during_stance, stance_hold_completed, min_com_margin_during_reach
        nonlocal robot_fell_during_reach, torso_pitch_max_during_reach, reach_paused_for_stability
        nonlocal max_pitch, max_roll
        nonlocal distance_to_pick_at_stop, robot_stopped_at_pick_point_metric
        nonlocal base_xy_at_stop
        nonlocal selected_reach_preset, reach_presets_tested, best_preset_error_m
        nonlocal reach_animation_played, reach_animation_duration_s, grip_animation_duration_s
        nonlocal viewer_hold_completed
        nonlocal manipulation_stance_candidates_tested, selected_stance_forward_offset_m
        nonlocal selected_stance_width_offset_m, selected_knee_bend_rad, selected_torso_pitch_bias_rad
        nonlocal best_stance_com_margin_m, stance_selection_success, manipulation_stance_fallback
        nonlocal com_margin_improved_by_stance, torso_backward_lean_detected, robot_fell_backward
        nonlocal max_backward_pitch_rad
        if reach_only:
            distance_to_pick_at_stop = 0.0
            robot_stopped_at_pick_point_metric = True
            base_xy_at_stop = np.asarray(data.qpos[:2], dtype=float).copy()
        else:
            while float(data.time) < float(timeout):
                dist = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - pick))
                if dist <= 0.20 or robot_fell_during_walk:
                    break
                step_body(np.array([0.6, 0.0, 0.0], dtype=np.float32))

            base_xy_at_stop = np.asarray(data.qpos[:2], dtype=float).copy()
            distance_to_pick_at_stop = float(np.linalg.norm(np.asarray(data.qpos[:2], dtype=float) - pick))
            robot_stopped_at_pick_point_metric = bool(distance_to_pick_at_stop <= 0.25)
        phases_completed.append("WALK_TO_PICK_POINT")
        stop_start = float(data.time)
        while float(data.time) - stop_start < float(stop_stabilize_duration) and float(data.time) < float(timeout):
            if reach_only:
                data.time += float(model.opt.timestep)
                mujoco.mj_forward(model, data)
                if sync_after_step is not None:
                    sync_after_step()
            else:
                step_body(np.zeros(3, dtype=np.float32))

        phases_completed.append("STOP_AND_STABILIZE")
        stop_pelvis_speed = float(np.linalg.norm(np.asarray(data.qvel[:2], dtype=float)))
        stop_roll, stop_pitch = pelvis_roll_pitch_deg(data, model)
        stop_torso_pitch = abs(float(stop_pitch)) if stop_pitch is not None else 0.0
        stop_torso_roll = abs(float(stop_roll)) if stop_roll is not None else 0.0
        support_after_stop = _support_metrics(model, data)

        stance_enabled = stance_enabled_config
        if stance_enabled:
            pre_stance_qpos = np.asarray(data.qpos, dtype=float).copy()
            pre_stance_qvel = np.asarray(data.qvel, dtype=float).copy()
            base_stance_targets = target_pos.copy()
            apply_selected_stance = True
            if auto_select_manipulation_stance:
                selection = _select_manipulation_stance(
                    model,
                    data,
                    joint_map,
                    base_targets=base_stance_targets,
                    stance_foot=str(stance_foot),
                    requested_forward_offset=float(stance_forward_offset),
                    requested_width_offset=float(stance_width_offset),
                    requested_knee_bend=float(manipulation_knee_bend),
                    initial_pelvis_z=initial_pelvis_z,
                    margin_before=float(support_after_stop["com_margin_m"]),
                )
                stance_targets = np.asarray(selection["target"], dtype=np.float32)
                manipulation_stance_candidates_tested = int(selection["candidates_tested"])
                stance_selection_success = bool(selection["success"])
                manipulation_stance_fallback = str(selection["fallback"])
                selected = selection["selected"]
                selected_stance_forward_offset_m = float(selected["forward_offset"])
                selected_stance_width_offset_m = float(selected["width_offset"])
                selected_knee_bend_rad = float(selected["knee_bend"])
                selected_torso_pitch_bias_rad = float(selected["torso_pitch_bias"])
                best_stance_com_margin_m = float(selection["metrics"]["com_margin_m"])
                apply_selected_stance = bool(selection["success"])
            else:
                stance_targets = _make_manipulation_stance_targets(
                    joint_map,
                    base_stance_targets,
                    stance_foot=str(stance_foot),
                    forward_offset=float(stance_forward_offset),
                    width_offset=float(stance_width_offset),
                    knee_bend=float(manipulation_knee_bend),
                    torso_pitch_bias=0.0,
                )
                stance_selection_success = True
                manipulation_stance_fallback = "none"
                selected_stance_forward_offset_m = float(stance_forward_offset)
                selected_stance_width_offset_m = float(stance_width_offset)
                selected_knee_bend_rad = float(manipulation_knee_bend)
            if apply_selected_stance:
                stance_start = float(data.time)
                while float(data.time) - stance_start < float(stance_duration) and float(data.time) < float(timeout):
                    u = min(1.0, max(0.0, (float(data.time) - stance_start) / max(float(stance_duration), 1e-6)))
                    target_pos = base_stance_targets + u * (stance_targets - base_stance_targets)
                    _apply_body_targets(data, joint_map, target_pos)
                    mujoco.mj_step(model, data)
                    fell, pitch_abs, roll_abs = _robot_fell(model, data, initial_pelvis_z)
                    _, pitch_signed = pelvis_roll_pitch_deg(data, model)
                    backward = max(0.0, -float(pitch_signed or 0.0))
                    max_backward_pitch_rad = max(max_backward_pitch_rad, backward)
                    robot_fell_during_stance = robot_fell_during_stance or fell
                    max_pitch = max(max_pitch, pitch_abs)
                    max_roll = max(max_roll, roll_abs)
                    if sync_after_step is not None:
                        sync_after_step()
                stance_hold_completed = bool(float(data.time) - stance_start >= float(stance_duration) - float(model.opt.timestep))
            else:
                stance_hold_completed = True
        support_after_stance = _support_metrics(model, data)
        stance_roll, stance_pitch = pelvis_roll_pitch_deg(data, model)
        torso_pitch_after_stance = abs(float(stance_pitch)) if stance_pitch is not None else 0.0
        torso_roll_after_stance = abs(float(stance_roll)) if stance_roll is not None else 0.0
        support_grew = bool(
            float(support_after_stance["support_polygon_length_m"]) > float(support_after_stop["support_polygon_length_m"]) + 1e-4
            or float(support_after_stance["support_polygon_area_m2"]) > float(support_after_stop["support_polygon_area_m2"]) + 1e-5
        )
        com_margin_improved_by_stance = bool(
            float(support_after_stance["com_margin_m"]) >= float(support_after_stop["com_margin_m"]) - 0.005
        )
        torso_backward_lean_detected = bool(max_backward_pitch_rad > 0.20)
        robot_fell_backward = bool(
            torso_backward_lean_detected
            or float(support_after_stance["com_margin_m"]) <= -0.01
        )
        robot_stable_in_manipulation_stance = bool(
            not robot_fell_during_stance
            and bool(support_after_stance["com_inside_support_polygon"])
            and torso_pitch_after_stance < 0.35
            and not robot_fell_backward
            and stance_hold_completed
        )
        enter_manipulation_stance_success = bool(
            (not stance_enabled)
            or (
                stance_selection_success
                and robot_stable_in_manipulation_stance
                and (support_grew or com_margin_improved_by_stance)
            )
        )
        if stance_enabled and not enter_manipulation_stance_success:
            np.copyto(data.qpos, pre_stance_qpos)
            np.copyto(data.qvel, pre_stance_qvel)
            mujoco.mj_forward(model, data)
            target_pos = base_stance_targets.copy()
            support_after_stance = dict(support_after_stop)
            torso_pitch_after_stance = stop_torso_pitch
            torso_roll_after_stance = stop_torso_roll
            if manipulation_stance_fallback == "none":
                manipulation_stance_fallback = "neutral_stand"
            fallback_start = float(data.time)
            while float(data.time) - fallback_start < 0.75 and float(data.time) < float(timeout):
                _apply_body_targets(data, joint_map, target_pos)
                mujoco.mj_step(model, data)
                if sync_after_step is not None:
                    sync_after_step()
        phases_completed.append("ENTER_MANIPULATION_STANCE")
        phases_completed.append("DETECT_BOX")

        detection = detect_box_ground_truth(model, data)
        current_detection = detection
        finger = Dex3FingerController()
        finger.reset(_finger_qpos_snapshot(model, data))
        open_targets = finger.step("pregrasp_spread")
        selected_preset_name, preset_targets, best_preset_error, preset_count = _select_reach_preset(
            model,
            data,
            detection=detection,
            requested=str(reach_preset),
        )
        selected_reach_preset = selected_preset_name
        best_preset_error_m = float(best_preset_error)
        reach_presets_tested = int(preset_count)
        arm_targets, left_err, right_err = _solve_arm_targets(
            model,
            data,
            left_target=np.asarray(detection.left_grasp_target_world, dtype=float),
            right_target=np.asarray(detection.right_grasp_target_world, dtype=float),
        )
        if reach_mode == "staged":
            preset_left, preset_right, _ = _preview_arm_targets_error(
                model,
                data,
                arm_targets=preset_targets,
                detection=detection,
            )
            if max(preset_left, preset_right) < 0.30 or max(preset_left, preset_right) <= max(left_err, right_err) + 0.05:
                arm_targets = preset_targets
                left_err, right_err = preset_left, preset_right
        start_arm_targets: dict[str, float] = {}
        for name in (*LEFT_ARM_CHAIN, *RIGHT_ARM_CHAIN):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                start_arm_targets[name] = float(data.qpos[int(model.jnt_qposadr[jid])])
        blended_arm_targets = dict(start_arm_targets)
        if reach_only:
            reach_start = float(data.time)
            while float(data.time) - reach_start < float(reach_duration) and float(data.time) < float(timeout):
                raw_u = max(0.0, (float(data.time) - reach_start) / max(float(reach_duration), 1e-6))
                u = raw_u * raw_u * (3.0 - 2.0 * raw_u)
                for name, target in arm_targets.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        data.qpos[int(model.jnt_qposadr[jid])] = float(start_arm_targets.get(name, target) + u * (target - start_arm_targets.get(name, target)))
                data.time += float(model.opt.timestep)
                mujoco.mj_forward(model, data)
                min_com_margin_during_reach = min(min_com_margin_during_reach, float(_support_metrics(model, data)["com_margin_m"]))
                if sync_after_step is not None:
                    sync_after_step()
            reach_animation_played = True
            reach_animation_duration_s = float(data.time - reach_start)
            phases_completed.append("TWO_ARM_REACH")
            close_targets = finger.targets_for_mode("side_support_grasp")
            open_now = finger.targets_for_mode("pregrasp_spread")
            grip_start = float(data.time)
            while float(data.time) - grip_start < float(grip_duration) and float(data.time) < float(timeout):
                gu = max(0.0, (float(data.time) - grip_start) / max(float(grip_duration), 1e-6))
                gu = gu * gu * (3.0 - 2.0 * gu)
                for name, target in arm_targets.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        data.qpos[int(model.jnt_qposadr[jid])] = float(target)
                for name in close_targets:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid >= 0:
                        data.qpos[int(model.jnt_qposadr[jid])] = float(open_now.get(name, 0.0) + gu * (close_targets.get(name, 0.0) - open_now.get(name, 0.0)))
                data.time += float(model.opt.timestep)
                mujoco.mj_forward(model, data)
                if sync_after_step is not None:
                    sync_after_step()
            grip_animation_duration_s = float(data.time - grip_start)
            phases_completed.append("GRIP_BOX")
            phases_completed.append("DONE")
            return detection, left_err, right_err
        if stance_enabled and not enter_manipulation_stance_success:
            reach_paused_for_stability = True
            min_com_margin_during_reach = float(_support_metrics(model, data)["com_margin_m"])
            phases_completed.append("TWO_ARM_REACH")
            phases_completed.append("GRIP_BOX")
            phases_completed.append("DONE")
            return detection, left_err, right_err
        reach_start = float(data.time)
        while float(data.time) - reach_start < float(reach_duration) and float(data.time) < float(timeout):
            support_now = _support_metrics(model, data)
            min_com_margin_during_reach = min(min_com_margin_during_reach, float(support_now["com_margin_m"]))
            fell, pitch_abs, roll_abs = _robot_fell(model, data, initial_pelvis_z)
            robot_fell_during_reach = robot_fell_during_reach or fell
            torso_pitch_max_during_reach = max(torso_pitch_max_during_reach, pitch_abs)
            max_pitch = max(max_pitch, pitch_abs)
            max_roll = max(max_roll, roll_abs)
            if (reach_only or enter_manipulation_stance_success) and float(support_now["com_margin_m"]) > 0.005 and pitch_abs < 0.35:
                cap = 1.0 if reach_only else 0.25
                raw_u = max(0.0, (float(data.time) - reach_start) / max(float(reach_duration), 1e-6))
                uu = min(cap, raw_u)
                u = uu * uu * (3.0 - 2.0 * uu)
                for name, target in arm_targets.items():
                    blended_arm_targets[name] = float(start_arm_targets.get(name, target) + u * (target - start_arm_targets.get(name, target)))
            else:
                reach_paused_for_stability = True
            _apply_body_targets(data, joint_map, target_pos, arm_targets=blended_arm_targets)
            _apply_finger_targets(model, data, open_targets)
            mujoco.mj_step(model, data)
            counts = _table_contact_counts(model, data)
            max_arm_table_contact_count = max(max_arm_table_contact_count, int(counts["arm_table_contact_count"]))
            max_hand_table_contact_count = max(max_hand_table_contact_count, int(counts["hand_table_contact_count"]))
            left_hand_table_contact = left_hand_table_contact or bool(counts["left_hand_table_contact"])
            right_hand_table_contact = right_hand_table_contact or bool(counts["right_hand_table_contact"])
            if sync_after_step is not None:
                sync_after_step()

        reach_animation_played = True
        reach_animation_duration_s = float(data.time - reach_start)
        phases_completed.append("TWO_ARM_REACH")
        close_targets = finger.targets_for_mode("side_support_grasp")
        grip_start = float(data.time)
        while float(data.time) - grip_start < float(grip_duration) and float(data.time) < float(timeout):
            gu = min(1.0, max(0.0, (float(data.time) - grip_start) / max(float(grip_duration), 1e-6)))
            gu = gu * gu * (3.0 - 2.0 * gu)
            open_now = finger.targets_for_mode("pregrasp_spread")
            finger_targets = {
                name: float(open_now.get(name, 0.0) + gu * (close_targets.get(name, 0.0) - open_now.get(name, 0.0)))
                for name in close_targets
            }
            _apply_body_targets(data, joint_map, target_pos, arm_targets=blended_arm_targets)
            _apply_finger_targets(model, data, finger_targets)
            mujoco.mj_step(model, data)
            counts = _table_contact_counts(model, data)
            max_arm_table_contact_count = max(max_arm_table_contact_count, int(counts["arm_table_contact_count"]))
            max_hand_table_contact_count = max(max_hand_table_contact_count, int(counts["hand_table_contact_count"]))
            left_hand_table_contact = left_hand_table_contact or bool(counts["left_hand_table_contact"])
            right_hand_table_contact = right_hand_table_contact or bool(counts["right_hand_table_contact"])
            if sync_after_step is not None:
                sync_after_step()

        grip_animation_duration_s = float(data.time - grip_start)
        phases_completed.append("GRIP_BOX")
        phases_completed.append("DONE")
        return detection, left_err, right_err

    result_holder: dict[str, Any] = {}
    if headless:
        detection, left_err, right_err = run_headless_phase_loop()
    else:
        hold_after_done = True if hold_viewer_after_done is None else bool(hold_viewer_after_done)
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            def _sync() -> None:
                if debug_markers:
                    _draw_debug_markers(
                        viewer,
                        model,
                        data,
                        pick=pick,
                        detection=current_detection,
                        table_front_edge_x=float(scene_metrics["table_front_edge_x"]),
                    )
                viewer.sync()
                time.sleep(max(0.001, float(model.opt.timestep) / max(float(viewer_speed), 1e-3)))

            sync_after_step = _sync
            detection, left_err, right_err = run_headless_phase_loop()
            if hold_after_done:
                hold_start = time.time()
                while viewer.is_running() and time.time() - hold_start < float(post_done_hold):
                    _sync()
                viewer_hold_completed = bool(time.time() - hold_start >= float(post_done_hold))
            else:
                viewer.sync()
                time.sleep(0.5 / max(float(viewer_speed), 1e-3))
        result_holder["viewer_closed"] = True

    dist_to_pick = float(distance_to_pick_at_stop)
    dist_robot_to_box = float(np.linalg.norm(base_xy_at_stop - np.asarray(detection.box_position_world[:2], dtype=float)))
    dist_robot_to_edge = float(abs(float(scene_metrics["table_front_edge_x"]) - float(base_xy_at_stop[0])))
    left_hand_pos = _site_pos(model, data, "left_palm")
    right_hand_pos = _site_pos(model, data, "right_palm")
    left_runtime_err = float(np.linalg.norm(left_hand_pos - np.asarray(detection.left_grasp_target_world, dtype=float)))
    right_runtime_err = float(np.linalg.norm(right_hand_pos - np.asarray(detection.right_grasp_target_world, dtype=float)))
    left_contact = _body_or_geom_contact(model, data, "left")
    right_contact = _body_or_geom_contact(model, data, "right")
    two_arm_reach_success = bool(left_runtime_err < 0.08 and right_runtime_err < 0.08)
    grasp_attempt_success = bool(left_contact and right_contact)
    if not np.isfinite(min_com_margin_during_reach):
        min_com_margin_during_reach = float(support_after_stance["com_margin_m"])
    reach_stability_success = bool(
        not robot_fell_during_reach
        and torso_pitch_max_during_reach < 0.45
        and min_com_margin_during_reach > 0.005
    )
    max_hand_error = max(left_runtime_err, right_runtime_err)
    hand_target_error_improved = bool(max_hand_error < 0.40)
    reach_near_success = bool(max_hand_error < 0.25)
    reach_success = bool(max_hand_error < 0.12)
    reach_improvement_success = bool(max_hand_error < 0.30)
    joint_diag = _arm_joint_mapping_diagnostics(model, data)
    left_hand_measurement_name = "left_palm"
    right_hand_measurement_name = "right_palm"
    hand_measurement_frame_valid = bool(_site_id(model, left_hand_measurement_name) >= 0 and _site_id(model, right_hand_measurement_name) >= 0)
    table_collision_blocking = bool(
        (max_hand_table_contact_count > 0 or max_arm_table_contact_count >= 3)
        and not two_arm_reach_success
    )
    reach_only_success = bool(
        reach_only
        and not robot_fell_during_reach
        and not table_collision_blocking
        and reach_animation_played
        and float(data.time) >= float(reach_duration)
        and reach_improvement_success
    )
    pick_pose_reachable = bool(0.20 <= dist_robot_to_edge <= 0.55 and dist_robot_to_box <= 0.70)
    geometry_success = bool(
        scene_metrics["box_initial_pose_valid"]
        and scene_metrics["box_on_table"]
        and scene_metrics["box_graspable_size"]
        and scene_metrics["box_near_table_edge"]
        and pick_pose_reachable
        and detection.grasp_targets_above_table
        and detection.grasp_targets_clear_table_edge
        and not table_collision_blocking
        and not robot_fell_during_walk
    )
    reach_failure_reasons: list[str] = []
    if not two_arm_reach_success:
        reach_failure_reasons.append("two_arm_reach_error_above_threshold")
    if not grasp_attempt_success:
        reach_failure_reasons.append("dual_hand_contact_not_confirmed")
    if robot_fell_during_reach:
        reach_failure_reasons.append("robot_fell_during_reach")
    if torso_pitch_max_during_reach >= 0.45:
        reach_failure_reasons.append("torso_pitch_too_high_during_reach")
    if min_com_margin_during_reach <= 0.005:
        reach_failure_reasons.append("low_com_margin_during_reach")
    if not hand_target_error_improved:
        reach_failure_reasons.append("hand_target_error_not_improved")

    failure_reasons: list[str] = []
    if not bool(scene_metrics["box_initial_pose_valid"]):
        failure_reasons.append("box_initial_pose_invalid")
    if robot_fell_during_walk:
        failure_reasons.append("robot_fell_before_manipulation")
    if dist_to_pick > 0.25:
        failure_reasons.append("walk_to_pick_incomplete")
    if not detection.box_detected:
        failure_reasons.append("box_not_detected")
    if not bool(scene_metrics["box_graspable_size"]):
        failure_reasons.append("box_not_graspable_size")
    if not bool(scene_metrics["box_near_table_edge"]):
        failure_reasons.append("box_not_near_table_edge")
    if not pick_pose_reachable:
        failure_reasons.append("pick_pose_not_reachable")
    if not detection.grasp_targets_above_table:
        failure_reasons.append("grasp_targets_not_above_table")
    if not detection.grasp_targets_clear_table_edge:
        failure_reasons.append("grasp_targets_not_clear_table_edge")
    if table_collision_blocking:
        failure_reasons.append("table_collision_blocking_reach")
    if stance_enabled_config and not enter_manipulation_stance_success:
        failure_reasons.append("manipulation_stance_failed")
    if robot_fell_backward:
        failure_reasons.append("robot_fell_backward")
    if not reach_stability_success:
        failure_reasons.append("reach_stability_failed")
    if not hand_target_error_improved:
        failure_reasons.append("hand_target_error_not_improved")

    expected_payload_forward_moment_nm = float(
        expected_payload_mass_kg
        * 9.81
        * max(0.0, float(detection.box_position_world[0]) - float(data.qpos[0]))
    )
    payload_balance_margin_estimate = float(
        float(support_after_stance["com_margin_m"])
        - 0.01 * expected_payload_forward_moment_nm
    )

    out: dict[str, Any] = {
        **scene_metrics,
        **result_holder,
        "state_machine_enabled": True,
        "reach_only_mode": bool(reach_only),
        "phases_completed": phases_completed,
        "walker_policy_loaded": True,
        "walker_controls_legs_waist_only": True,
        "walker_arm_outputs_masked": True,
        "debug_markers_enabled": bool(debug_markers),
        "pick_stand_off_m": float(pick_stand_off),
        "best_pick_stand_off_m": float(best_pick_stand_off_m),
        "best_reach_error_m": float(best_reach_error_m if np.isfinite(best_reach_error_m) else best_preset_error_m),
        "pick_stand_off_candidates_tested": int(pick_stand_off_candidates_tested),
        "computed_pick_point": [float(pick[0]), float(pick[1])],
        "distance_robot_to_box_at_stop_m": float(dist_robot_to_box),
        "distance_robot_to_table_edge_at_stop_m": float(dist_robot_to_edge),
        "pick_pose_reachable": bool(pick_pose_reachable),
        "walk_to_pick_success": bool(robot_stopped_at_pick_point_metric and not robot_fell_during_walk),
        "distance_to_pick_point_m": float(dist_to_pick),
        "robot_stopped_at_pick_point": bool(robot_stopped_at_pick_point_metric),
        "robot_fell_during_walk": bool(robot_fell_during_walk),
        "stop_and_stabilize_success": bool(stop_pelvis_speed < 0.20 and stop_torso_pitch < 0.30 and stop_torso_roll < 0.30),
        "pelvis_velocity_after_stop": float(stop_pelvis_speed),
        "torso_pitch_after_stop": float(stop_torso_pitch),
        "torso_roll_after_stop": float(stop_torso_roll),
        "manipulation_stance_enabled": bool(stance_enabled_config),
        "auto_select_manipulation_stance": bool(auto_select_manipulation_stance),
        "manipulation_stance_candidates_tested": int(manipulation_stance_candidates_tested),
        "selected_stance_forward_offset_m": float(selected_stance_forward_offset_m),
        "selected_stance_width_offset_m": float(selected_stance_width_offset_m),
        "selected_knee_bend_rad": float(selected_knee_bend_rad),
        "selected_torso_pitch_bias_rad": float(selected_torso_pitch_bias_rad),
        "best_stance_com_margin_m": float(best_stance_com_margin_m),
        "stance_selection_success": bool(stance_selection_success),
        "manipulation_stance_fallback": str(manipulation_stance_fallback),
        "enter_manipulation_stance_success": bool(enter_manipulation_stance_success),
        "stance_foot": str(stance_foot),
        "stance_forward_offset_m": float(stance_forward_offset),
        "stance_width_offset_m": float(stance_width_offset),
        "manipulation_knee_bend_rad": float(manipulation_knee_bend),
        "support_polygon_length_before_m": float(support_after_stop["support_polygon_length_m"]),
        "support_polygon_length_after_m": float(support_after_stance["support_polygon_length_m"]),
        "support_polygon_area_before_m2": float(support_after_stop["support_polygon_area_m2"]),
        "support_polygon_area_after_m2": float(support_after_stance["support_polygon_area_m2"]),
        "com_margin_before_stance_m": float(support_after_stop["com_margin_m"]),
        "com_margin_after_stance_m": float(support_after_stance["com_margin_m"]),
        "com_margin_improved_by_stance": bool(com_margin_improved_by_stance),
        "com_margin_before_reach_m": float(support_after_stop["com_margin_m"]),
        "torso_pitch_after_stance_rad": float(torso_pitch_after_stance),
        "torso_roll_after_stance_rad": float(torso_roll_after_stance),
        "robot_stable_in_manipulation_stance": bool(robot_stable_in_manipulation_stance),
        "torso_backward_lean_detected": bool(torso_backward_lean_detected),
        "robot_fell_backward": bool(robot_fell_backward),
        "max_backward_pitch_rad": float(max_backward_pitch_rad),
        "box_detected": bool(detection.box_detected),
        "box_position_world": list(detection.box_position_world),
        "left_grasp_target_world": list(detection.left_grasp_target_world),
        "right_grasp_target_world": list(detection.right_grasp_target_world),
        "grasp_target_height_above_table_m": float(detection.grasp_target_height_above_table_m),
        "grasp_targets_above_table": bool(detection.grasp_targets_above_table),
        "grasp_targets_clear_table_edge": bool(detection.grasp_targets_clear_table_edge),
        "detection_mode": detection.detection_mode,
        "two_arm_reach_attempted": True,
        "reach_mode": str(reach_mode),
        "selected_reach_preset": str(selected_reach_preset),
        "reach_presets_tested": int(reach_presets_tested),
        "best_preset_error_m": float(best_preset_error_m),
        "reach_duration_s": float(reach_duration),
        "stop_stabilize_duration_s": float(stop_stabilize_duration),
        "reach_animation_played": bool(reach_animation_played),
        "reach_animation_duration_s": float(reach_animation_duration_s),
        "grip_animation_duration_s": float(grip_animation_duration_s),
        "post_done_hold_s": float(post_done_hold),
        "viewer_hold_completed": bool(viewer_hold_completed),
        "left_hand_measurement_name": left_hand_measurement_name,
        "right_hand_measurement_name": right_hand_measurement_name,
        "hand_measurement_frame_valid": bool(hand_measurement_frame_valid),
        "left_hand_to_target_error_m": float(left_runtime_err),
        "right_hand_to_target_error_m": float(right_runtime_err),
        "final_left_hand_to_target_error_m": float(left_runtime_err),
        "final_right_hand_to_target_error_m": float(right_runtime_err),
        "max_hand_to_target_error_m": float(max_hand_error),
        "hand_target_error_improved": bool(hand_target_error_improved),
        "left_ik_planned_error_m": float(left_err),
        "right_ik_planned_error_m": float(right_err),
        "two_arm_reach_success": bool(two_arm_reach_success),
        "reach_near_success": bool(reach_near_success),
        "reach_success": bool(reach_success),
        "reach_improvement_success": bool(reach_improvement_success),
        "reach_only_success": bool(reach_only_success),
        "min_com_margin_during_reach_m": float(min_com_margin_during_reach),
        "robot_fell_during_reach": bool(robot_fell_during_reach),
        "torso_pitch_max_during_reach_rad": float(torso_pitch_max_during_reach),
        "reach_paused_for_stability": bool(reach_paused_for_stability),
        "reach_stability_success": bool(reach_stability_success),
        "right_reacher_used": bool(use_right_reacher),
        "right_reacher_error_m": float(right_runtime_err),
        "right_reacher_improved_error": bool(use_right_reacher and right_runtime_err < 0.449),
        "left_arm_method": str(left_arm_method),
        **joint_diag,
        "hand_grip_commanded": True,
        "left_hand_table_contact": bool(left_hand_table_contact),
        "right_hand_table_contact": bool(right_hand_table_contact),
        "arm_table_contact_count": int(max_arm_table_contact_count),
        "hand_table_contact_count": int(max_hand_table_contact_count),
        "table_collision_blocking_reach": bool(table_collision_blocking),
        "left_hand_contact_box": bool(left_contact),
        "right_hand_contact_box": bool(right_contact),
        "dual_hand_contact_box": bool(left_contact and right_contact),
        "grasp_attempt_success": bool(grasp_attempt_success),
        "actual_floor_contact_detected": bool(floor_contact_seen),
        "left_foot_z": float(_foot_center_xyz(model, data, "left")[2]),
        "right_foot_z": float(_foot_center_xyz(model, data, "right")[2]),
        "max_torso_pitch_rad": float(max_pitch),
        "max_torso_roll_rad": float(max_roll),
        "expected_payload_mass_kg": float(expected_payload_mass_kg),
        "expected_payload_forward_moment_nm": float(expected_payload_forward_moment_nm),
        "payload_balance_margin_estimate": float(payload_balance_margin_estimate),
        "success": bool(geometry_success and not failure_reasons),
        "failure_reason": None if not failure_reasons else "; ".join(failure_reasons),
        "reach_failure_reason": None if not reach_failure_reasons else "; ".join(reach_failure_reasons),
        "sim_time_s": float(data.time),
    }
    if verbose:
        print("----- lucky_pick_transport_state_machine -----")
        for key, value in out.items():
            print(f"{key}: {value}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lucky pick/transport state-machine milestone.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--viewer-speed", type=float, default=1.0)
    ap.add_argument("--pick-x", type=float, default=None)
    ap.add_argument("--pick-y", type=float, default=None)
    ap.add_argument("--table-x", type=float, default=0.351)
    ap.add_argument("--box-y", type=float, default=0.026)
    ap.add_argument("--box-size-x", type=float, default=0.18)
    ap.add_argument("--box-size-y", type=float, default=0.12)
    ap.add_argument("--box-size-z", type=float, default=0.10)
    ap.add_argument("--box-edge-offset", type=float, default=0.08)
    ap.add_argument("--pick-stand-off", type=float, default=0.35)
    ap.add_argument("--reach-only", action="store_true")
    ap.add_argument("--auto-tune-pick-stand-off", action="store_true")
    ap.add_argument("--manipulation-stance", action="store_true")
    ap.add_argument("--auto-select-manipulation-stance", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stance-foot", choices=("left", "right"), default="left")
    ap.add_argument("--stance-forward-offset", type=float, default=0.08)
    ap.add_argument("--stance-width-offset", type=float, default=0.03)
    ap.add_argument("--manipulation-knee-bend", type=float, default=0.10)
    ap.add_argument("--stance-duration", type=float, default=2.0)
    ap.add_argument("--reach-mode", choices=("staged", "direct"), default="staged")
    ap.add_argument("--reach-preset", choices=(*REACH_PRESETS.keys(), "auto"), default="auto")
    ap.add_argument("--use-right-reacher", action="store_true")
    ap.add_argument("--left-arm-method", choices=("mirrored_reacher", "direct_ik", "staged_pose"), default="direct_ik")
    ap.add_argument("--reach-duration", type=float, default=5.0)
    ap.add_argument("--stop-stabilize-duration", type=float, default=1.0)
    ap.add_argument("--grip-duration", type=float, default=1.0)
    ap.add_argument("--hold-viewer-after-done", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--post-done-hold", type=float, default=3.0)
    ap.add_argument("--expected-payload-mass-kg", type=float, default=2.0)
    ap.add_argument("--debug-markers", action="store_true")
    ns = ap.parse_args()
    run_g1_lucky_pick_transport_state_machine(
        headless=bool(ns.headless),
        timeout=float(ns.timeout),
        viewer_speed=float(ns.viewer_speed),
        pick_x=None if ns.pick_x is None else float(ns.pick_x),
        pick_y=None if ns.pick_y is None else float(ns.pick_y),
        table_x=float(ns.table_x),
        box_y=float(ns.box_y),
        box_size_x=float(ns.box_size_x),
        box_size_y=float(ns.box_size_y),
        box_size_z=float(ns.box_size_z),
        box_edge_offset=float(ns.box_edge_offset),
        pick_stand_off=float(ns.pick_stand_off),
        reach_only=bool(ns.reach_only),
        auto_tune_pick_stand_off=bool(ns.auto_tune_pick_stand_off),
        manipulation_stance=bool(ns.manipulation_stance),
        auto_select_manipulation_stance=bool(ns.auto_select_manipulation_stance),
        stance_foot=str(ns.stance_foot),
        stance_forward_offset=float(ns.stance_forward_offset),
        stance_width_offset=float(ns.stance_width_offset),
        manipulation_knee_bend=float(ns.manipulation_knee_bend),
        stance_duration=float(ns.stance_duration),
        reach_mode=str(ns.reach_mode),
        reach_preset=str(ns.reach_preset),
        use_right_reacher=bool(ns.use_right_reacher),
        left_arm_method=str(ns.left_arm_method),
        reach_duration=float(ns.reach_duration),
        stop_stabilize_duration=float(ns.stop_stabilize_duration),
        grip_duration=float(ns.grip_duration),
        hold_viewer_after_done=ns.hold_viewer_after_done,
        post_done_hold=float(ns.post_done_hold),
        expected_payload_mass_kg=float(ns.expected_payload_mass_kg),
        debug_markers=bool(ns.debug_markers),
    )
    if not ns.headless:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
