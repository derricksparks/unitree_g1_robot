#!/usr/bin/env python3
"""Validate the Lucky Robots G1 walker policy from this repository wrapper.

This first milestone intentionally runs locomotion only:
- no payload,
- no shelf/table support,
- no manipulation overlay,
- walker outputs are applied only to legs + waist.
"""

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
    LUCKY_SCENE_XML,
    LUCKY_WALKER_ONNX,
    missing_lucky_assets,
)
from lucky_bridge.lucky_policy_loader import (  # noqa: E402
    LuckyPolicyLoadError,
    load_lucky_config,
    load_walker_policy,
)
from run_g1_posture_hold import pelvis_roll_pitch_deg  # noqa: E402


FLOOR_TOKENS = ("floor", "ground", "plane")
OBSTACLE_TOKENS = ("table", "block", "cylinder", "object", "target")
FOOT_BODY_TOKEN = "ankle_roll"
VISIBLE_FOOT_LIFT_M = 0.008
VISIBLE_FORWARD_SWING_M = 0.010
TORSO_FALL_RAD = 0.50
ROLL_FALL_RAD = 0.50
PELVIS_DROP_M = 0.12


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w = float(quat[0])
    xyz = np.asarray(quat[1:4], dtype=float)
    v = np.asarray(vec, dtype=float)
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    if int(obj_id) < 0:
        return ""
    return str(mujoco.mj_id2name(model, obj_type, int(obj_id)) or "")


def _disable_lucky_front_obstacles(model: mujoco.MjModel) -> dict[str, Any]:
    disabled: list[int] = []
    moved_bodies: set[str] = set()
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        text = f"{_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)} {_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)}".lower()
        if any(tok in text for tok in FLOOR_TOKENS):
            continue
        if not any(tok in text for tok in OBSTACLE_TOKENS):
            continue
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        disabled.append(int(gid))
        if bid > 0:
            model.body_pos[bid, 0] = 10.0
            moved_bodies.add(_name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
    return {
        "front_obstacles_disabled": bool(disabled),
        "disabled_obstacle_geom_count": len(disabled),
        "moved_obstacle_bodies": sorted(b for b in moved_bodies if b),
    }


def _foot_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    out: list[int] = []
    prefix = f"{side}_"
    for gid in range(model.ngeom):
        bname = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])).lower()
        if prefix in bname and FOOT_BODY_TOKEN in bname:
            out.append(int(gid))
    return out


def _foot_center_xyz(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    gids = _foot_geom_ids(model, side)
    if not gids:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        return np.asarray(data.xpos[bid], dtype=float).copy() if bid >= 0 else np.zeros(3)
    return np.mean([np.asarray(data.geom_xpos[gid], dtype=float) for gid in gids], axis=0)


def _floor_contact_detected(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    floor_gids = {
        gid
        for gid in range(model.ngeom)
        if any(tok in _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid).lower() for tok in FLOOR_TOKENS)
    }
    foot_gids = set(_foot_geom_ids(model, "left")) | set(_foot_geom_ids(model, "right"))
    for cid in range(data.ncon):
        con = data.contact[cid]
        if int(con.geom1) in floor_gids and int(con.geom2) in foot_gids:
            return True
        if int(con.geom2) in floor_gids and int(con.geom1) in foot_gids:
            return True
    return False


def _obstacle_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    n = 0
    for cid in range(data.ncon):
        con = data.contact[cid]
        for gid in (int(con.geom1), int(con.geom2)):
            bid = int(model.geom_bodyid[gid])
            text = f"{_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)} {_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)}".lower()
            if any(tok in text for tok in OBSTACLE_TOKENS):
                n += 1
                break
    return n


def _apply_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    target_pos: np.ndarray,
    *,
    hold_masked_arms_at_default: bool = True,
) -> None:
    for idx in joint_map.controlled_indices:
        jname = joint_map.joint_names[idx]
        aid = joint_map.actuator_ids.get(jname, -1)
        if aid >= 0:
            data.ctrl[int(aid)] = float(target_pos[idx])
    if hold_masked_arms_at_default:
        for idx in joint_map.arm_indices:
            jname = joint_map.joint_names[idx]
            aid = joint_map.actuator_ids.get(jname, -1)
            if aid >= 0:
                data.ctrl[int(aid)] = float(joint_map.default_joint_pos[idx])


def _make_observation(
    data: mujoco.MjData,
    joint_map: LuckyJointMap,
    last_action: np.ndarray,
    cmd: np.ndarray,
) -> np.ndarray:
    quat = np.asarray(data.qpos[3:7], dtype=float)
    lin_vel_body = _quat_apply_inverse(quat, np.asarray(data.qvel[:3], dtype=float))
    ang_vel_body = np.asarray(data.qvel[3:6], dtype=float)
    projected_gravity = _quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))
    q = np.zeros(joint_map.action_dim, dtype=np.float32)
    dq = np.zeros(joint_map.action_dim, dtype=np.float32)
    for i, name in enumerate(joint_map.joint_names):
        q_adr = joint_map.qpos_indices.get(name)
        v_adr = joint_map.qvel_indices.get(name)
        if q_adr is not None:
            q[i] = float(data.qpos[q_adr]) - float(joint_map.default_joint_pos[i])
        if v_adr is not None:
            dq[i] = float(data.qvel[v_adr])
    return np.concatenate(
        [
            lin_vel_body.astype(np.float32),
            ang_vel_body.astype(np.float32),
            projected_gravity.astype(np.float32),
            q,
            dq,
            np.asarray(last_action, dtype=np.float32),
            np.asarray(cmd, dtype=np.float32),
        ]
    ).astype(np.float32)


def run_g1_lucky_locomotion_validation(
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
            "lucky_locomotion_enabled": True,
            "policy_loaded": False,
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
    except LuckyPolicyLoadError as exc:
        return {
            "lucky_locomotion_enabled": True,
            "policy_loaded": False,
            "walker_policy_path": str(LUCKY_WALKER_ONNX),
            "success": False,
            "failure_reason": str(exc),
        }

    joint_map = build_lucky_joint_map(model, config, control_waist=True)
    cmd = np.array([float(cmd_x), float(cmd_y), float(cmd_yaw)], dtype=np.float32)
    last_action = np.zeros(joint_map.action_dim, dtype=np.float32)
    initial_pelvis = np.asarray(data.qpos[:3], dtype=float).copy()
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
    control_step = 0
    target_pos = joint_map.default_joint_pos.copy()
    max_torso_pitch_rad = 0.0
    max_torso_roll_rad = 0.0
    robot_fell = False
    floor_contact_seen = False
    max_obstacle_contacts = 0
    start_wall = time.time()

    def step_once() -> bool:
        nonlocal control_step, target_pos
        nonlocal last_action, left_prev_lift, right_prev_lift, left_step_start, right_step_start
        nonlocal left_peak_lift, right_peak_lift, left_peak_forward, right_peak_forward
        nonlocal max_left_lift_ever, max_right_lift_ever
        nonlocal actual_visible_step_count, max_torso_pitch_rad, max_torso_roll_rad
        nonlocal robot_fell, floor_contact_seen, max_obstacle_contacts

        if float(data.time) >= float(timeout):
            return False
        if control_step % 4 == 0:
            obs = _make_observation(data, joint_map, last_action, cmd)
            action = walker(obs)
            last_action = np.asarray(action, dtype=np.float32).reshape(joint_map.action_dim)
            target_pos = action_to_joint_targets(last_action, joint_map)
        _apply_targets(model, data, joint_map, target_pos)
        mujoco.mj_step(model, data)
        control_step += 1

        left = _foot_center_xyz(model, data, "left")
        right = _foot_center_xyz(model, data, "right")
        for side, cur, start, prev_lift in (
            ("left", left, left_step_start, left_prev_lift),
            ("right", right, right_step_start, right_prev_lift),
        ):
            lift = max(0.0, float(cur[2] - start[2]))
            fwd = max(0.0, float(cur[0] - start[0]))
            is_lift = lift >= VISIBLE_FOOT_LIFT_M
            if side == "left":
                left_peak_lift = max(left_peak_lift, lift)
                left_peak_forward = max(left_peak_forward, fwd)
                max_left_lift_ever = max(max_left_lift_ever, lift)
                if prev_lift and not is_lift and left_peak_lift >= VISIBLE_FOOT_LIFT_M and left_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    left_step_start = cur.copy()
                    left_peak_lift = 0.0
                    left_peak_forward = 0.0
                left_prev_lift = is_lift
            else:
                right_peak_lift = max(right_peak_lift, lift)
                right_peak_forward = max(right_peak_forward, fwd)
                max_right_lift_ever = max(max_right_lift_ever, lift)
                if prev_lift and not is_lift and right_peak_lift >= VISIBLE_FOOT_LIFT_M and right_peak_forward >= VISIBLE_FORWARD_SWING_M:
                    actual_visible_step_count += 1
                    right_step_start = cur.copy()
                    right_peak_lift = 0.0
                    right_peak_forward = 0.0
                right_prev_lift = is_lift

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
    max_left_lift = max(max_left_lift_ever, max(0.0, float(_foot_center_xyz(model, data, "left")[2] - left_start[2])))
    max_right_lift = max(max_right_lift_ever, max(0.0, float(_foot_center_xyz(model, data, "right")[2] - right_start[2])))
    failure_reasons: list[str] = []
    if not joint_map.mapping_ok:
        failure_reasons.append("joint_mapping_missing_actuators")
    if robot_fell:
        failure_reasons.append("robot_fell")
    if actual_visible_step_count < 1 and abs(avg_speed) < 0.01:
        failure_reasons.append("no_visible_walking_motion")
    if not floor_contact_seen:
        failure_reasons.append("no_floor_contact")
    if max_obstacle_contacts > 0:
        failure_reasons.append("obstacle_contact_detected")
    success = bool(
        not failure_reasons
        and walker.output_dim == joint_map.action_dim
        and actual_visible_step_count >= 1
    )
    if not success and not failure_reasons:
        failure_reasons.append("walker_validation_incomplete")

    out: dict[str, Any] = {
        "lucky_locomotion_enabled": True,
        "policy_loaded": True,
        "walker_policy_path": str(LUCKY_WALKER_ONNX),
        "observation_dim": int(walker.input_dim),
        "action_dim": int(walker.output_dim),
        "joint_mapping_ok": bool(joint_map.mapping_ok and walker.output_dim == joint_map.action_dim),
        "controlled_joint_count": int(len(joint_map.controlled_indices)),
        "controlled_joints": [joint_map.joint_names[i] for i in joint_map.controlled_indices],
        "masked_arm_joint_count": int(len(joint_map.arm_indices)),
        "robot_fell": bool(robot_fell),
        "actual_visible_step_count": int(actual_visible_step_count),
        "pelvis_forward_progress_m": float(pelvis_forward),
        "average_forward_speed_mps": float(avg_speed),
        "actual_floor_contact_detected": bool(floor_contact_seen),
        "max_torso_pitch_rad": float(max_torso_pitch_rad),
        "max_torso_roll_rad": float(max_torso_roll_rad),
        "left_actual_foot_lift_m": float(max_left_lift),
        "right_actual_foot_lift_m": float(max_right_lift),
        "front_obstacles_disabled": bool(obstacle_info["front_obstacles_disabled"]),
        "disabled_obstacle_geom_count": int(obstacle_info["disabled_obstacle_geom_count"]),
        "obstacle_contact_count": int(max_obstacle_contacts),
        "wall_time_s": float(time.time() - start_wall),
        "sim_time_s": float(data.time),
        "success": bool(success),
        "failure_reason": None if success else "; ".join(failure_reasons),
    }
    if verbose:
        print("----- lucky_locomotion_validation -----")
        for key, value in out.items():
            print(f"{key}: {value}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Lucky G1 walker policy in MuJoCo.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--viewer-speed", type=float, default=1.0)
    ap.add_argument("--cmd-x", type=float, default=1.0)
    ap.add_argument("--cmd-y", type=float, default=0.0)
    ap.add_argument("--cmd-yaw", type=float, default=0.0)
    ns = ap.parse_args()
    run_g1_lucky_locomotion_validation(
        headless=bool(ns.headless),
        timeout=float(ns.timeout),
        viewer_speed=float(ns.viewer_speed),
        cmd_x=float(ns.cmd_x),
        cmd_y=float(ns.cmd_y),
        cmd_yaw=float(ns.cmd_yaw),
    )


if __name__ == "__main__":
    main()
