#!/usr/bin/env python3
"""
Load-aware standing balance validation for G1 holding the Dex3 box.

This milestone reuses the Dex3 box scene and treats the current direct-side
reach hold as a payload state. It is a whole-body balance validation only:
the box is kept on the scripted vertical slide / scene pose, so this does not
validate passive physical finger grasp quality.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_balance.py --headless --timeout 8
"""

from __future__ import annotations

import argparse
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

from g1_dex3_finger_control import Dex3FingerController, merge_dex3_neutral_into  # noqa: E402
from run_g1_dual_arm_box import _load_dual_scene_model  # noqa: E402
from run_g1_posture_hold import (  # noqa: E402
    DEFAULT_PELVIS_Z,
    NEUTRAL_POSTURE,
    apply_neutral_pose,
    build_actuator_id_map,
    build_hinge_joint_address_map,
    command_position_actuators,
    floating_base_address_map,
    stabilize_floating_base,
)

BOX_BODY = "reach_target_box"
BOX_GEOM = "box_geom"
FLOOR_GEOM = "reach_floor"
WAIST_PITCH_JOINT = "waist_pitch_joint"
PHYSICS_TIMESTEP_S = 0.1
GRAVITY = 9.80665

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 1.0,
    "LOAD_HOLD": 1.5,
    "TORSO_COMPENSATE": 1.0,
    "HOLD_COMPENSATED": 2.0,
}

DIRECT_SIDE_REACH_HOLD_POSTURE: dict[str, float] = {
    # Approximate reusable side-hold posture from the direct_side_reach milestone.
    # Payload-balance uses it as a static arm load case; finger manipulation is
    # intentionally not tuned here.
    "left_shoulder_pitch_joint": 0.34,
    "left_shoulder_roll_joint": 0.34,
    "left_shoulder_yaw_joint": -0.18,
    "left_elbow_joint": 1.28,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": -0.34,
    "left_wrist_yaw_joint": -0.32,
    "right_shoulder_pitch_joint": 0.34,
    "right_shoulder_roll_joint": -0.34,
    "right_shoulder_yaw_joint": 0.18,
    "right_elbow_joint": 1.28,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": -0.34,
    "right_wrist_yaw_joint": 0.32,
}

LEG_JOINTS: tuple[str, ...] = (
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


class BalancePhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    TORSO_COMPENSATE = auto()
    HOLD_COMPENSATED = auto()
    DONE = auto()


def _hinge_names(model: mujoco.MjModel) -> set[str]:
    names: set[str] = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            names.add(name)
    return names


def _phase_at_time(t: float) -> BalancePhase:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        elapsed += float(dur)
        if t < elapsed:
            return BalancePhase[name]
    return BalancePhase.DONE


def _set_qpos_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: dict[str, float],
    hinge_addrs: dict[str, dict[str, int]],
) -> None:
    for jname, q in targets.items():
        if jname not in hinge_addrs:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        data.qpos[int(hinge_addrs[jname]["qpos_adr"])] = float(np.clip(q, lo, hi))


def _command_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: dict[str, float],
    actuator_ids: dict[str, int],
) -> None:
    command_position_actuators(model, data, targets, actuator_ids)


def _body_com(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    include_body_ids: set[int],
) -> tuple[np.ndarray, float]:
    mass = 0.0
    weighted = np.zeros(3, dtype=float)
    for bid in include_body_ids:
        m = float(model.body_mass[bid])
        if m <= 0.0:
            continue
        mass += m
        weighted += m * np.asarray(data.xipos[bid, :3], dtype=float)
    if mass <= 0.0:
        return np.zeros(3, dtype=float), 0.0
    return weighted / mass, mass


def _robot_and_box_coms(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY)
    robot_ids = {bid for bid in range(1, model.nbody) if bid != box_bid}
    robot_com, robot_mass = _body_com(model, data, include_body_ids=robot_ids)
    box_com, box_mass = _body_com(model, data, include_body_ids={box_bid})
    total_mass = robot_mass + box_mass
    combined = (robot_com * robot_mass + box_com * box_mass) / total_mass
    return {
        "robot_com": robot_com,
        "robot_mass_kg": robot_mass,
        "box_com": box_com,
        "box_mass_kg": box_mass,
        "combined_com": combined,
        "combined_mass_kg": total_mass,
    }


def _foot_body_ids(model: mujoco.MjModel, side: str) -> set[int]:
    body_name = f"{side}_ankle_roll_link"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return {int(bid)} if bid >= 0 else set()


def _foot_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    bids = _foot_body_ids(model, side)
    out: list[int] = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) in bids:
            out.append(gid)
    return out


def _foot_contact_force_n(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
) -> float:
    foot_gids = set(_foot_geom_ids(model, side))
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    total = 0.0
    force6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        con = data.contact[i]
        if floor_gid not in (int(con.geom1), int(con.geom2)):
            continue
        if int(con.geom1) not in foot_gids and int(con.geom2) not in foot_gids:
            continue
        mujoco.mj_contactForce(model, data, i, force6)
        total += abs(float(force6[0]))
    return total


def _foot_center_xy(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    pts = [
        np.asarray(data.geom_xpos[gid, :2], dtype=float)
        for gid in _foot_geom_ids(model, side)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not pts:
        return np.zeros(2, dtype=float)
    return np.mean(np.asarray(pts, dtype=float), axis=0)


def _support_polygon_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, Any]:
    points: list[np.ndarray] = []
    for side in ("left", "right"):
        for gid in _foot_geom_ids(model, side):
            # The four small sphere geoms mark each foot corner in this MJCF.
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
                continue
            r = float(model.geom_size[gid, 0])
            p = np.asarray(data.geom_xpos[gid, :2], dtype=float)
            points.extend(
                [
                    p + np.array([r, r]),
                    p + np.array([r, -r]),
                    p + np.array([-r, r]),
                    p + np.array([-r, -r]),
                ]
            )
    pts = np.asarray(points, dtype=float)
    xy_min = np.min(pts, axis=0)
    xy_max = np.max(pts, axis=0)
    corners = np.array(
        [
            [xy_min[0], xy_min[1]],
            [xy_max[0], xy_min[1]],
            [xy_max[0], xy_max[1]],
            [xy_min[0], xy_max[1]],
        ],
        dtype=float,
    )
    return {
        "xy_min": xy_min,
        "xy_max": xy_max,
        "corners": corners,
        "center": 0.5 * (xy_min + xy_max),
    }


def _support_margin(com_xy: np.ndarray, support: dict[str, Any]) -> tuple[bool, float]:
    xy_min = np.asarray(support["xy_min"], dtype=float)
    xy_max = np.asarray(support["xy_max"], dtype=float)
    margins = np.array(
        [
            com_xy[0] - xy_min[0],
            xy_max[0] - com_xy[0],
            com_xy[1] - xy_min[1],
            xy_max[1] - com_xy[1],
        ],
        dtype=float,
    )
    inside = bool(np.all(margins >= 0.0))
    if inside:
        return inside, float(np.min(margins))
    outside = np.maximum(-margins, 0.0)
    return inside, -float(np.max(outside))


def _leg_delta_from_neutral(
    data: mujoco.MjData,
    hinge_addrs: dict[str, dict[str, int]],
    neutral: dict[str, float],
) -> float:
    return float(
        max(
            abs(float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])]) - float(neutral[jn]))
            for jn in LEG_JOINTS
        )
    )


def _evaluate_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    torso_pitch_rad: float,
    hinge_addrs: dict[str, dict[str, int]],
    neutral: dict[str, float],
) -> dict[str, Any]:
    coms = _robot_and_box_coms(model, data)
    support = _support_polygon_aabb(model, data)
    inside, margin = _support_margin(np.asarray(coms["combined_com"][:2]), support)
    left_force = _foot_contact_force_n(model, data, side="left")
    right_force = _foot_contact_force_n(model, data, side="right")
    if left_force + right_force <= 1e-9:
        left_center = _foot_center_xy(model, data, "left")
        right_center = _foot_center_xy(model, data, "right")
        span_y = float(left_center[1] - right_center[1])
        left_frac = 0.5
        if abs(span_y) > 1e-9:
            left_frac = float(
                np.clip((float(coms["combined_com"][1]) - right_center[1]) / span_y, 0.0, 1.0)
            )
        total_weight = float(coms["combined_mass_kg"] * GRAVITY)
        left_force = total_weight * left_frac
        right_force = total_weight - left_force
    force_sum = left_force + right_force
    force_ratio = float(left_force / force_sum) if force_sum > 1e-9 else 0.5
    payload_arm_x = float(coms["box_com"][0] - support["center"][0])
    payload_moment = float(coms["box_mass_kg"] * GRAVITY * payload_arm_x)
    suggested = float(
        np.clip(
            -payload_moment / max(float(coms["robot_mass_kg"]) * GRAVITY * 0.55, 1e-9),
            -0.12,
            0.12,
        )
    )
    return {
        "robot_com": np.asarray(coms["robot_com"], dtype=float).tolist(),
        "box_com": np.asarray(coms["box_com"], dtype=float).tolist(),
        "combined_robot_box_com": np.asarray(coms["combined_com"], dtype=float).tolist(),
        "support_polygon_xy": np.asarray(support["corners"], dtype=float).tolist(),
        "com_inside_support": bool(inside),
        "min_com_margin_m": float(margin),
        "max_torso_pitch_rad": abs(float(torso_pitch_rad)),
        "left_foot_force_n": float(left_force),
        "right_foot_force_n": float(right_force),
        "foot_force_balance_ratio": float(force_ratio),
        "payload_mass_kg": float(coms["box_mass_kg"]),
        "payload_moment_nm": float(payload_moment),
        "forward_pitch_moment_from_payload_nm": float(payload_moment),
        "suggested_torso_pitch_compensation_rad": float(suggested),
        "torso_pitch_rad": float(torso_pitch_rad),
        "leg_joint_max_delta_from_neutral_rad": _leg_delta_from_neutral(
            data, hinge_addrs, neutral
        ),
    }


def _choose_torso_pitch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_pose: dict[str, float],
    actuator_ids: dict[str, int],
    hinge_addrs: dict[str, dict[str, int]],
    neutral: dict[str, float],
    nominal_base: np.ndarray,
    base_map: dict[str, int],
) -> float:
    candidates = np.linspace(-0.12, 0.12, 13)
    best_pitch = 0.0
    best_margin = -float("inf")
    for pitch in candidates:
        pose = dict(base_pose)
        pose[WAIST_PITCH_JOINT] = float(pitch)
        _set_qpos_targets(model, data, pose, hinge_addrs)
        _command_targets(model, data, pose, actuator_ids)
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)
        metrics = _evaluate_metrics(
            model,
            data,
            torso_pitch_rad=float(pitch),
            hinge_addrs=hinge_addrs,
            neutral=neutral,
        )
        margin = float(metrics["min_com_margin_m"])
        if margin > best_margin:
            best_margin = margin
            best_pitch = float(pitch)
    return best_pitch


def run_g1_payload_balance(
    *,
    headless: bool = False,
    timeout: float = 8.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
) -> dict[str, Any]:
    del headless  # Viewer mode is intentionally not opened by this validation script yet.
    model = _load_dual_scene_model()
    if float(model.opt.timestep) < PHYSICS_TIMESTEP_S:
        model.opt.timestep = PHYSICS_TIMESTEP_S

    data = mujoco.MjData(model)
    base_map = floating_base_address_map(model)
    actuator_ids = build_actuator_id_map(model)
    hinge_addrs = build_hinge_joint_address_map(model)
    neutral = merge_dex3_neutral_into(dict(NEUTRAL_POSTURE), hinge_joint_names=_hinge_names(model))

    nominal_base = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=neutral,
        base_map=base_map,
    )
    command_position_actuators(model, data, neutral, actuator_ids)

    finger_hold = Dex3FingerController().targets_for_mode("side_support_grasp")
    load_hold_pose = dict(neutral)
    load_hold_pose.update(DIRECT_SIDE_REACH_HOLD_POSTURE)
    load_hold_pose.update(finger_hold)

    neutral_metrics: dict[str, Any] | None = None
    load_hold_metrics: dict[str, Any] | None = None
    compensated_metrics: dict[str, Any] | None = None
    final_phase = BalancePhase.STAND_NEUTRAL
    last_sampled_phase: BalancePhase | None = None
    torso_pitch = 0.0
    compensation_selected = False
    t0_wall = time.time()

    while data.time < float(timeout):
        phase = _phase_at_time(float(data.time))
        final_phase = phase
        if phase == BalancePhase.STAND_NEUTRAL:
            pose = dict(neutral)
            torso_pitch = 0.0
        elif phase == BalancePhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            torso_pitch = 0.0
        elif phase in (BalancePhase.TORSO_COMPENSATE, BalancePhase.HOLD_COMPENSATED):
            if not compensation_selected:
                torso_pitch = _choose_torso_pitch(
                    model,
                    data,
                    base_pose=load_hold_pose,
                    actuator_ids=actuator_ids,
                    hinge_addrs=hinge_addrs,
                    neutral=neutral,
                    nominal_base=nominal_base,
                    base_map=base_map,
                )
                compensation_selected = True
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
        else:
            break

        _set_qpos_targets(model, data, pose, hinge_addrs)
        _command_targets(model, data, pose, actuator_ids)
        mujoco.mj_step(model, data)
        _set_qpos_targets(model, data, pose, hinge_addrs)
        data.qvel[:] = 0.0
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)

        sample_now = phase != last_sampled_phase
        if sample_now:
            metrics = _evaluate_metrics(
                model,
                data,
                torso_pitch_rad=torso_pitch,
                hinge_addrs=hinge_addrs,
                neutral=neutral,
            )
            if phase == BalancePhase.STAND_NEUTRAL:
                neutral_metrics = metrics
            elif phase == BalancePhase.LOAD_HOLD:
                load_hold_metrics = metrics
            elif phase == BalancePhase.HOLD_COMPENSATED:
                compensated_metrics = metrics
            last_sampled_phase = phase

        if phase == BalancePhase.DONE:
            break

    if compensated_metrics is None:
        compensated_metrics = _evaluate_metrics(
            model,
            data,
            torso_pitch_rad=torso_pitch,
            hinge_addrs=hinge_addrs,
            neutral=neutral,
        )
    if load_hold_metrics is None:
        load_hold_metrics = compensated_metrics
    if neutral_metrics is None:
        neutral_metrics = load_hold_metrics

    margin_before = float(load_hold_metrics["min_com_margin_m"])
    margin_after = float(compensated_metrics["min_com_margin_m"])
    out = dict(compensated_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in BalancePhase],
            "final_phase": final_phase.name,
            "scene": "g1_reach_box_scene_dex3.xml",
            "payload_balance_validation_only": True,
            "physical_grasp_validation": False,
            "box_payload_mode": "scripted_vertical_slide_scene_pose",
            "no_locomotion_attempted": True,
            "com_margin_before_torso_comp_m": margin_before,
            "com_margin_after_torso_comp_m": margin_after,
            "torso_compensation_improved_margin": bool(margin_after > margin_before + 1e-6),
            "neutral_com_margin_m": float(neutral_metrics["min_com_margin_m"]),
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_balance -----")
        print("payload_balance_validation_only: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        print(f"robot_com: {out['robot_com']}")
        print(f"box_com: {out['box_com']}")
        print(f"combined_robot_box_com: {out['combined_robot_box_com']}")
        print(f"left_foot_force_n: {out['left_foot_force_n']:.3f}")
        print(f"right_foot_force_n: {out['right_foot_force_n']:.3f}")
        print(f"support_polygon_xy: {out['support_polygon_xy']}")
        print(f"com_inside_support: {out['com_inside_support']}")
        print(f"min_com_margin_m: {out['min_com_margin_m']:.5f}")
        print(
            "forward_pitch_moment_from_payload_nm: "
            f"{out['forward_pitch_moment_from_payload_nm']:.3f}"
        )
        print(
            "suggested_torso_pitch_compensation_rad: "
            f"{out['suggested_torso_pitch_compensation_rad']:.5f}"
        )
        for key in (
            "max_torso_pitch_rad",
            "foot_force_balance_ratio",
            "payload_mass_kg",
            "payload_moment_nm",
            "com_margin_before_torso_comp_m",
            "com_margin_after_torso_comp_m",
            "torso_compensation_improved_margin",
            "no_locomotion_attempted",
        ):
            val = out[key]
            if isinstance(val, float):
                print(f"{key}: {val:.5f}")
            else:
                print(f"{key}: {val}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 load-aware standing balance with Dex3 box payload.")
    ap.add_argument("--headless", action="store_true", help="Run without viewer.")
    ap.add_argument("--timeout", type=float, default=8.0, help="Simulation timeout in seconds.")
    args = ap.parse_args()
    run_g1_payload_balance(headless=args.headless, timeout=args.timeout, verbose=True)


if __name__ == "__main__":
    main()
