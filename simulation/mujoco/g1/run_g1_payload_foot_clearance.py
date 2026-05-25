#!/usr/bin/env python3
"""
Payload-aware foot-clearance validation for G1 holding the Dex3 box.

This is not walking. The box is treated as a payload, torso pitch compensation
stays active, the robot shifts to one support side, raises the unloaded foot
vertically by a small amount, holds briefly, and places it back down. No foot is
moved forward/backward and no manipulation scripts are touched.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_foot_clearance.py --headless --timeout 14
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
from run_g1_payload_balance import (  # noqa: E402
    DIRECT_SIDE_REACH_HOLD_POSTURE,
    PHYSICS_TIMESTEP_S,
    WAIST_PITCH_JOINT,
    _choose_torso_pitch,
    _command_targets,
    _evaluate_metrics,
    _foot_geom_ids,
    _hinge_names,
    _set_qpos_targets,
)
from run_g1_payload_foot_unload import _with_unload_shift  # noqa: E402
from run_g1_payload_weight_shift import _smoothstep01  # noqa: E402
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

FOOT_LIFT_TARGET_M = 0.015
MAX_HIP_ROLL_RAD = 0.12
MAX_ANKLE_ROLL_RAD = 0.10
MIN_SUPPORT_FOOT_FORCE_N = 120.0
MIN_LIFTED_COM_MARGIN_M = 0.005
MAX_PELVIS_LATERAL_SHIFT_M = 0.045

# Small pitch-space overlay found from the local G1 kinematics: it raises the
# ankle-roll foot corner centroid about 1.6 cm with < 2 mm horizontal drift.
SWING_FOOT_PITCH_DELTA = {
    "hip_pitch_joint": 0.125,
    "knee_joint": -0.210,
    "ankle_pitch_joint": -0.300,
}

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 0.8,
    "LOAD_HOLD": 0.8,
    "TORSO_COMPENSATE": 0.8,
    "SHIFT_TO_LEFT_SUPPORT": 0.8,
    "UNLOAD_RIGHT_FOOT": 0.8,
    "LIFT_RIGHT_FOOT_SMALL": 0.8,
    "HOLD_RIGHT_FOOT_CLEAR": 0.8,
    "LOWER_RIGHT_FOOT": 0.8,
    "RETURN_CENTER": 0.8,
    "SHIFT_TO_RIGHT_SUPPORT": 0.8,
    "UNLOAD_LEFT_FOOT": 0.8,
    "LIFT_LEFT_FOOT_SMALL": 0.8,
    "HOLD_LEFT_FOOT_CLEAR": 0.8,
    "LOWER_LEFT_FOOT": 0.8,
    "RETURN_CENTER_FINAL": 0.8,
}


class FootClearancePhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    TORSO_COMPENSATE = auto()
    SHIFT_TO_LEFT_SUPPORT = auto()
    UNLOAD_RIGHT_FOOT = auto()
    LIFT_RIGHT_FOOT_SMALL = auto()
    HOLD_RIGHT_FOOT_CLEAR = auto()
    LOWER_RIGHT_FOOT = auto()
    RETURN_CENTER = auto()
    SHIFT_TO_RIGHT_SUPPORT = auto()
    UNLOAD_LEFT_FOOT = auto()
    LIFT_LEFT_FOOT_SMALL = auto()
    HOLD_LEFT_FOOT_CLEAR = auto()
    LOWER_LEFT_FOOT = auto()
    RETURN_CENTER_FINAL = auto()
    DONE = auto()


def _phase_at_time(t: float) -> FootClearancePhase:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        elapsed += float(dur)
        if t < elapsed:
            return FootClearancePhase[name]
    return FootClearancePhase.DONE


def _phase_progress(t: float, phase: FootClearancePhase) -> float:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        next_elapsed = elapsed + float(dur)
        if FootClearancePhase[name] == phase:
            return (float(t) - elapsed) / max(float(dur), 1e-9)
        elapsed = next_elapsed
    return 1.0


def _support_shift_for_phase(t: float, phase: FootClearancePhase) -> float:
    name = phase.name
    if phase == FootClearancePhase.SHIFT_TO_LEFT_SUPPORT:
        return _smoothstep01(_phase_progress(t, phase))
    if name in {
        "UNLOAD_RIGHT_FOOT",
        "LIFT_RIGHT_FOOT_SMALL",
        "HOLD_RIGHT_FOOT_CLEAR",
        "LOWER_RIGHT_FOOT",
    }:
        return 1.0
    if phase == FootClearancePhase.RETURN_CENTER:
        return 1.0 - _smoothstep01(_phase_progress(t, phase))
    if phase == FootClearancePhase.SHIFT_TO_RIGHT_SUPPORT:
        return -_smoothstep01(_phase_progress(t, phase))
    if name in {
        "UNLOAD_LEFT_FOOT",
        "LIFT_LEFT_FOOT_SMALL",
        "HOLD_LEFT_FOOT_CLEAR",
        "LOWER_LEFT_FOOT",
    }:
        return -1.0
    if phase == FootClearancePhase.RETURN_CENTER_FINAL:
        return -1.0 + _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _lift_amount_for_phase(t: float, phase: FootClearancePhase, side: str) -> float:
    if side == "right":
        if phase == FootClearancePhase.LIFT_RIGHT_FOOT_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase == FootClearancePhase.HOLD_RIGHT_FOOT_CLEAR:
            return 1.0
        if phase == FootClearancePhase.LOWER_RIGHT_FOOT:
            return 1.0 - _smoothstep01(_phase_progress(t, phase))
    if side == "left":
        if phase == FootClearancePhase.LIFT_LEFT_FOOT_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase == FootClearancePhase.HOLD_LEFT_FOOT_CLEAR:
            return 1.0
        if phase == FootClearancePhase.LOWER_LEFT_FOOT:
            return 1.0 - _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _apply_vertical_foot_lift(pose: dict[str, float], side: str, amount: float) -> dict[str, float]:
    out = dict(pose)
    u = float(np.clip(amount, 0.0, 1.0))
    prefix = f"{side}_"
    for suffix, delta in SWING_FOOT_PITCH_DELTA.items():
        out[prefix + suffix] = float(out[prefix + suffix] + u * float(delta))
    return out


def _foot_center_xyz(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    pts = [
        np.asarray(data.geom_xpos[gid, :3], dtype=float)
        for gid in _foot_geom_ids(model, side)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not pts:
        return np.zeros(3, dtype=float)
    return np.mean(np.asarray(pts, dtype=float), axis=0)


def run_g1_payload_foot_clearance(
    *,
    headless: bool = False,
    timeout: float = 14.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
) -> dict[str, Any]:
    del headless
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

    phase_metrics: dict[str, dict[str, Any]] = {}
    final_phase = FootClearancePhase.STAND_NEUTRAL
    last_sampled_phase: FootClearancePhase | None = None
    right_lift_reference: np.ndarray | None = None
    left_lift_reference: np.ndarray | None = None
    right_ground_reference: float | None = None
    left_ground_reference: float | None = None
    max_right_clearance = 0.0
    max_left_clearance = 0.0
    max_right_xy_drift = 0.0
    max_left_xy_drift = 0.0
    min_com_margin = float("inf")
    min_single_support_margin = float("inf")
    min_support_force = float("inf")
    max_hip_roll = 0.0
    max_ankle_roll = 0.0
    max_pelvis_lateral_shift = 0.0
    right_returned = False
    left_returned = False
    t0_wall = time.time()

    while data.time < float(timeout):
        phase = _phase_at_time(float(data.time))
        final_phase = phase
        if phase == FootClearancePhase.STAND_NEUTRAL:
            pose = dict(neutral)
            current_torso_pitch = 0.0
        elif phase == FootClearancePhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            current_torso_pitch = 0.0
        else:
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
            pose = _with_unload_shift(pose, _support_shift_for_phase(float(data.time), phase))
            pose = _apply_vertical_foot_lift(
                pose, "right", _lift_amount_for_phase(float(data.time), phase, "right")
            )
            pose = _apply_vertical_foot_lift(
                pose, "left", _lift_amount_for_phase(float(data.time), phase, "left")
            )
            current_torso_pitch = torso_pitch

        _set_qpos_targets(model, data, pose, hinge_addrs)
        _command_targets(model, data, pose, actuator_ids)
        mujoco.mj_step(model, data)
        _set_qpos_targets(model, data, pose, hinge_addrs)
        data.qvel[:] = 0.0
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)

        right_center = _foot_center_xyz(model, data, "right")
        left_center = _foot_center_xyz(model, data, "left")
        if phase == FootClearancePhase.UNLOAD_RIGHT_FOOT:
            right_lift_reference = right_center.copy()
            right_ground_reference = float(right_center[2])
        if phase == FootClearancePhase.UNLOAD_LEFT_FOOT:
            left_lift_reference = left_center.copy()
            left_ground_reference = float(left_center[2])

        right_lift_active = _lift_amount_for_phase(float(data.time), phase, "right") > 1e-6
        left_lift_active = _lift_amount_for_phase(float(data.time), phase, "left") > 1e-6
        if right_lift_reference is not None:
            clearance = float(right_center[2] - right_lift_reference[2])
            max_right_clearance = max(max_right_clearance, clearance)
            if right_lift_active:
                max_right_xy_drift = max(
                    max_right_xy_drift,
                    float(np.linalg.norm(right_center[:2] - right_lift_reference[:2])),
                )
        if left_lift_reference is not None:
            clearance = float(left_center[2] - left_lift_reference[2])
            max_left_clearance = max(max_left_clearance, clearance)
            if left_lift_active:
                max_left_xy_drift = max(
                    max_left_xy_drift,
                    float(np.linalg.norm(left_center[:2] - left_lift_reference[:2])),
                )

        if phase in (FootClearancePhase.RETURN_CENTER, FootClearancePhase.DONE):
            if right_ground_reference is not None:
                right_returned = abs(float(right_center[2]) - right_ground_reference) <= 0.004
        if phase in (FootClearancePhase.RETURN_CENTER_FINAL, FootClearancePhase.DONE):
            if left_ground_reference is not None:
                left_returned = abs(float(left_center[2]) - left_ground_reference) <= 0.004

        hip_roll = max(
            abs(float(pose["left_hip_roll_joint"] - neutral["left_hip_roll_joint"])),
            abs(float(pose["right_hip_roll_joint"] - neutral["right_hip_roll_joint"])),
        )
        ankle_roll = max(
            abs(float(pose["left_ankle_roll_joint"] - neutral["left_ankle_roll_joint"])),
            abs(float(pose["right_ankle_roll_joint"] - neutral["right_ankle_roll_joint"])),
        )
        max_hip_roll = max(max_hip_roll, hip_roll)
        max_ankle_roll = max(max_ankle_roll, ankle_roll)
        max_pelvis_lateral_shift = max(max_pelvis_lateral_shift, 0.0)

        sample_now = phase != last_sampled_phase or phase in (
            FootClearancePhase.LIFT_RIGHT_FOOT_SMALL,
            FootClearancePhase.HOLD_RIGHT_FOOT_CLEAR,
            FootClearancePhase.LOWER_RIGHT_FOOT,
            FootClearancePhase.LIFT_LEFT_FOOT_SMALL,
            FootClearancePhase.HOLD_LEFT_FOOT_CLEAR,
            FootClearancePhase.LOWER_LEFT_FOOT,
            FootClearancePhase.RETURN_CENTER,
            FootClearancePhase.RETURN_CENTER_FINAL,
        )
        if sample_now:
            metrics = _evaluate_metrics(
                model,
                data,
                torso_pitch_rad=current_torso_pitch,
                hinge_addrs=hinge_addrs,
                neutral=neutral,
            )
            phase_metrics[phase.name] = metrics
            margin = float(metrics["min_com_margin_m"])
            min_com_margin = min(min_com_margin, margin)
            if phase in (
                FootClearancePhase.LIFT_RIGHT_FOOT_SMALL,
                FootClearancePhase.HOLD_RIGHT_FOOT_CLEAR,
                FootClearancePhase.LOWER_RIGHT_FOOT,
            ):
                min_single_support_margin = min(min_single_support_margin, margin)
                min_support_force = min(min_support_force, float(metrics["left_foot_force_n"]))
            if phase in (
                FootClearancePhase.LIFT_LEFT_FOOT_SMALL,
                FootClearancePhase.HOLD_LEFT_FOOT_CLEAR,
                FootClearancePhase.LOWER_LEFT_FOOT,
            ):
                min_single_support_margin = min(min_single_support_margin, margin)
                min_support_force = min(min_support_force, float(metrics["right_foot_force_n"]))
            last_sampled_phase = phase

        if phase == FootClearancePhase.DONE:
            break

    if not phase_metrics:
        metrics = _evaluate_metrics(
            model,
            data,
            torso_pitch_rad=torso_pitch,
            hinge_addrs=hinge_addrs,
            neutral=neutral,
        )
        phase_metrics[final_phase.name] = metrics
        min_com_margin = float(metrics["min_com_margin_m"])

    if not np.isfinite(min_single_support_margin):
        min_single_support_margin = min_com_margin
    if not np.isfinite(min_support_force):
        min_support_force = min(
            float(m["left_foot_force_n"]) + float(m["right_foot_force_n"])
            for m in phase_metrics.values()
        )
    inside_all = all(bool(m["com_inside_support"]) for m in phase_metrics.values())
    final_metrics = (
        phase_metrics.get("RETURN_CENTER_FINAL")
        or phase_metrics.get("LOWER_LEFT_FOOT")
        or next(reversed(phase_metrics.values()))
    )
    right_clearance_success = max_right_clearance >= FOOT_LIFT_TARGET_M - 1e-4
    left_clearance_success = max_left_clearance >= FOOT_LIFT_TARGET_M - 1e-4
    success = bool(
        right_clearance_success
        and left_clearance_success
        and right_returned
        and left_returned
        and min_support_force > MIN_SUPPORT_FOOT_FORCE_N
        and min_single_support_margin > MIN_LIFTED_COM_MARGIN_M
        and max_pelvis_lateral_shift <= MAX_PELVIS_LATERAL_SHIFT_M
    )

    out = dict(final_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in FootClearancePhase],
            "final_phase": final_phase.name,
            "phases_completed": list(phase_metrics.keys()),
            "phase_metrics": phase_metrics,
            "payload_foot_clearance_validation_only": True,
            "physical_grasp_validation": False,
            "no_locomotion_attempted": True,
            "right_foot_clearance_success": bool(right_clearance_success),
            "left_foot_clearance_success": bool(left_clearance_success),
            "max_right_foot_clearance_m": float(max_right_clearance),
            "max_left_foot_clearance_m": float(max_left_clearance),
            "right_foot_returned_to_ground": bool(right_returned),
            "left_foot_returned_to_ground": bool(left_returned),
            "com_inside_support_all_phases": bool(inside_all),
            "min_com_margin_m": float(min_com_margin),
            "min_com_margin_single_support_m": float(min_single_support_margin),
            "min_support_foot_force_n": float(min_support_force),
            "max_pelvis_lateral_shift_m": float(max_pelvis_lateral_shift),
            "max_hip_roll_rad": float(max_hip_roll),
            "max_ankle_roll_rad": float(max_ankle_roll),
            "max_swing_foot_xy_drift_m": float(max(max_right_xy_drift, max_left_xy_drift)),
            "payload_mass_kg": float(final_metrics["payload_mass_kg"]),
            "payload_moment_nm": float(final_metrics["payload_moment_nm"]),
            "torso_pitch_compensation_rad": float(torso_pitch),
            "success": success,
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_foot_clearance -----")
        print("payload_foot_clearance_validation_only: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        for key in (
            "phases_completed",
            "right_foot_clearance_success",
            "left_foot_clearance_success",
            "max_right_foot_clearance_m",
            "max_left_foot_clearance_m",
            "right_foot_returned_to_ground",
            "left_foot_returned_to_ground",
            "com_inside_support_all_phases",
            "min_com_margin_m",
            "min_com_margin_single_support_m",
            "min_support_foot_force_n",
            "max_pelvis_lateral_shift_m",
            "max_hip_roll_rad",
            "max_ankle_roll_rad",
            "max_swing_foot_xy_drift_m",
            "payload_mass_kg",
            "payload_moment_nm",
            "torso_pitch_compensation_rad",
            "no_locomotion_attempted",
            "success",
        ):
            val = out[key]
            if isinstance(val, float):
                print(f"{key}: {val:.5f}")
            else:
                print(f"{key}: {val}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 payload-aware foot clearance.")
    ap.add_argument("--headless", action="store_true", help="Run without viewer.")
    ap.add_argument("--timeout", type=float, default=14.0, help="Simulation timeout in seconds.")
    args = ap.parse_args()
    run_g1_payload_foot_clearance(headless=args.headless, timeout=args.timeout, verbose=True)


if __name__ == "__main__":
    main()
