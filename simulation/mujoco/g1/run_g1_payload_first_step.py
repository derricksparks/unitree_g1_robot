#!/usr/bin/env python3
"""
Payload-aware quasi-static first-step validation for G1 holding the Dex3 box.

This is not continuous walking. It performs a tiny right-foot forward step
after payload-aware unload/clearance, verifies double support, then repeats
with the left foot only if the right step is safe. The box remains a payload
validation object; no physical grasp validation, RL, or manipulation tuning is
introduced.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_first_step.py --headless --timeout 18
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
from run_g1_payload_foot_clearance import (  # noqa: E402
    SWING_FOOT_PITCH_DELTA,
    _foot_center_xyz,
    _support_shift_for_phase as _clearance_support_shift,
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

STEP_FORWARD_DELTA = {
    "hip_pitch_joint": -0.033333333333333326,
    "knee_joint": 0.0,
    "ankle_pitch_joint": 0.06666666666666665,
}

MIN_STEP_LENGTH_M = 0.015
MAX_STEP_LENGTH_M = 0.025
MIN_FOOT_CLEARANCE_M = 0.015
MAX_SWING_LATERAL_DRIFT_M = 0.006
MIN_SUPPORT_FOOT_FORCE_N = 180.0
MIN_DOUBLE_SUPPORT_MARGIN_M = 0.015
MIN_SINGLE_SUPPORT_MARGIN_M = 0.005
MAX_PELVIS_LATERAL_SHIFT_M = 0.05

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 0.8,
    "LOAD_HOLD": 0.8,
    "TORSO_COMPENSATE": 0.8,
    "SHIFT_TO_LEFT_SUPPORT": 0.8,
    "UNLOAD_RIGHT_FOOT": 0.8,
    "LIFT_RIGHT_FOOT_SMALL": 0.8,
    "MOVE_RIGHT_FOOT_FORWARD_SMALL": 0.8,
    "LOWER_RIGHT_FOOT": 0.8,
    "RIGHT_STEP_DOUBLE_SUPPORT_HOLD": 0.8,
    "RETURN_CENTER": 0.8,
    "SHIFT_TO_RIGHT_SUPPORT": 0.8,
    "UNLOAD_LEFT_FOOT": 0.8,
    "LIFT_LEFT_FOOT_SMALL": 0.8,
    "MOVE_LEFT_FOOT_FORWARD_SMALL": 0.8,
    "LOWER_LEFT_FOOT": 0.8,
    "LEFT_STEP_DOUBLE_SUPPORT_HOLD": 0.8,
    "RETURN_CENTER_FINAL": 0.8,
}


class FirstStepPhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    TORSO_COMPENSATE = auto()
    SHIFT_TO_LEFT_SUPPORT = auto()
    UNLOAD_RIGHT_FOOT = auto()
    LIFT_RIGHT_FOOT_SMALL = auto()
    MOVE_RIGHT_FOOT_FORWARD_SMALL = auto()
    LOWER_RIGHT_FOOT = auto()
    RIGHT_STEP_DOUBLE_SUPPORT_HOLD = auto()
    RETURN_CENTER = auto()
    SHIFT_TO_RIGHT_SUPPORT = auto()
    UNLOAD_LEFT_FOOT = auto()
    LIFT_LEFT_FOOT_SMALL = auto()
    MOVE_LEFT_FOOT_FORWARD_SMALL = auto()
    LOWER_LEFT_FOOT = auto()
    LEFT_STEP_DOUBLE_SUPPORT_HOLD = auto()
    RETURN_CENTER_FINAL = auto()
    DONE = auto()


def _phase_at_time(t: float) -> FirstStepPhase:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        elapsed += float(dur)
        if t < elapsed:
            return FirstStepPhase[name]
    return FirstStepPhase.DONE


def _phase_progress(t: float, phase: FirstStepPhase) -> float:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        next_elapsed = elapsed + float(dur)
        if FirstStepPhase[name] == phase:
            return (float(t) - elapsed) / max(float(dur), 1e-9)
        elapsed = next_elapsed
    return 1.0


def _support_shift_for_phase(t: float, phase: FirstStepPhase) -> float:
    name = phase.name
    if name in {
        "MOVE_RIGHT_FOOT_FORWARD_SMALL",
        "LOWER_RIGHT_FOOT",
        "RIGHT_STEP_DOUBLE_SUPPORT_HOLD",
    }:
        return 1.0
    if name in {
        "MOVE_LEFT_FOOT_FORWARD_SMALL",
        "LOWER_LEFT_FOOT",
        "LEFT_STEP_DOUBLE_SUPPORT_HOLD",
    }:
        return -1.0
    try:
        clearance_phase = type("P", (), {"name": name})()
        return _clearance_support_shift(t, clearance_phase)
    except Exception:
        return 0.0


def _lift_amount_for_phase(t: float, phase: FirstStepPhase, side: str) -> float:
    if side == "right":
        if phase == FirstStepPhase.LIFT_RIGHT_FOOT_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase == FirstStepPhase.MOVE_RIGHT_FOOT_FORWARD_SMALL:
            return 1.0
        if phase == FirstStepPhase.LOWER_RIGHT_FOOT:
            return 1.0 - _smoothstep01(_phase_progress(t, phase))
    if side == "left":
        if phase == FirstStepPhase.LIFT_LEFT_FOOT_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase == FirstStepPhase.MOVE_LEFT_FOOT_FORWARD_SMALL:
            return 1.0
        if phase == FirstStepPhase.LOWER_LEFT_FOOT:
            return 1.0 - _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _forward_amount_for_phase(t: float, phase: FirstStepPhase, side: str) -> float:
    if side == "right":
        if phase == FirstStepPhase.MOVE_RIGHT_FOOT_FORWARD_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase in (
            FirstStepPhase.LOWER_RIGHT_FOOT,
            FirstStepPhase.RIGHT_STEP_DOUBLE_SUPPORT_HOLD,
            FirstStepPhase.RETURN_CENTER,
        ):
            return 1.0
    if side == "left":
        if phase == FirstStepPhase.MOVE_LEFT_FOOT_FORWARD_SMALL:
            return _smoothstep01(_phase_progress(t, phase))
        if phase in (
            FirstStepPhase.LOWER_LEFT_FOOT,
            FirstStepPhase.LEFT_STEP_DOUBLE_SUPPORT_HOLD,
            FirstStepPhase.RETURN_CENTER_FINAL,
        ):
            return 1.0
    return 0.0


def _apply_leg_pitch_overlay(
    pose: dict[str, float],
    side: str,
    overlay: dict[str, float],
    amount: float,
) -> dict[str, float]:
    out = dict(pose)
    u = float(np.clip(amount, 0.0, 1.0))
    prefix = f"{side}_"
    for suffix, delta in overlay.items():
        out[prefix + suffix] = float(out[prefix + suffix] + u * float(delta))
    return out


def _foot_grounded(center: np.ndarray, reference_z: float | None) -> bool:
    if reference_z is None:
        return False
    return abs(float(center[2]) - float(reference_z)) <= 0.005


def run_g1_payload_first_step(
    *,
    headless: bool = False,
    timeout: float = 18.0,
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
    final_phase = FirstStepPhase.STAND_NEUTRAL
    last_sampled_phase: FirstStepPhase | None = None
    right_ref: np.ndarray | None = None
    left_ref: np.ndarray | None = None
    right_ground_z: float | None = None
    left_ground_z: float | None = None
    max_right_clearance = 0.0
    max_left_clearance = 0.0
    right_step_length = 0.0
    left_step_length = 0.0
    max_lateral_drift = 0.0
    right_foot_landed = False
    left_foot_landed = False
    right_step_success = False
    left_step_success = False
    left_step_attempted = False
    left_step_skipped_reason = ""
    min_com_margin = float("inf")
    min_single_margin = float("inf")
    min_double_landing_margin = float("inf")
    min_support_force = float("inf")
    max_pelvis_lateral_shift = 0.0
    right_double_support_safe = False
    t0_wall = time.time()

    while data.time < float(timeout):
        phase = _phase_at_time(float(data.time))
        if not right_double_support_safe and phase.value >= FirstStepPhase.SHIFT_TO_RIGHT_SUPPORT.value:
            final_phase = phase
            left_step_skipped_reason = "right_double_support_not_verified"
            break

        final_phase = phase
        if phase == FirstStepPhase.STAND_NEUTRAL:
            pose = dict(neutral)
            current_torso_pitch = 0.0
        elif phase == FirstStepPhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            current_torso_pitch = 0.0
        else:
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
            pose = _with_unload_shift(pose, _support_shift_for_phase(float(data.time), phase))
            pose = _apply_leg_pitch_overlay(
                pose, "right", SWING_FOOT_PITCH_DELTA, _lift_amount_for_phase(float(data.time), phase, "right")
            )
            pose = _apply_leg_pitch_overlay(
                pose, "right", STEP_FORWARD_DELTA, _forward_amount_for_phase(float(data.time), phase, "right")
            )
            pose = _apply_leg_pitch_overlay(
                pose, "left", SWING_FOOT_PITCH_DELTA, _lift_amount_for_phase(float(data.time), phase, "left")
            )
            pose = _apply_leg_pitch_overlay(
                pose, "left", STEP_FORWARD_DELTA, _forward_amount_for_phase(float(data.time), phase, "left")
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
        if phase == FirstStepPhase.UNLOAD_RIGHT_FOOT:
            right_ref = right_center.copy()
            right_ground_z = float(right_center[2])
        if phase == FirstStepPhase.UNLOAD_LEFT_FOOT:
            left_step_attempted = True
            left_ref = left_center.copy()
            left_ground_z = float(left_center[2])

        if right_ref is not None:
            max_right_clearance = max(max_right_clearance, float(right_center[2] - right_ref[2]))
            right_step_length = max(right_step_length, float(right_center[0] - right_ref[0]))
            if phase in (
                FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
                FirstStepPhase.MOVE_RIGHT_FOOT_FORWARD_SMALL,
                FirstStepPhase.LOWER_RIGHT_FOOT,
            ):
                max_lateral_drift = max(max_lateral_drift, abs(float(right_center[1] - right_ref[1])))
        if left_ref is not None:
            max_left_clearance = max(max_left_clearance, float(left_center[2] - left_ref[2]))
            left_step_length = max(left_step_length, float(left_center[0] - left_ref[0]))
            if phase in (
                FirstStepPhase.LIFT_LEFT_FOOT_SMALL,
                FirstStepPhase.MOVE_LEFT_FOOT_FORWARD_SMALL,
                FirstStepPhase.LOWER_LEFT_FOOT,
            ):
                max_lateral_drift = max(max_lateral_drift, abs(float(left_center[1] - left_ref[1])))

        if phase == FirstStepPhase.RIGHT_STEP_DOUBLE_SUPPORT_HOLD:
            right_foot_landed = _foot_grounded(right_center, right_ground_z)
        if phase == FirstStepPhase.LEFT_STEP_DOUBLE_SUPPORT_HOLD:
            left_foot_landed = _foot_grounded(left_center, left_ground_z)

        sample_now = phase != last_sampled_phase or phase in (
            FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
            FirstStepPhase.MOVE_RIGHT_FOOT_FORWARD_SMALL,
            FirstStepPhase.LOWER_RIGHT_FOOT,
            FirstStepPhase.RIGHT_STEP_DOUBLE_SUPPORT_HOLD,
            FirstStepPhase.LIFT_LEFT_FOOT_SMALL,
            FirstStepPhase.MOVE_LEFT_FOOT_FORWARD_SMALL,
            FirstStepPhase.LOWER_LEFT_FOOT,
            FirstStepPhase.LEFT_STEP_DOUBLE_SUPPORT_HOLD,
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
                FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
                FirstStepPhase.MOVE_RIGHT_FOOT_FORWARD_SMALL,
                FirstStepPhase.LOWER_RIGHT_FOOT,
            ):
                min_single_margin = min(min_single_margin, margin)
                min_support_force = min(min_support_force, float(metrics["left_foot_force_n"]))
            if phase in (
                FirstStepPhase.LIFT_LEFT_FOOT_SMALL,
                FirstStepPhase.MOVE_LEFT_FOOT_FORWARD_SMALL,
                FirstStepPhase.LOWER_LEFT_FOOT,
            ):
                min_single_margin = min(min_single_margin, margin)
                min_support_force = min(min_support_force, float(metrics["right_foot_force_n"]))
            if phase == FirstStepPhase.RIGHT_STEP_DOUBLE_SUPPORT_HOLD:
                min_double_landing_margin = min(min_double_landing_margin, margin)
                right_step_success = bool(
                    right_foot_landed
                    and right_step_length >= MIN_STEP_LENGTH_M
                    and right_step_length <= MAX_STEP_LENGTH_M
                    and max_right_clearance >= MIN_FOOT_CLEARANCE_M
                    and margin > MIN_DOUBLE_SUPPORT_MARGIN_M
                )
                right_double_support_safe = right_step_success
            if phase == FirstStepPhase.LEFT_STEP_DOUBLE_SUPPORT_HOLD:
                min_double_landing_margin = min(min_double_landing_margin, margin)
                left_step_success = bool(
                    left_foot_landed
                    and left_step_length >= MIN_STEP_LENGTH_M
                    and left_step_length <= MAX_STEP_LENGTH_M
                    and max_left_clearance >= MIN_FOOT_CLEARANCE_M
                    and margin > MIN_DOUBLE_SUPPORT_MARGIN_M
                )
            last_sampled_phase = phase

        if phase == FirstStepPhase.DONE:
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
    if not np.isfinite(min_single_margin):
        min_single_margin = min_com_margin
    if not np.isfinite(min_double_landing_margin):
        min_double_landing_margin = min_com_margin
    if not np.isfinite(min_support_force):
        min_support_force = min(
            float(m["left_foot_force_n"]) + float(m["right_foot_force_n"])
            for m in phase_metrics.values()
        )
    if left_step_attempted and not left_step_success and not left_step_skipped_reason:
        left_step_skipped_reason = "left_step_attempted_but_not_successful"
    if not left_step_attempted and not left_step_skipped_reason:
        left_step_skipped_reason = "not_reached_before_timeout"

    inside_all = all(bool(m["com_inside_support"]) for m in phase_metrics.values())
    final_metrics = next(reversed(phase_metrics.values()))
    success = bool(
        right_step_success
        and right_step_length >= MIN_STEP_LENGTH_M
        and right_foot_landed
        and max_right_clearance >= MIN_FOOT_CLEARANCE_M
        and inside_all
        and min_support_force > MIN_SUPPORT_FOOT_FORCE_N
        and min_single_margin > MIN_SINGLE_SUPPORT_MARGIN_M
        and min_double_landing_margin > MIN_DOUBLE_SUPPORT_MARGIN_M
        and max_lateral_drift < MAX_SWING_LATERAL_DRIFT_M
    )

    out = dict(final_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in FirstStepPhase],
            "final_phase": final_phase.name,
            "phases_completed": list(phase_metrics.keys()),
            "phase_metrics": phase_metrics,
            "payload_first_step_validation_only": True,
            "physical_grasp_validation": False,
            "no_continuous_walking_attempted": True,
            "no_rl_used": True,
            "right_step_attempted": bool(right_ref is not None),
            "right_step_success": bool(right_step_success),
            "right_step_failure_reason": "" if right_step_success else "right_step_safety_check_failed",
            "left_step_attempted": bool(left_step_attempted),
            "left_step_success": bool(left_step_success),
            "left_step_skipped_reason": "" if left_step_success else left_step_skipped_reason,
            "right_step_length_m": float(right_step_length),
            "left_step_length_m": float(left_step_length),
            "max_right_foot_clearance_m": float(max_right_clearance),
            "max_left_foot_clearance_m": float(max_left_clearance),
            "max_swing_foot_lateral_drift_m": float(max_lateral_drift),
            "right_foot_landed": bool(right_foot_landed),
            "left_foot_landed": bool(left_foot_landed),
            "com_inside_support_all_phases": bool(inside_all),
            "min_com_margin_m": float(min_com_margin),
            "min_com_margin_single_support_m": float(min_single_margin),
            "min_com_margin_double_support_after_landing_m": float(min_double_landing_margin),
            "min_support_foot_force_n": float(min_support_force),
            "max_pelvis_lateral_shift_m": float(max_pelvis_lateral_shift),
            "payload_mass_kg": float(final_metrics["payload_mass_kg"]),
            "payload_moment_nm": float(final_metrics["payload_moment_nm"]),
            "torso_pitch_compensation_rad": float(torso_pitch),
            "success": success,
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_first_step -----")
        print("payload_first_step_validation_only: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        for key in (
            "phases_completed",
            "right_step_attempted",
            "right_step_success",
            "right_step_failure_reason",
            "left_step_attempted",
            "left_step_success",
            "left_step_skipped_reason",
            "right_step_length_m",
            "left_step_length_m",
            "max_right_foot_clearance_m",
            "max_left_foot_clearance_m",
            "max_swing_foot_lateral_drift_m",
            "right_foot_landed",
            "left_foot_landed",
            "com_inside_support_all_phases",
            "min_com_margin_m",
            "min_com_margin_single_support_m",
            "min_com_margin_double_support_after_landing_m",
            "min_support_foot_force_n",
            "max_pelvis_lateral_shift_m",
            "payload_mass_kg",
            "payload_moment_nm",
            "torso_pitch_compensation_rad",
            "no_continuous_walking_attempted",
            "no_rl_used",
            "success",
        ):
            val = out[key]
            if isinstance(val, float):
                print(f"{key}: {val:.5f}")
            else:
                print(f"{key}: {val}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 payload-aware quasi-static first step.")
    ap.add_argument("--headless", action="store_true", help="Run without viewer.")
    ap.add_argument("--timeout", type=float, default=18.0, help="Simulation timeout in seconds.")
    args = ap.parse_args()
    run_g1_payload_first_step(headless=args.headless, timeout=args.timeout, verbose=True)


if __name__ == "__main__":
    main()
