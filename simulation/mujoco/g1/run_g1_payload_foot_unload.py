#!/usr/bin/env python3
"""
Payload-aware single-foot unload validation for G1 holding the Dex3 box.

This is a pre-step milestone only. The box is treated as payload, both feet
stay on the ground, and the controller shifts load far enough to reduce one
foot force while keeping positive contact. It does not lift a foot, step, walk,
or tune finger manipulation.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_foot_unload.py --headless --timeout 12
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
    _hinge_names,
    _set_qpos_targets,
)
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

HIP_UNLOAD_ROLL_RAD = 0.095
ANKLE_UNLOAD_COUNTER_ROLL_RAD = 0.070
UNLOAD_RATIO_MIN = 0.20
UNLOAD_RATIO_MAX = 0.35
MIN_UNLOADED_FOOT_FORCE_N = 20.0
MIN_COM_MARGIN_M = 0.015

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 1.0,
    "LOAD_HOLD": 1.0,
    "TORSO_COMPENSATE": 1.0,
    "SHIFT_TO_LEFT_SUPPORT": 1.0,
    "UNLOAD_RIGHT_FOOT": 1.0,
    "HOLD_RIGHT_UNLOADED": 1.0,
    "RETURN_CENTER": 1.0,
    "SHIFT_TO_RIGHT_SUPPORT": 1.0,
    "UNLOAD_LEFT_FOOT": 1.0,
    "HOLD_LEFT_UNLOADED": 1.0,
    "RETURN_CENTER_FINAL": 1.0,
}


class FootUnloadPhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    TORSO_COMPENSATE = auto()
    SHIFT_TO_LEFT_SUPPORT = auto()
    UNLOAD_RIGHT_FOOT = auto()
    HOLD_RIGHT_UNLOADED = auto()
    RETURN_CENTER = auto()
    SHIFT_TO_RIGHT_SUPPORT = auto()
    UNLOAD_LEFT_FOOT = auto()
    HOLD_LEFT_UNLOADED = auto()
    RETURN_CENTER_FINAL = auto()
    DONE = auto()


def _phase_at_time(t: float) -> FootUnloadPhase:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        elapsed += float(dur)
        if t < elapsed:
            return FootUnloadPhase[name]
    return FootUnloadPhase.DONE


def _phase_progress(t: float, phase: FootUnloadPhase) -> float:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        next_elapsed = elapsed + float(dur)
        if FootUnloadPhase[name] == phase:
            return (float(t) - elapsed) / max(float(dur), 1e-9)
        elapsed = next_elapsed
    return 1.0


def _support_shift_for_phase(t: float, phase: FootUnloadPhase) -> float:
    """Positive values bias toward left support, unloading the right foot."""
    if phase in (
        FootUnloadPhase.STAND_NEUTRAL,
        FootUnloadPhase.LOAD_HOLD,
        FootUnloadPhase.TORSO_COMPENSATE,
    ):
        return 0.0
    if phase == FootUnloadPhase.SHIFT_TO_LEFT_SUPPORT:
        return _smoothstep01(_phase_progress(t, phase))
    if phase in (FootUnloadPhase.UNLOAD_RIGHT_FOOT, FootUnloadPhase.HOLD_RIGHT_UNLOADED):
        return 1.0
    if phase == FootUnloadPhase.RETURN_CENTER:
        return 1.0 - _smoothstep01(_phase_progress(t, phase))
    if phase == FootUnloadPhase.SHIFT_TO_RIGHT_SUPPORT:
        return -_smoothstep01(_phase_progress(t, phase))
    if phase in (FootUnloadPhase.UNLOAD_LEFT_FOOT, FootUnloadPhase.HOLD_LEFT_UNLOADED):
        return -1.0
    if phase == FootUnloadPhase.RETURN_CENTER_FINAL:
        return -1.0 + _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _with_unload_shift(pose: dict[str, float], shift_u: float) -> dict[str, float]:
    out = dict(pose)
    # Same sign convention as the weight-shift milestone: positive shift_u
    # increases left support and unloads the right foot.
    s = -float(np.clip(shift_u, -1.0, 1.0))
    out["left_hip_roll_joint"] = float(out["left_hip_roll_joint"] + HIP_UNLOAD_ROLL_RAD * s)
    out["right_hip_roll_joint"] = float(out["right_hip_roll_joint"] + HIP_UNLOAD_ROLL_RAD * s)
    out["left_ankle_roll_joint"] = float(
        out["left_ankle_roll_joint"] - ANKLE_UNLOAD_COUNTER_ROLL_RAD * s
    )
    out["right_ankle_roll_joint"] = float(
        out["right_ankle_roll_joint"] - ANKLE_UNLOAD_COUNTER_ROLL_RAD * s
    )
    return out


def _left_ratio(metrics: dict[str, Any]) -> float:
    return float(metrics["foot_force_balance_ratio"])


def _right_ratio(metrics: dict[str, Any]) -> float:
    return 1.0 - _left_ratio(metrics)


def _ratio_in_unload_band(ratio: float) -> bool:
    return UNLOAD_RATIO_MIN <= float(ratio) <= UNLOAD_RATIO_MAX


def run_g1_payload_foot_unload(
    *,
    headless: bool = False,
    timeout: float = 12.0,
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
    final_phase = FootUnloadPhase.STAND_NEUTRAL
    last_sampled_phase: FootUnloadPhase | None = None
    min_left_force = float("inf")
    min_right_force = float("inf")
    min_left_ratio = float("inf")
    min_right_ratio = float("inf")
    max_support_ratio = 0.0
    max_hip_roll = 0.0
    max_ankle_roll = 0.0
    right_unload_success = False
    left_unload_success = False
    t0_wall = time.time()

    while data.time < float(timeout):
        phase = _phase_at_time(float(data.time))
        final_phase = phase
        if phase == FootUnloadPhase.STAND_NEUTRAL:
            pose = dict(neutral)
            current_torso_pitch = 0.0
        elif phase == FootUnloadPhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            current_torso_pitch = 0.0
        else:
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
            pose = _with_unload_shift(pose, _support_shift_for_phase(float(data.time), phase))
            current_torso_pitch = torso_pitch

        _set_qpos_targets(model, data, pose, hinge_addrs)
        _command_targets(model, data, pose, actuator_ids)
        mujoco.mj_step(model, data)
        _set_qpos_targets(model, data, pose, hinge_addrs)
        data.qvel[:] = 0.0
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)

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

        sample_now = phase != last_sampled_phase or phase in (
            FootUnloadPhase.UNLOAD_RIGHT_FOOT,
            FootUnloadPhase.HOLD_RIGHT_UNLOADED,
            FootUnloadPhase.UNLOAD_LEFT_FOOT,
            FootUnloadPhase.HOLD_LEFT_UNLOADED,
            FootUnloadPhase.RETURN_CENTER,
            FootUnloadPhase.RETURN_CENTER_FINAL,
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
            left_ratio = _left_ratio(metrics)
            right_ratio = _right_ratio(metrics)
            left_force = float(metrics["left_foot_force_n"])
            right_force = float(metrics["right_foot_force_n"])
            min_left_force = min(min_left_force, left_force)
            min_right_force = min(min_right_force, right_force)
            min_left_ratio = min(min_left_ratio, left_ratio)
            min_right_ratio = min(min_right_ratio, right_ratio)
            max_support_ratio = max(max_support_ratio, left_ratio, right_ratio)
            if phase in (
                FootUnloadPhase.UNLOAD_RIGHT_FOOT,
                FootUnloadPhase.HOLD_RIGHT_UNLOADED,
            ):
                right_unload_success = right_unload_success or (
                    _ratio_in_unload_band(right_ratio)
                    and right_force > MIN_UNLOADED_FOOT_FORCE_N
                )
            if phase in (
                FootUnloadPhase.UNLOAD_LEFT_FOOT,
                FootUnloadPhase.HOLD_LEFT_UNLOADED,
            ):
                left_unload_success = left_unload_success or (
                    _ratio_in_unload_band(left_ratio)
                    and left_force > MIN_UNLOADED_FOOT_FORCE_N
                )
            last_sampled_phase = phase

        if phase == FootUnloadPhase.DONE:
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

    active_metrics = [
        m for name, m in phase_metrics.items() if name not in {"STAND_NEUTRAL", "DONE"}
    ] or list(phase_metrics.values())
    margins = [float(m["min_com_margin_m"]) for m in active_metrics]
    inside_all = all(bool(m["com_inside_support"]) for m in phase_metrics.values())
    final_metrics = (
        phase_metrics.get("RETURN_CENTER_FINAL")
        or phase_metrics.get("HOLD_LEFT_UNLOADED")
        or next(reversed(phase_metrics.values()))
    )
    success = bool(
        inside_all
        and min(margins) > MIN_COM_MARGIN_M
        and right_unload_success
        and left_unload_success
        and min_right_force > MIN_UNLOADED_FOOT_FORCE_N
        and min_left_force > MIN_UNLOADED_FOOT_FORCE_N
    )

    out = dict(final_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in FootUnloadPhase],
            "final_phase": final_phase.name,
            "phases_completed": list(phase_metrics.keys()),
            "phase_metrics": phase_metrics,
            "payload_foot_unload_validation_only": True,
            "physical_grasp_validation": False,
            "no_locomotion_attempted": True,
            "com_inside_support_all_phases": bool(inside_all),
            "min_com_margin_m": float(min(margins)),
            "min_right_foot_force_n": float(min_right_force),
            "min_left_foot_force_n": float(min_left_force),
            "min_right_foot_force_ratio": float(min_right_ratio),
            "min_left_foot_force_ratio": float(min_left_ratio),
            "right_foot_unload_success": bool(right_unload_success),
            "left_foot_unload_success": bool(left_unload_success),
            "max_support_foot_force_ratio": float(max_support_ratio),
            "max_hip_roll_rad": float(max_hip_roll),
            "max_ankle_roll_rad": float(max_ankle_roll),
            "torso_pitch_compensation_rad": float(torso_pitch),
            "success": success,
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_foot_unload -----")
        print("payload_foot_unload_validation_only: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        for key in (
            "phases_completed",
            "com_inside_support_all_phases",
            "min_com_margin_m",
            "min_right_foot_force_n",
            "min_left_foot_force_n",
            "min_right_foot_force_ratio",
            "min_left_foot_force_ratio",
            "right_foot_unload_success",
            "left_foot_unload_success",
            "max_support_foot_force_ratio",
            "max_hip_roll_rad",
            "max_ankle_roll_rad",
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
    ap = argparse.ArgumentParser(description="G1 payload-aware single-foot unload.")
    ap.add_argument("--headless", action="store_true", help="Run without viewer.")
    ap.add_argument("--timeout", type=float, default=12.0, help="Simulation timeout in seconds.")
    args = ap.parse_args()
    run_g1_payload_foot_unload(headless=args.headless, timeout=args.timeout, verbose=True)


if __name__ == "__main__":
    main()
