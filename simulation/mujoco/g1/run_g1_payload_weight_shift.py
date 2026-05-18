#!/usr/bin/env python3
"""
Payload-aware lateral weight-shift validation for G1 holding the Dex3 box.

This milestone builds on ``run_g1_payload_balance.py``. The box is still a
payload validation object, not a passive physical grasp validation. Both feet
remain in the standing support phase; no stepping or locomotion is attempted.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_weight_shift.py --headless --timeout 10
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

SHIFT_ROLL_RAD = 0.045
ANKLE_COUNTER_ROLL_RAD = 0.030

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 1.0,
    "LOAD_HOLD": 1.0,
    "TORSO_COMPENSATE": 1.0,
    "SHIFT_LEFT": 1.0,
    "HOLD_LEFT": 1.0,
    "SHIFT_RIGHT": 1.0,
    "HOLD_RIGHT": 1.0,
    "RETURN_CENTER": 1.0,
}


class WeightShiftPhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    TORSO_COMPENSATE = auto()
    SHIFT_LEFT = auto()
    HOLD_LEFT = auto()
    SHIFT_RIGHT = auto()
    HOLD_RIGHT = auto()
    RETURN_CENTER = auto()
    DONE = auto()


def _phase_at_time(t: float) -> WeightShiftPhase:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        elapsed += float(dur)
        if t < elapsed:
            return WeightShiftPhase[name]
    return WeightShiftPhase.DONE


def _smoothstep01(x: float) -> float:
    u = float(np.clip(x, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _phase_progress(t: float, phase: WeightShiftPhase) -> float:
    elapsed = 0.0
    for name, dur in PHASE_DURATIONS_S.items():
        next_elapsed = elapsed + float(dur)
        if WeightShiftPhase[name] == phase:
            return (float(t) - elapsed) / max(float(dur), 1e-9)
        elapsed = next_elapsed
    return 1.0


def _with_lateral_shift(pose: dict[str, float], shift_u: float) -> dict[str, float]:
    """Apply small hip/ankle roll offsets; positive ``shift_u`` biases left."""
    out = dict(pose)
    s = -float(np.clip(shift_u, -1.0, 1.0))
    out["left_hip_roll_joint"] = float(out["left_hip_roll_joint"] + SHIFT_ROLL_RAD * s)
    out["right_hip_roll_joint"] = float(out["right_hip_roll_joint"] + SHIFT_ROLL_RAD * s)
    out["left_ankle_roll_joint"] = float(
        out["left_ankle_roll_joint"] - ANKLE_COUNTER_ROLL_RAD * s
    )
    out["right_ankle_roll_joint"] = float(
        out["right_ankle_roll_joint"] - ANKLE_COUNTER_ROLL_RAD * s
    )
    return out


def _shift_for_phase(t: float, phase: WeightShiftPhase) -> float:
    if phase in (WeightShiftPhase.STAND_NEUTRAL, WeightShiftPhase.LOAD_HOLD):
        return 0.0
    if phase == WeightShiftPhase.TORSO_COMPENSATE:
        return 0.0
    if phase == WeightShiftPhase.SHIFT_LEFT:
        return _smoothstep01(_phase_progress(t, phase))
    if phase == WeightShiftPhase.HOLD_LEFT:
        return 1.0
    if phase == WeightShiftPhase.SHIFT_RIGHT:
        return 1.0 - 2.0 * _smoothstep01(_phase_progress(t, phase))
    if phase == WeightShiftPhase.HOLD_RIGHT:
        return -1.0
    if phase == WeightShiftPhase.RETURN_CENTER:
        return -1.0 + _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _foot_contact_ok(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["left_foot_force_n"]) > 20.0
        and float(metrics["right_foot_force_n"]) > 20.0
    )


def run_g1_payload_weight_shift(
    *,
    headless: bool = False,
    timeout: float = 10.0,
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
    final_phase = WeightShiftPhase.STAND_NEUTRAL
    last_sampled_phase: WeightShiftPhase | None = None
    max_left_ratio = 0.0
    max_right_ratio = 0.0
    max_hip_roll = 0.0
    max_ankle_roll = 0.0
    t0_wall = time.time()

    while data.time < float(timeout):
        phase = _phase_at_time(float(data.time))
        final_phase = phase
        if phase == WeightShiftPhase.STAND_NEUTRAL:
            pose = dict(neutral)
            current_torso_pitch = 0.0
        elif phase == WeightShiftPhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            current_torso_pitch = 0.0
        else:
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
            pose = _with_lateral_shift(pose, _shift_for_phase(float(data.time), phase))
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
            WeightShiftPhase.HOLD_LEFT,
            WeightShiftPhase.HOLD_RIGHT,
            WeightShiftPhase.RETURN_CENTER,
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
            left_ratio = float(metrics["foot_force_balance_ratio"])
            max_left_ratio = max(max_left_ratio, left_ratio)
            max_right_ratio = max(max_right_ratio, 1.0 - left_ratio)
            last_sampled_phase = phase

        if phase == WeightShiftPhase.DONE:
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

    active_phase_names = [
        name for name in phase_metrics if name not in {"STAND_NEUTRAL", "DONE"}
    ]
    active_metrics = [phase_metrics[name] for name in active_phase_names] or list(
        phase_metrics.values()
    )
    margins = [float(m["min_com_margin_m"]) for m in active_metrics]
    inside_all = all(bool(m["com_inside_support"]) for m in phase_metrics.values())
    contact_all = all(_foot_contact_ok(m) for m in phase_metrics.values())
    min_left_force = min(float(m["left_foot_force_n"]) for m in phase_metrics.values())
    min_right_force = min(float(m["right_foot_force_n"]) for m in phase_metrics.values())
    center = phase_metrics.get("TORSO_COMPENSATE") or phase_metrics.get("LOAD_HOLD") or next(
        iter(phase_metrics.values())
    )
    left = phase_metrics.get("HOLD_LEFT") or phase_metrics.get("SHIFT_LEFT") or center
    right = phase_metrics.get("HOLD_RIGHT") or phase_metrics.get("SHIFT_RIGHT") or center
    final_metrics = phase_metrics.get("RETURN_CENTER") or phase_metrics.get("HOLD_RIGHT") or center

    out = dict(final_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in WeightShiftPhase],
            "final_phase": final_phase.name,
            "phases_completed": list(phase_metrics.keys()),
            "phase_metrics": phase_metrics,
            "payload_weight_shift_validation_only": True,
            "physical_grasp_validation": False,
            "no_locomotion_attempted": True,
            "com_inside_support_all_phases": bool(inside_all),
            "all_feet_kept_contact": bool(contact_all),
            "success": bool(
                inside_all
                and min(margins) > 0.02
                and min_left_force > 20.0
                and min_right_force > 20.0
            ),
            "min_com_margin_m": float(min(margins)),
            "center_com_margin_m": float(center["min_com_margin_m"]),
            "min_com_margin_left_shift_m": float(left["min_com_margin_m"]),
            "min_com_margin_right_shift_m": float(right["min_com_margin_m"]),
            "left_foot_force_n": float(final_metrics["left_foot_force_n"]),
            "right_foot_force_n": float(final_metrics["right_foot_force_n"]),
            "min_left_foot_force_n": float(min_left_force),
            "min_right_foot_force_n": float(min_right_force),
            "max_left_foot_force_ratio": float(max_left_ratio),
            "max_right_foot_force_ratio": float(max_right_ratio),
            "max_hip_roll_rad": float(max_hip_roll),
            "max_ankle_roll_rad": float(max_ankle_roll),
            "torso_pitch_compensation_rad": float(torso_pitch),
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_weight_shift -----")
        print("payload_weight_shift_validation_only: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        for key in (
            "phases_completed",
            "combined_robot_box_com",
            "support_polygon_xy",
            "com_inside_support_all_phases",
            "min_com_margin_m",
            "center_com_margin_m",
            "min_com_margin_left_shift_m",
            "min_com_margin_right_shift_m",
            "left_foot_force_n",
            "right_foot_force_n",
            "min_left_foot_force_n",
            "min_right_foot_force_n",
            "max_left_foot_force_ratio",
            "max_right_foot_force_ratio",
            "max_hip_roll_rad",
            "max_ankle_roll_rad",
            "payload_mass_kg",
            "payload_moment_nm",
            "torso_pitch_compensation_rad",
            "all_feet_kept_contact",
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
    ap = argparse.ArgumentParser(description="G1 payload-aware lateral weight shifting.")
    ap.add_argument("--headless", action="store_true", help="Run without viewer.")
    ap.add_argument("--timeout", type=float, default=10.0, help="Simulation timeout in seconds.")
    args = ap.parse_args()
    run_g1_payload_weight_shift(headless=args.headless, timeout=args.timeout, verbose=True)


if __name__ == "__main__":
    main()
