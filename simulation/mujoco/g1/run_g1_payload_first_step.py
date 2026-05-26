#!/usr/bin/env python3
"""
Payload-aware quasi-static first forward step for G1 holding the Dex3 box.

Builds directly on ``run_g1_payload_foot_clearance``: shift to left support, unload and
lightly lift the right foot, swing it forward by a small amount, lower it ahead of its
starting pose, stabilize, then unwind lateral weight bias. Full walking gait, dynamics,
RL, ROS, and finger grasp tuning are out of scope.

Run:

    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_first_step.py
    ./.venv/bin/python simulation/mujoco/g1/run_g1_payload_first_step.py --headless --timeout 16
"""

from __future__ import annotations

import argparse
import importlib
import os
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
    _support_polygon_aabb,
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

# Non-backtracking swing overlays (negative hip pitch = forward; no positive hip lift).
NONBACKTRACKING_LIFT_DELTA = {
    "hip_pitch_joint": -0.025,
    "knee_joint": 0.050,
    "ankle_pitch_joint": -0.025,
}
NONBACKTRACKING_LIFT_CLEARANCE_AUX = {
    "hip_pitch_joint": -0.032,
    "knee_joint": 0.008,
    "ankle_pitch_joint": -0.210,
}
FORWARD_SWING_DELTA = {
    "hip_pitch_joint": -0.045,
    "knee_joint": 0.020,
    "ankle_pitch_joint": 0.055,
}
PLACE_DELTA = {
    "hip_pitch_joint": -0.020,
    "knee_joint": 0.000,
    "ankle_pitch_joint": 0.035,
}
EARLY_FORWARD_LIFT_FRACTION = 0.25
MIN_SWING_KNEE_FLEXION_RAD = 0.080
MIN_SWING_FOOT_X_DELTA_M = -0.003
MAX_SWING_FOOT_BACKWARD_DRIFT_M = 0.003

MIN_RIGHT_FOOT_FORWARD_DISPLACEMENT_M = 0.025
MIN_RIGHT_FOOT_CLEARANCE_PASS_M = 0.010
FLOOR_Z_M = 0.0
SWING_GROUND_CLEARANCE_LIFT_U_MIN = 0.25
MIN_MIDSWING_GROUND_CLEARANCE_M = 0.006
MIN_PEAK_GROUND_CLEARANCE_M = 0.010
SCRAPE_GROUND_CLEARANCE_M = 0.002
GROUND_Z_TOL_M = 0.005
MIN_SUPPORT_FOOT_FORCE_N = 120.0
MIN_POST_LAND_EACH_FOOT_N = 95.0
MIN_LIFTED_COM_MARGIN_M = 0.005
MIN_DOUBLE_SUPPORT_AFTER_STEP_M = 0.015
MAX_SWING_XY_DRIFT_M = 0.015

# Support-dominant lateral transfer (replaces symmetric unload on both legs).
SUPPORT_HIP_ROLL_RAD = 0.055
SUPPORT_ANKLE_ROLL_RAD = 0.045
WAIST_ROLL_JOINT = "waist_roll_joint"
WAIST_ROLL_PER_SHIFT_RAD = 0.012
SWING_ROLL_UNLOAD_FRAC = 0.0

# Conservative symmetric bent-knee walking-ready posture (both legs).
WALK_READY_CROUCH_PITCH = {
    "hip_pitch_joint": -0.055,
    "knee_joint": 0.110,
    "ankle_pitch_joint": -0.055,
}

PHASE_DURATIONS_S = {
    "STAND_NEUTRAL": 0.8,
    "LOAD_HOLD": 0.8,
    "WALK_READY_CROUCH": 0.8,
    "TORSO_COMPENSATE": 0.8,
    "SHIFT_TO_LEFT_SUPPORT": 0.8,
    "UNLOAD_RIGHT_FOOT": 0.8,
    "LIFT_RIGHT_FOOT_SMALL": 0.8,
    "SWING_RIGHT_FOOT_FORWARD": 0.85,
    "PLACE_RIGHT_FOOT_FORWARD": 0.85,
    "STABILIZE_AFTER_RIGHT_STEP": 0.9,
    "RETURN_CENTER_OR_HOLD_STEPPED": 0.9,
}


class FirstStepPhase(Enum):
    STAND_NEUTRAL = auto()
    LOAD_HOLD = auto()
    WALK_READY_CROUCH = auto()
    TORSO_COMPENSATE = auto()
    SHIFT_TO_LEFT_SUPPORT = auto()
    UNLOAD_RIGHT_FOOT = auto()
    LIFT_RIGHT_FOOT_SMALL = auto()
    SWING_RIGHT_FOOT_FORWARD = auto()
    PLACE_RIGHT_FOOT_FORWARD = auto()
    STABILIZE_AFTER_RIGHT_STEP = auto()
    RETURN_CENTER_OR_HOLD_STEPPED = auto()
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


def _support_side_for_shift_u(shift_u: float) -> str:
    """Positive ``shift_u`` loads the left leg; negative loads the right."""
    return "left" if float(shift_u) >= 0.0 else "right"


def _swing_side_for_support(support_side: str) -> str:
    return "right" if support_side == "left" else "left"


def _make_support_dominant_shift_pose(
    base_pose: dict[str, float],
    *,
    neutral_ref: dict[str, float],
    support_side: str,
    shift_u: float,
) -> tuple[dict[str, float], float, float]:
    """
    Bias roll onto the support leg only; keep the swing leg near neutral during transfer.

    Sign convention matches the prior passing unload milestone (``s = -shift_u`` on support).
    """
    out = dict(base_pose)
    s = -float(np.clip(shift_u, -1.0, 1.0))
    if abs(s) < 1e-9:
        swing_side = _swing_side_for_support(support_side)
        return out, 0.0, 0.0

    swing_side = _swing_side_for_support(support_side)
    sup_hip = f"{support_side}_hip_roll_joint"
    sup_ankle = f"{support_side}_ankle_roll_joint"
    sw_hip = f"{swing_side}_hip_roll_joint"

    out[sup_hip] = float(out[sup_hip] + SUPPORT_HIP_ROLL_RAD * s)
    out[sup_ankle] = float(out[sup_ankle] - SUPPORT_ANKLE_ROLL_RAD * s)

    if SWING_ROLL_UNLOAD_FRAC > 0.0:
        out[sw_hip] = float(out[sw_hip] + SUPPORT_HIP_ROLL_RAD * SWING_ROLL_UNLOAD_FRAC * s)

    if WAIST_ROLL_JOINT in out:
        out[WAIST_ROLL_JOINT] = float(out[WAIST_ROLL_JOINT] + WAIST_ROLL_PER_SHIFT_RAD * s)

    sup_delta = abs(float(out[sup_hip] - float(neutral_ref[sup_hip])))
    sw_delta = abs(float(out[sw_hip] - float(neutral_ref[sw_hip])))
    return out, sup_delta, sw_delta


def _crouch_amount_for_phase(t: float, phase: FirstStepPhase) -> float:
    if phase in (FirstStepPhase.STAND_NEUTRAL, FirstStepPhase.LOAD_HOLD, FirstStepPhase.DONE):
        return 0.0
    if phase == FirstStepPhase.WALK_READY_CROUCH:
        return _smoothstep01(_phase_progress(t, phase))
    return 1.0


def _apply_walking_ready_crouch(
    pose: dict[str, float],
    crouch_u: float,
    *,
    neutral_ref: dict[str, float],
) -> tuple[dict[str, float], float, float, float]:
    """Apply symmetric slight knee bend; returns max knee/hip/ankle pitch deltas vs neutral."""
    out = dict(pose)
    u = float(np.clip(crouch_u, 0.0, 1.0))
    max_knee = 0.0
    max_hip = 0.0
    max_ankle = 0.0
    for side in ("left", "right"):
        for suffix, delta in WALK_READY_CROUCH_PITCH.items():
            jn = f"{side}_{suffix}"
            if jn not in out:
                continue
            out[jn] = float(out[jn] + u * float(delta))
            ref = float(neutral_ref.get(jn, 0.0))
            d = abs(float(out[jn] - ref))
            if suffix == "knee_joint":
                max_knee = max(max_knee, d)
            elif suffix == "hip_pitch_joint":
                max_hip = max(max_hip, d)
            elif suffix == "ankle_pitch_joint":
                max_ankle = max(max_ankle, d)
    return out, max_knee, max_hip, max_ankle


def _shutdown_passive_viewer(viewer: Any) -> None:
    """Clear custom geoms and close the passive viewer before GLFW teardown."""
    user_scn = getattr(viewer, "user_scn", None)
    if user_scn is not None:
        user_scn.ngeom = 0
    close_fn = getattr(viewer, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def _support_shift_for_phase(t: float, phase: FirstStepPhase) -> float:
    """Positive bias loads the left foot and unloads the right (unload/clearance convention)."""
    name = phase.name
    if phase == FirstStepPhase.SHIFT_TO_LEFT_SUPPORT:
        return _smoothstep01(_phase_progress(t, phase))
    if name in {
        "UNLOAD_RIGHT_FOOT",
        "LIFT_RIGHT_FOOT_SMALL",
        "SWING_RIGHT_FOOT_FORWARD",
        "PLACE_RIGHT_FOOT_FORWARD",
        "STABILIZE_AFTER_RIGHT_STEP",
    }:
        return 1.0
    if phase == FirstStepPhase.RETURN_CENTER_OR_HOLD_STEPPED:
        return 1.0 - _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _right_lift_amount(t: float, phase: FirstStepPhase) -> float:
    if phase == FirstStepPhase.LIFT_RIGHT_FOOT_SMALL:
        return _smoothstep01(_phase_progress(t, phase))
    if phase == FirstStepPhase.SWING_RIGHT_FOOT_FORWARD:
        return 1.0
    if phase == FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD:
        return 1.0 - _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _right_forward_swing_amount(t: float, phase: FirstStepPhase) -> float:
    if phase == FirstStepPhase.SWING_RIGHT_FOOT_FORWARD:
        return _smoothstep01(_phase_progress(t, phase))
    if phase.value >= FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD.value:
        return 1.0
    return 0.0


def _right_forward_place_amount(t: float, phase: FirstStepPhase) -> float:
    if phase == FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD:
        return _smoothstep01(_phase_progress(t, phase))
    return 0.0


def _knee_flexion_rad(
    pose: dict[str, float],
    side: str,
    *,
    neutral_ref: dict[str, float],
) -> float:
    jn = f"{side}_knee_joint"
    return float(pose[jn] - float(neutral_ref[jn]))


def _apply_crouch_preserving_swing_overlay(
    pose: dict[str, float],
    side: str,
    lift_u: float,
    forward_swing_u: float,
    forward_place_u: float = 0.0,
    *,
    neutral_ref: dict[str, float] | None = None,
) -> dict[str, float]:
    """Lift and swing forward on crouch without straightening the knee or pulling foot backward."""
    lift_u = float(np.clip(lift_u, 0.0, 1.0))
    out = _apply_leg_pitch_overlay(pose, side, NONBACKTRACKING_LIFT_DELTA, lift_u)
    if lift_u > 1e-6:
        out = _apply_leg_pitch_overlay(out, side, NONBACKTRACKING_LIFT_CLEARANCE_AUX, lift_u)
    forward_u = min(1.0, EARLY_FORWARD_LIFT_FRACTION * lift_u + float(forward_swing_u))
    out = _apply_leg_pitch_overlay(out, side, FORWARD_SWING_DELTA, forward_u)
    if forward_place_u > 1e-6:
        out = _apply_leg_pitch_overlay(out, side, PLACE_DELTA, forward_place_u)
    if neutral_ref is not None:
        knee_jn = f"{side}_knee_joint"
        floor = float(neutral_ref[knee_jn]) + MIN_SWING_KNEE_FLEXION_RAD
        if knee_jn in out and float(out[knee_jn]) < floor:
            out[knee_jn] = floor
    return out


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
        jn = prefix + suffix
        if jn in out:
            out[jn] = float(out[jn] + u * float(delta))
    return out


def _update_first_step_debug_scene(
    user_scn: mujoco.MjvScene,
    *,
    right_init_xyz: np.ndarray | None,
    right_final_xyz: np.ndarray | None,
    com_xyz: np.ndarray | None,
    support_corners_xy: np.ndarray | None,
    swing_foot_lowest_xyz: np.ndarray | None = None,
    floor_z: float = 0.002,
) -> None:
    """Optional viewer overlays: foot start/end, COM projection, support polygon."""
    user_scn.ngeom = 0
    idx = 0
    identity = np.eye(3, dtype=np.float64).reshape(-1)

    def add_sphere(pos: np.ndarray, rgba: tuple[float, float, float, float], radius: float) -> None:
        nonlocal idx
        if idx >= user_scn.maxgeom:
            return
        geom = user_scn.geoms[idx]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([radius, radius, radius], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            identity,
            np.asarray(rgba, dtype=np.float32),
        )
        idx += 1

    if right_init_xyz is not None:
        add_sphere(np.asarray(right_init_xyz, dtype=float), (0.15, 0.85, 0.25, 0.9), 0.018)
    if right_final_xyz is not None:
        add_sphere(np.asarray(right_final_xyz, dtype=float), (0.25, 0.45, 1.0, 0.9), 0.018)
    if com_xyz is not None:
        com_proj = np.array([float(com_xyz[0]), float(com_xyz[1]), floor_z], dtype=float)
        add_sphere(com_proj, (1.0, 0.25, 0.2, 0.95), 0.022)

    if support_corners_xy is not None and len(support_corners_xy) >= 4:
        corners = np.asarray(support_corners_xy, dtype=float)
        for c in corners[:4]:
            corner_xyz = np.array([float(c[0]), float(c[1]), floor_z], dtype=float)
            add_sphere(corner_xyz, (0.95, 0.85, 0.15, 0.85), 0.010)
    if swing_foot_lowest_xyz is not None:
        add_sphere(np.asarray(swing_foot_lowest_xyz, dtype=float), (0.2, 0.9, 0.95, 0.95), 0.012)

    user_scn.ngeom = idx


def _foot_geom_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> float:
    """World Z of the lowest point on a foot geom (sphere center minus radius)."""
    z_center = float(data.geom_xpos[gid, 2])
    if int(model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z_center - float(model.geom_size[gid, 0])
    sz0 = float(model.geom_size[gid, 0]) if model.geom_size.shape[1] > 0 else 0.0
    return z_center - sz0


def _get_foot_bottom_clearance_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    floor_z: float = FLOOR_Z_M,
) -> tuple[float, np.ndarray]:
    """Min clearance of swing-foot geoms above ``floor_z`` and the lowest point XYZ."""
    min_bottom_z = float("inf")
    lowest_xyz = np.zeros(3, dtype=float)
    for gid in _foot_geom_ids(model, side):
        bottom_z = _foot_geom_bottom_z(model, data, gid)
        if bottom_z < min_bottom_z:
            min_bottom_z = bottom_z
            lowest_xyz = np.array(
                [float(data.geom_xpos[gid, 0]), float(data.geom_xpos[gid, 1]), bottom_z],
                dtype=float,
            )
    if not np.isfinite(min_bottom_z):
        return 0.0, lowest_xyz
    return float(min_bottom_z - floor_z), lowest_xyz


def _swing_geom_clearance_sample_active(
    phase: Any,
    lift_u: float,
    *,
    lift_phase: Any,
    swing_phase: Any,
    place_phase: Any,
) -> bool:
    """Sample geom clearance during lift/swing; skip unload and late place touchdown."""
    if phase == swing_phase:
        return True
    if phase == lift_phase and lift_u > SWING_GROUND_CLEARANCE_LIFT_U_MIN:
        return True
    if phase == place_phase and lift_u > SWING_GROUND_CLEARANCE_LIFT_U_MIN:
        return True
    return False


def _foot_center_xyz(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    pts = [
        np.asarray(data.geom_xpos[gid, :3], dtype=float)
        for gid in _foot_geom_ids(model, side)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not pts:
        return np.zeros(3, dtype=float)
    return np.mean(np.asarray(pts, dtype=float), axis=0)


def run_g1_payload_first_step(
    *,
    headless: bool = False,
    timeout: float = 16.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    viewer_speed: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
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

    right_swing_xy0: np.ndarray | None = None
    right_ground_z_ref: float | None = None
    forward_baseline_x: float | None = None

    max_right_clearance = 0.0
    max_forward_disp = 0.0
    max_swing_xy_drift = 0.0
    min_support_force = float("inf")
    min_com_margin = float("inf")
    min_single_margin = float("inf")
    min_margin_after_land = float("inf")
    max_pelvis_lateral_shift = 0.0
    max_torso_pitch_seen = abs(float(torso_pitch))
    max_support_leg_roll_delta = 0.0
    max_swing_leg_roll_delta = 0.0
    max_knee_crouch_delta = 0.0
    max_hip_pitch_crouch_delta = 0.0
    max_ankle_pitch_crouch_delta = 0.0
    min_right_swing_knee_flexion = float("inf")
    min_right_swing_foot_x_delta_m = float("inf")
    min_right_swing_foot_ground_clearance_m = float("inf")
    max_right_swing_foot_ground_clearance_m = 0.0
    right_swing_foot_scrape_detected = False

    right_step_attempted = False
    right_returned_to_ground = False
    right_step_success = False
    landed_ok = False
    stabilize_ok = False

    load_com_xy0: np.ndarray | None = None

    right_init_xyz_vis: np.ndarray | None = None
    right_final_xyz_vis: np.ndarray | None = None
    right_swing_lowest_xyz_vis: np.ndarray | None = None

    t0_wall = time.time()

    def _advance_one_step() -> bool:
        nonlocal final_phase, last_sampled_phase, right_swing_xy0, right_ground_z_ref
        nonlocal forward_baseline_x, max_right_clearance, max_forward_disp, max_swing_xy_drift
        nonlocal min_support_force, min_com_margin, min_single_margin, min_margin_after_land
        nonlocal max_pelvis_lateral_shift, max_torso_pitch_seen, right_step_attempted
        nonlocal right_returned_to_ground, landed_ok, stabilize_ok, load_com_xy0
        nonlocal right_init_xyz_vis, right_final_xyz_vis
        nonlocal max_support_leg_roll_delta, max_swing_leg_roll_delta
        nonlocal max_knee_crouch_delta, max_hip_pitch_crouch_delta, max_ankle_pitch_crouch_delta
        nonlocal min_right_swing_knee_flexion, min_right_swing_foot_x_delta_m
        nonlocal min_right_swing_foot_ground_clearance_m, max_right_swing_foot_ground_clearance_m
        nonlocal right_swing_foot_scrape_detected, right_swing_lowest_xyz_vis

        if data.time >= float(timeout):
            return False

        phase = _phase_at_time(float(data.time))
        final_phase = phase
        if phase == FirstStepPhase.DONE:
            metrics = _evaluate_metrics(
                model,
                data,
                torso_pitch_rad=torso_pitch,
                hinge_addrs=hinge_addrs,
                neutral=neutral,
            )
            phase_metrics[phase.name] = metrics
            margin = float(metrics["min_com_margin_m"])
            min_com_margin = min(min_com_margin, margin)
            final_phase = phase
            return False

        if phase == FirstStepPhase.STAND_NEUTRAL:
            pose = dict(neutral)
            current_torso_pitch = 0.0
        elif phase == FirstStepPhase.LOAD_HOLD:
            pose = dict(load_hold_pose)
            current_torso_pitch = 0.0
        elif phase in (FirstStepPhase.WALK_READY_CROUCH, FirstStepPhase.TORSO_COMPENSATE):
            pose = dict(load_hold_pose)
            crouch_u = _crouch_amount_for_phase(float(data.time), phase)
            pose, kd, hd, ad = _apply_walking_ready_crouch(
                pose, crouch_u, neutral_ref=neutral
            )
            max_knee_crouch_delta = max(max_knee_crouch_delta, kd)
            max_hip_pitch_crouch_delta = max(max_hip_pitch_crouch_delta, hd)
            max_ankle_pitch_crouch_delta = max(max_ankle_pitch_crouch_delta, ad)
            if phase == FirstStepPhase.TORSO_COMPENSATE:
                pose[WAIST_PITCH_JOINT] = torso_pitch
                current_torso_pitch = torso_pitch
            else:
                current_torso_pitch = 0.0
        else:
            pose = dict(load_hold_pose)
            pose[WAIST_PITCH_JOINT] = torso_pitch
            pose, kd, hd, ad = _apply_walking_ready_crouch(pose, 1.0, neutral_ref=neutral)
            max_knee_crouch_delta = max(max_knee_crouch_delta, kd)
            max_hip_pitch_crouch_delta = max(max_hip_pitch_crouch_delta, hd)
            max_ankle_pitch_crouch_delta = max(max_ankle_pitch_crouch_delta, ad)
            shift_u = _support_shift_for_phase(float(data.time), phase)
            if abs(shift_u) > 1e-6:
                pose, sup_roll, sw_roll = _make_support_dominant_shift_pose(
                    pose,
                    neutral_ref=neutral,
                    support_side=_support_side_for_shift_u(shift_u),
                    shift_u=shift_u,
                )
                max_support_leg_roll_delta = max(max_support_leg_roll_delta, sup_roll)
                max_swing_leg_roll_delta = max(max_swing_leg_roll_delta, sw_roll)

            lf = _right_lift_amount(float(data.time), phase)
            ff_swing = _right_forward_swing_amount(float(data.time), phase)
            ff_place = _right_forward_place_amount(float(data.time), phase)
            pose = _apply_crouch_preserving_swing_overlay(
                pose, "right", lf, ff_swing, ff_place, neutral_ref=neutral
            )
            if phase in (
                FirstStepPhase.UNLOAD_RIGHT_FOOT,
                FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
                FirstStepPhase.SWING_RIGHT_FOOT_FORWARD,
            ) or (lf > 1e-3 and phase != FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD):
                min_right_swing_knee_flexion = min(
                    min_right_swing_knee_flexion,
                    _knee_flexion_rad(pose, "right", neutral_ref=neutral),
                )

            current_torso_pitch = torso_pitch

        _set_qpos_targets(model, data, pose, hinge_addrs)
        _command_targets(model, data, pose, actuator_ids)
        mujoco.mj_step(model, data)
        _set_qpos_targets(model, data, pose, hinge_addrs)
        data.qvel[:] = 0.0
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)

        right_center = _foot_center_xyz(model, data, "right")

        if phase == FirstStepPhase.UNLOAD_RIGHT_FOOT and right_swing_xy0 is None:
            right_swing_xy0 = np.asarray(right_center[:2], dtype=float).copy()
            right_ground_z_ref = float(right_center[2])
            forward_baseline_x = float(right_center[0])
            right_init_xyz_vis = np.asarray(right_center, dtype=float).copy()

        swing_active_phases = (
            FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
            FirstStepPhase.SWING_RIGHT_FOOT_FORWARD,
            FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD,
        )
        if phase in swing_active_phases:
            right_step_attempted = True

        lf_now = _right_lift_amount(float(data.time), phase)

        if right_swing_xy0 is not None and forward_baseline_x is not None:
            foot_x_delta = float(right_center[0] - forward_baseline_x)
            max_forward_disp = max(max_forward_disp, foot_x_delta)
            if phase in (
                FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
                FirstStepPhase.SWING_RIGHT_FOOT_FORWARD,
            ) or (lf_now > 1e-3 and phase == FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD):
                min_right_swing_foot_x_delta_m = min(
                    min_right_swing_foot_x_delta_m, foot_x_delta
                )
            if right_ground_z_ref is not None:
                clearance = float(right_center[2] - float(right_ground_z_ref))
                max_right_clearance = max(max_right_clearance, clearance)

            # Lateral drift only: forward displacement is tracked separately via
            # ``right_foot_forward_displacement_m`` (baseline X captured at unload).
            if lf_now > 1e-3 or phase in swing_active_phases:
                drift_lateral = abs(float(right_center[1]) - float(right_swing_xy0[1]))
                max_swing_xy_drift = max(max_swing_xy_drift, drift_lateral)

        if _swing_geom_clearance_sample_active(
            phase,
            lf_now,
            lift_phase=FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
            swing_phase=FirstStepPhase.SWING_RIGHT_FOOT_FORWARD,
            place_phase=FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD,
        ):
            ground_clearance, lowest_xyz = _get_foot_bottom_clearance_m(
                model, data, "right", floor_z=FLOOR_Z_M
            )
            min_right_swing_foot_ground_clearance_m = min(
                min_right_swing_foot_ground_clearance_m, ground_clearance
            )
            max_right_swing_foot_ground_clearance_m = max(
                max_right_swing_foot_ground_clearance_m, ground_clearance
            )
            if ground_clearance < SCRAPE_GROUND_CLEARANCE_M:
                right_swing_foot_scrape_detected = True
            right_swing_lowest_xyz_vis = np.asarray(lowest_xyz, dtype=float).copy()

        if right_ground_z_ref is not None and lf_now < 1e-2:
            if phase in (
                FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD,
                FirstStepPhase.STABILIZE_AFTER_RIGHT_STEP,
                FirstStepPhase.RETURN_CENTER_OR_HOLD_STEPPED,
            ):
                grounded = abs(float(right_center[2]) - float(right_ground_z_ref)) <= GROUND_Z_TOL_M
                right_returned_to_ground = right_returned_to_ground or grounded
                landed_ok = landed_ok or grounded

        sample_now = phase != last_sampled_phase or phase in (
            FirstStepPhase.WALK_READY_CROUCH,
            FirstStepPhase.TORSO_COMPENSATE,
            FirstStepPhase.UNLOAD_RIGHT_FOOT,
            FirstStepPhase.LIFT_RIGHT_FOOT_SMALL,
            FirstStepPhase.SWING_RIGHT_FOOT_FORWARD,
            FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD,
            FirstStepPhase.STABILIZE_AFTER_RIGHT_STEP,
            FirstStepPhase.RETURN_CENTER_OR_HOLD_STEPPED,
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
            max_torso_pitch_seen = max(
                max_torso_pitch_seen, float(metrics.get("max_torso_pitch_rad", abs(current_torso_pitch)))
            )

            com_xy_now = np.asarray(metrics["combined_robot_box_com"], dtype=float)[:2]
            if phase == FirstStepPhase.LOAD_HOLD and load_com_xy0 is None:
                load_com_xy0 = com_xy_now.copy()
            elif load_com_xy0 is not None:
                sway = float(np.linalg.norm(com_xy_now - load_com_xy0))
                max_pelvis_lateral_shift = max(max_pelvis_lateral_shift, sway)

            lf_s = _right_lift_amount(float(data.time), phase)
            if phase in swing_active_phases and lf_s > 1e-3:
                min_single_margin = min(min_single_margin, margin)
                min_support_force = min(min_support_force, float(metrics["left_foot_force_n"]))

            if phase == FirstStepPhase.STABILIZE_AFTER_RIGHT_STEP:
                min_margin_after_land = min(min_margin_after_land, margin)
                stabilize_ok = stabilize_ok or bool(
                    metrics["com_inside_support"]
                    and float(metrics["left_foot_force_n"]) > MIN_POST_LAND_EACH_FOOT_N
                    and float(metrics["right_foot_force_n"]) > MIN_POST_LAND_EACH_FOOT_N
                )

            last_sampled_phase = phase

        if phase.value >= FirstStepPhase.PLACE_RIGHT_FOOT_FORWARD.value:
            right_final_xyz_vis = np.asarray(right_center, dtype=float).copy()

        return True

    if headless:
        while _advance_one_step():
            pass
    else:
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            try:
                while viewer.is_running():
                    step_start = time.time()
                    if not _advance_one_step():
                        break
                    if viewer.is_running():
                        if viewer.user_scn is not None:
                            support = _support_polygon_aabb(model, data)
                            com_vis = np.asarray(
                                _evaluate_metrics(
                                    model,
                                    data,
                                    torso_pitch_rad=torso_pitch,
                                    hinge_addrs=hinge_addrs,
                                    neutral=neutral,
                                )["combined_robot_box_com"],
                                dtype=float,
                            )
                            _update_first_step_debug_scene(
                                viewer.user_scn,
                                right_init_xyz=right_init_xyz_vis,
                                right_final_xyz=right_final_xyz_vis,
                                com_xyz=com_vis,
                                support_corners_xy=np.asarray(support["corners"], dtype=float),
                                swing_foot_lowest_xyz=right_swing_lowest_xyz_vis,
                            )
                        viewer.sync()
                    sim_dt = float(model.opt.timestep) / max(float(viewer_speed), 1e-3)
                    time.sleep(max(0.0, sim_dt - (time.time() - step_start)))
            finally:
                _shutdown_passive_viewer(viewer)

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
    if not np.isfinite(min_margin_after_land):
        min_margin_after_land = min_com_margin
    if not np.isfinite(min_support_force):
        min_support_force = min(
            float(m["left_foot_force_n"]) for m in phase_metrics.values()
        )
    if not np.isfinite(min_right_swing_knee_flexion):
        min_right_swing_knee_flexion = float(WALK_READY_CROUCH_PITCH["knee_joint"])
    if not np.isfinite(min_right_swing_foot_x_delta_m):
        min_right_swing_foot_x_delta_m = 0.0
    if not np.isfinite(min_right_swing_foot_ground_clearance_m):
        min_right_swing_foot_ground_clearance_m = 0.0

    right_swing_clearance_success = bool(
        not right_swing_foot_scrape_detected
        and min_right_swing_foot_ground_clearance_m >= MIN_MIDSWING_GROUND_CLEARANCE_M
        and max_right_swing_foot_ground_clearance_m >= MIN_PEAK_GROUND_CLEARANCE_M
    )

    right_swing_foot_backward_drift_m = float(
        max(0.0, -min_right_swing_foot_x_delta_m)
    )
    swing_foot_nonbacktracking = bool(
        min_right_swing_foot_x_delta_m >= MIN_SWING_FOOT_X_DELTA_M
        and right_swing_foot_backward_drift_m <= MAX_SWING_FOOT_BACKWARD_DRIFT_M
    )

    inside_all = all(bool(m["com_inside_support"]) for m in phase_metrics.values())
    crouch_preserved_during_swing = bool(min_right_swing_knee_flexion >= MIN_SWING_KNEE_FLEXION_RAD)
    final_metrics = next(reversed(phase_metrics.values()))

    right_step_success = bool(
        right_step_attempted
        and max_forward_disp >= MIN_RIGHT_FOOT_FORWARD_DISPLACEMENT_M
        and max_right_clearance >= MIN_RIGHT_FOOT_CLEARANCE_PASS_M
        and min_single_margin > MIN_LIFTED_COM_MARGIN_M
        and min_margin_after_land > MIN_DOUBLE_SUPPORT_AFTER_STEP_M
        and stabilize_ok
        and landed_ok
        and max_swing_xy_drift <= MAX_SWING_XY_DRIFT_M
        and min_support_force > MIN_SUPPORT_FOOT_FORCE_N
        and crouch_preserved_during_swing
        and swing_foot_nonbacktracking
        and right_swing_clearance_success
    )

    success = bool(
        right_step_success
        and inside_all
        and right_returned_to_ground
        and swing_foot_nonbacktracking
        and right_swing_clearance_success
        and float(final_metrics["left_foot_force_n"]) > 0.0
        and float(final_metrics["right_foot_force_n"]) > 0.0
    )

    phases_completed_keys = list(phase_metrics.keys())
    support_to_swing_roll_ratio = float(
        max_support_leg_roll_delta / max(max_swing_leg_roll_delta, 1e-4)
    )

    out = dict(final_metrics)
    out.update(
        {
            "phase_sequence": [p.name for p in FirstStepPhase],
            "final_phase": final_phase.name,
            "phases_completed": phases_completed_keys,
            "phase_metrics": phase_metrics,
            "first_step_validation_only": True,
            "no_full_locomotion_attempted": True,
            "physical_grasp_validation": False,
            "box_payload_mode": "scripted_vertical_slide_scene_pose",
            "right_step_attempted": bool(right_step_attempted),
            "right_step_success": bool(right_step_success),
            "right_foot_forward_displacement_m": float(max_forward_disp),
            "max_right_foot_clearance_m": float(max_right_clearance),
            "right_foot_returned_to_ground": bool(right_returned_to_ground),
            "min_support_foot_force_n": float(min_support_force),
            "min_com_margin_m": float(min_com_margin),
            "min_com_margin_single_support_m": float(min_single_margin),
            "com_inside_support_all_phases": bool(inside_all),
            "max_pelvis_lateral_shift_m": float(max_pelvis_lateral_shift),
            "max_swing_foot_xy_drift_m": float(max_swing_xy_drift),
            "support_dominant_shift_enabled": True,
            "visual_shift_mode": "support_dominant",
            "max_support_leg_roll_delta_rad": float(max_support_leg_roll_delta),
            "max_swing_leg_roll_delta_rad": float(max_swing_leg_roll_delta),
            "support_to_swing_roll_ratio": support_to_swing_roll_ratio,
            "walk_ready_crouch_enabled": True,
            "max_knee_crouch_delta_rad": float(max_knee_crouch_delta),
            "max_hip_pitch_crouch_delta_rad": float(max_hip_pitch_crouch_delta),
            "max_ankle_pitch_crouch_delta_rad": float(max_ankle_pitch_crouch_delta),
            "visual_posture_mode": "bent_knee_nonbacktracking_step",
            "crouch_preserved_during_swing": crouch_preserved_during_swing,
            "swing_foot_nonbacktracking": swing_foot_nonbacktracking,
            "min_right_swing_foot_x_delta_m": float(min_right_swing_foot_x_delta_m),
            "right_swing_foot_backward_drift_m": right_swing_foot_backward_drift_m,
            "min_right_swing_knee_flexion_rad": float(min_right_swing_knee_flexion),
            "min_right_swing_foot_ground_clearance_m": float(
                min_right_swing_foot_ground_clearance_m
            ),
            "max_right_swing_foot_ground_clearance_m": float(
                max_right_swing_foot_ground_clearance_m
            ),
            "right_swing_foot_scrape_detected": bool(right_swing_foot_scrape_detected),
            "right_swing_clearance_success": bool(right_swing_clearance_success),
            "max_torso_pitch_rad": float(max_torso_pitch_seen),
            "payload_mass_kg": float(final_metrics["payload_mass_kg"]),
            "payload_moment_nm": float(final_metrics["payload_moment_nm"]),
            "torso_pitch_compensation_rad": float(torso_pitch),
            "success": success,
            "wall_time_s": float(time.time() - t0_wall),
        }
    )

    if verbose:
        print("----- payload_first_step -----")
        print("first_step_validation_only: True")
        print("no_full_locomotion_attempted: True")
        print("physical_grasp_validation: False")
        for phase_name in out["phase_sequence"]:
            print(f"phase: {phase_name}")
        for key in (
            "phases_completed",
            "right_step_attempted",
            "right_step_success",
            "right_foot_forward_displacement_m",
            "max_right_foot_clearance_m",
            "right_foot_returned_to_ground",
            "com_inside_support_all_phases",
            "min_support_foot_force_n",
            "min_com_margin_m",
            "min_com_margin_single_support_m",
            "max_pelvis_lateral_shift_m",
            "max_swing_foot_xy_drift_m",
            "support_dominant_shift_enabled",
            "visual_shift_mode",
            "max_support_leg_roll_delta_rad",
            "max_swing_leg_roll_delta_rad",
            "support_to_swing_roll_ratio",
            "walk_ready_crouch_enabled",
            "max_knee_crouch_delta_rad",
            "max_hip_pitch_crouch_delta_rad",
            "max_ankle_pitch_crouch_delta_rad",
            "visual_posture_mode",
            "crouch_preserved_during_swing",
            "swing_foot_nonbacktracking",
            "min_right_swing_foot_x_delta_m",
            "right_swing_foot_backward_drift_m",
            "min_right_swing_foot_ground_clearance_m",
            "max_right_swing_foot_ground_clearance_m",
            "right_swing_foot_scrape_detected",
            "right_swing_clearance_success",
            "min_right_swing_knee_flexion_rad",
            "max_torso_pitch_rad",
            "payload_mass_kg",
            "payload_moment_nm",
            "torso_pitch_compensation_rad",
            "success",
        ):
            val = out[key]
            if isinstance(val, float):
                print(f"{key}: {val:.5f}")
            else:
                print(f"{key}: {val}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 payload-aware quasi-static first forward step.")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (default: open MuJoCo passive viewer).",
    )
    ap.add_argument("--timeout", type=float, default=17.0, help="Simulation timeout in seconds.")
    ap.add_argument(
        "--viewer-speed",
        type=float,
        default=1.0,
        help="Viewer playback rate vs sim timestep (lower = slower). Ignored when --headless.",
    )
    args = ap.parse_args()
    out = run_g1_payload_first_step(
        headless=args.headless,
        timeout=args.timeout,
        viewer_speed=args.viewer_speed,
        verbose=True,
    )
    # Passive viewer + user_scn can segfault during normal interpreter teardown on some GLX builds.
    if not args.headless:
        os._exit(0 if bool(out.get("success")) else 1)


if __name__ == "__main__":
    main()
