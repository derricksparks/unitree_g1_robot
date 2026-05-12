#!/usr/bin/env python3
"""
Locked-base G1 demo: interpolate the **right arm** (and modest **waist** coupling) through
stow → forward → up → stow under smooth commands.

Reuses helpers from ``run_g1_posture_hold.py``. Base pose is locked each step like posture hold.

Examples::

    python simulation/mujoco/g1/run_g1_right_arm_reach.py
    python simulation/mujoco/g1/run_g1_right_arm_reach.py --headless --timeout 8
"""

from __future__ import annotations

import argparse
import importlib
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

from run_g1_posture_hold import (  # noqa: E402
    MODEL_PATH,
    NEUTRAL_POSTURE,
    DEFAULT_PELVIS_Z,
    PRINT_INTERVAL,
    SUFFIX_POS,
    apply_neutral_pose,
    build_actuator_id_map,
    build_hinge_joint_address_map,
    command_position_actuators,
    floating_base_address_map,
    joint_name_from_actuator,
    stabilize_floating_base,
)

RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

WAIST_ADJUST_KEYS: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
)

# Smooth only these commanded joints (right arm + small waist coupling for reach IK).
COMMAND_SMOOTH_KEYS: tuple[str, ...] = RIGHT_ARM_JOINTS + WAIST_ADJUST_KEYS

BEHIND_PELVIS_THRESHOLD = -0.05
RIGHT_ARM_KP_SCALE = 0.48  # Softer servo on right arm reduces limit-cycle shake.
EMA_TIME_CONSTANT_S = 0.22


class ReachPhase(Enum):
    ARM_STOW = auto()
    ARM_REACH_FORWARD = auto()
    ARM_REACH_UP = auto()


def merge_keyframe(patch: dict[str, float]) -> dict[str, float]:
    """Full hinge posture: legs + left arm + torso default from NEUTRAL_POSTURE, plus patch."""
    full = dict(NEUTRAL_POSTURE)
    for k, v in patch.items():
        full[k] = float(v)
    return full


# Stow matches NEUTRAL_POSTURE everywhere (baseline standing + relaxed arms).
FULL_ARM_STOW = dict(NEUTRAL_POSTURE)

# Forward: FK-tuned wrist x advances ~+0.10 m vs nominal stow while z rises slightly.
# Requires modest waist yaw+pitch inside MJCF hinge limits (“near zero” torso vs large twists).
_PATCH_FORWARD: dict[str, float] = {
    "waist_yaw_joint": 0.40,
    "waist_pitch_joint": 0.09,
    "waist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": -0.50,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": -0.12,
    "right_elbow_joint": 0.37,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
FULL_ARM_REACH_FORWARD = merge_keyframe(_PATCH_FORWARD)

# Raised reach: FK-tuned wrist z ~+13 cm vs stow, x stays > 0.28 (ahead of hips).
_PATCH_UP: dict[str, float] = {
    "waist_yaw_joint": 0.20,
    "waist_pitch_joint": 0.04,
    "waist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": -0.52,
    "right_shoulder_roll_joint": -0.24,
    "right_shoulder_yaw_joint": -0.08,
    "right_elbow_joint": 0.07,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
FULL_ARM_REACH_UP = merge_keyframe(_PATCH_UP)


def _smoothstep01(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    # Cosine ease: zero velocity derivative at endpoints (reduces command jerk vs linear alpha).
    return 0.5 - 0.5 * np.cos(np.pi * u)


def _lerp_full_pose(a: dict[str, float], b: dict[str, float], u_sm: float) -> dict[str, float]:
    u = float(np.clip(u_sm, 0.0, 1.0))
    return {k: float((1.0 - u) * a[k] + u * b[k]) for k in NEUTRAL_POSTURE}


def full_pose_at_trajectory_time(t: float, duration: float) -> tuple[dict[str, float], str]:
    """Smooth piecewise blends: STOW → FORWARD → UP → STOW over ``duration`` (sim seconds)."""
    if duration <= 0.0:
        return dict(FULL_ARM_STOW), ReachPhase.ARM_STOW.name

    t = float(np.clip(t, 0.0, duration))
    t0, t1, t2, t3 = 0.0, duration / 3.0, 2.0 * duration / 3.0, duration

    if t <= t1:
        u_lin = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        pose = _lerp_full_pose(FULL_ARM_STOW, FULL_ARM_REACH_FORWARD, _smoothstep01(u_lin))
        phase = ReachPhase.ARM_STOW.name if u_lin < 0.5 else ReachPhase.ARM_REACH_FORWARD.name
    elif t <= t2:
        u_lin = (t - t1) / (t2 - t1) if t2 > t1 else 1.0
        pose = _lerp_full_pose(FULL_ARM_REACH_FORWARD, FULL_ARM_REACH_UP, _smoothstep01(u_lin))
        phase = ReachPhase.ARM_REACH_FORWARD.name if u_lin < 0.5 else ReachPhase.ARM_REACH_UP.name
    else:
        u_lin = (t - t2) / (t3 - t2) if t3 > t2 else 1.0
        pose = _lerp_full_pose(FULL_ARM_REACH_UP, FULL_ARM_STOW, _smoothstep01(u_lin))
        phase = ReachPhase.ARM_REACH_UP.name if u_lin < 0.5 else ReachPhase.ARM_STOW.name

    return pose, phase


def apply_right_arm_kp_scaling(model: mujoco.MjModel, scale: float) -> None:
    """Reduce position actuator gains for ``RIGHT_ARM_JOINTS`` transmitters only."""
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not aname or not str(aname).endswith(SUFFIX_POS):
            continue
        jn = joint_name_from_actuator(str(aname))
        if jn not in RIGHT_ARM_JOINTS:
            continue
        model.actuator_gainprm[aid, 0] *= scale
        model.actuator_biasprm[aid, 1] *= scale


def max_right_arm_joint_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    full_targets: dict[str, float],
) -> float:
    hinge = build_hinge_joint_address_map(model)
    merr = 0.0
    for k in RIGHT_ARM_JOINTS:
        adr = hinge[k]["qpos_adr"]
        merr = max(merr, abs(float(data.qpos[adr]) - float(full_targets[k])))
    return merr


def right_hand_position(data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if bid < 0:
        return None
    return np.asarray(data.xpos[bid], dtype=float).copy()


def run_right_arm_reach(
    *,
    headless: bool = False,
    timeout: float = 6.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    kp_scale_right_arm: float = RIGHT_ARM_KP_SCALE,
    ema_time_constant_s: float = EMA_TIME_CONSTANT_S,
) -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if model.nu != 31:
        raise RuntimeError(f"Expected nu=31 (29 hinges + proxy finger motors), got {model.nu}")

    apply_right_arm_kp_scaling(model, kp_scale_right_arm)

    hinge_names = set(build_hinge_joint_address_map(model).keys())
    if set(NEUTRAL_POSTURE.keys()) != hinge_names:
        raise ValueError("NEUTRAL_POSTURE must list all hinges for this model")

    data = mujoco.MjData(model)
    actuator_ids = build_actuator_id_map(model)
    base_map = floating_base_address_map(model)

    nominal_base_qpos = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=NEUTRAL_POSTURE,
        base_map=base_map,
    )

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    duration = max(float(timeout), model.opt.timestep * 4)

    stow_hand = right_hand_position(data, model)
    if stow_hand is None:
        raise RuntimeError("Missing right_wrist_yaw_link body for diagnostics")
    ref_x, ref_z = float(stow_hand[0]), float(stow_hand[2])

    smoothed_pose = dict(NEUTRAL_POSTURE)
    max_target_delta_ra = 0.0

    beta = float(np.clip(model.opt.timestep / max(ema_time_constant_s, 1e-6), 0.0, 1.0))

    last_print = -PRINT_INTERVAL

    def report(phase: str, cmd_pose: dict[str, float]) -> None:
        if not verbose:
            return
        hp = right_hand_position(data, model)
        if hp is None:
            wx, wz, dx_m, dz_m = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            wx, wz = float(hp[0]), float(hp[2])
            dx_m = wx - ref_x
            dz_m = wz - ref_z

        pelvis = np.asarray(data.xpos[pelvis_bid], dtype=float).copy()
        rerr = max_right_arm_joint_error(model, data, cmd_pose)
        warn = ""
        if hp is not None and wx < BEHIND_PELVIS_THRESHOLD:
            warn = "  WARNING: arm behind pelvis (wrist_x < -0.05)"
        print(
            f"[{phase}] t={data.time:.3f}s  "
            f"wrist_xz=({wx:.4f},{wz:.4f})  "
            f"Δx_stow={dx_m:+.4f}  Δz_stow={dz_m:+.4f}  "
            f"pelvis_xyz={pelvis.tolist()}  "
            f"right_arm_max_err={rerr:.4f} rad{warn}"
        )

    def physics_substep(desired_pose: dict[str, float]) -> None:
        nonlocal smoothed_pose, max_target_delta_ra
        # Exponential smoothing on actuator commands only for arm + waist keys.
        for k in COMMAND_SMOOTH_KEYS:
            smoothed_pose[k] = float(
                smoothed_pose[k]
                + beta * (float(desired_pose[k]) - float(smoothed_pose[k]))
            )
            if k in RIGHT_ARM_JOINTS:
                max_target_delta_ra = max(
                    max_target_delta_ra,
                    abs(float(desired_pose[k]) - float(NEUTRAL_POSTURE[k])),
                )
        command_position_actuators(model, data, smoothed_pose, actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )

    def step_frame() -> None:
        nonlocal last_print
        desired, phase = full_pose_at_trajectory_time(data.time, duration)
        physics_substep(desired)
        if verbose and data.time - last_print >= PRINT_INTERVAL:
            report(phase, smoothed_pose)
            last_print = data.time

    if headless:
        while data.time < timeout:
            step_frame()
        phase = full_pose_at_trajectory_time(min(data.time, duration), duration)[1]
        report(phase, smoothed_pose)
    else:
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                step_frame()
                viewer.sync()
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - step_start)))
        phase = full_pose_at_trajectory_time(min(data.time, duration), duration)[1]
        report(phase, smoothed_pose)

    pos_f = np.asarray(data.xpos[pelvis_bid], dtype=float).tolist()

    return {
        "model_path": str(MODEL_PATH),
        "sim_time": float(data.time),
        "pelvis_xyz": pos_f,
        "reference_stow_wrist_x": ref_x,
        "reference_stow_wrist_z": ref_z,
        "max_right_arm_target_delta_from_neutral": float(max_target_delta_ra),
        "fix_base_in_world": bool(fix_base_in_world),
        "kp_scale_right_arm": float(kp_scale_right_arm),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="G1 locked-base right-arm reach demo.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument(
        "--pelvis-z",
        type=float,
        default=DEFAULT_PELVIS_Z,
        help=f"Locked pelvis height (m), default {DEFAULT_PELVIS_Z}",
    )
    args = parser.parse_args(argv)

    try:
        run_right_arm_reach(
            headless=args.headless,
            timeout=args.timeout,
            initial_pelvis_z=args.pelvis_z,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
