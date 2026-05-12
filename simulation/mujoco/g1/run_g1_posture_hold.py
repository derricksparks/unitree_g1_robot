#!/usr/bin/env python3
"""
First G1 integration sim: hold a conservative standing-like pose with position actuators.

Loads ``g1_position_actuated.xml`` next to this script — does **not** use the sliding-base prototype.

Examples::

    python simulation/mujoco/g1/run_g1_posture_hold.py
    python simulation/mujoco/g1/run_g1_posture_hold.py --headless --timeout 5
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

_HERE = Path(__file__).resolve().parent
MODEL_PATH = _HERE / "assets" / "g1_position_actuated.xml"

# Conservative standing-like neutral (radians). Symmetric L/R; tune per task.
NEUTRAL_POSTURE: dict[str, float] = {
    # Legs: mostly straight hips, slight knee flex, ankles compensate slightly
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.14,
    "left_ankle_pitch_joint": -0.07,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.14,
    "right_ankle_pitch_joint": -0.07,
    "right_ankle_roll_joint": 0.0,
    # Waist nominal
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # Arms: relaxed downward, slight elbow bend
    "left_shoulder_pitch_joint": -0.28,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.45,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.28,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.45,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

SUFFIX_POS = "_pos_actuator"
DEFAULT_PELVIS_Z = 0.84
PRINT_INTERVAL = 0.25


def build_actuator_id_map(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if name:
            out[name] = aid
    return out


def build_hinge_joint_address_map(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """All hinge joints: name -> qpos_adr, qvel_adr (scalar indices)."""
    out: dict[str, dict[str, int]] = {}
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname:
            continue
        out[jname] = {
            "qpos_adr": int(model.jnt_qposadr[jid]),
            "qvel_adr": int(model.jnt_dofadr[jid]),
        }
    return out


def floating_base_address_map(model: mujoco.MjModel) -> dict[str, int]:
    fj = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    if fj < 0:
        raise ValueError("floating_base_joint not found in model")
    return {
        "qpos_adr": int(model.jnt_qposadr[fj]),
        "qvel_adr": int(model.jnt_dofadr[fj]),
        "nq": 7,
        "nv": 6,
    }


def joint_name_from_actuator(act_name: str) -> str:
    if not act_name.endswith(SUFFIX_POS):
        raise ValueError(f"Unexpected actuator name: {act_name!r}")
    return act_name[: -len(SUFFIX_POS)]


def apply_neutral_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    initial_pelvis_z: float,
    neutral: dict[str, float],
    base_map: dict[str, int],
    base_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Initialize free base (identity quat), hinge qpos from ``neutral``, zero all qvel.

    Returns a length-7 snapshot of nominal floating-base ``qpos`` (px, py, pz, qw, qx, qy, qz).
    """
    qp = base_map["qpos_adr"]
    data.qpos[qp + 0 : qp + 3] = (base_xy[0], base_xy[1], initial_pelvis_z)
    data.qpos[qp + 3 : qp + 7] = (1.0, 0.0, 0.0, 0.0)  # w x y z

    hinge_addrs = build_hinge_joint_address_map(model)
    for jname, target in neutral.items():
        if jname not in hinge_addrs:
            raise KeyError(f"Neutral posture lists unknown hinge: {jname}")
        adr = hinge_addrs[jname]["qpos_adr"]
        data.qpos[adr] = float(target)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return np.asarray(data.qpos[qp + 0 : qp + 7], dtype=float).copy()


def stabilize_floating_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_map: dict[str, int],
    nominal_qpos7: np.ndarray,
) -> None:
    """Re-apply nominal floating-base pose and zero root velocity (world-frame hold)."""
    qp = base_map["qpos_adr"]
    qv = base_map["qvel_adr"]
    data.qpos[qp + 0 : qp + 7] = nominal_qpos7
    data.qvel[qv + 0 : qv + 6] = 0.0
    mujoco.mj_forward(model, data)


def command_position_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    neutral: dict[str, float],
    actuator_ids: dict[str, int],
) -> None:
    """Set ``data.ctrl`` from hinge ``neutral`` targets; non-hinge actuators default to 0."""
    for aname, aid in actuator_ids.items():
        if not str(aname).endswith(SUFFIX_POS):
            data.ctrl[aid] = 0.0
            continue
        jname = joint_name_from_actuator(str(aname))
        if jname not in neutral:
            raise KeyError(f"No neutral target for actuated joint {jname}")
        data.ctrl[aid] = float(neutral[jname])


def max_joint_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    neutral: dict[str, float],
) -> float:
    hinge_addrs = build_hinge_joint_address_map(model)
    merr = 0.0
    for jname, target in neutral.items():
        adr = hinge_addrs[jname]["qpos_adr"]
        merr = max(merr, abs(float(data.qpos[adr]) - float(target)))
    return merr


def pelvis_roll_pitch_deg(data: mujoco.MjData, model: mujoco.MjModel) -> tuple[float | None, float | None]:
    """Roll/pitch (rad) from pelvis world rotation matrix (ZYX-style extraction)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if bid < 0:
        return None, None
    r = data.xmat[bid].reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    return roll, pitch


def run_posture_hold(
    *,
    headless: bool = False,
    timeout: float = 60.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    neutral: dict[str, float] | None = None,
    verbose: bool = True,
    fix_base_in_world: bool = True,
) -> dict[str, Any]:
    """
    Load G1 position-actuated model, hold ``neutral`` pose, return final diagnostics.

    Args:
        headless: If True, step with no viewer until ``timeout`` sim seconds.
        timeout: Simulation time limit (seconds) when headless; ignored for viewer (runs until closed).
        initial_pelvis_z: Initial pelvis height (m) for free joint origin.
        fix_base_in_world: If True (default), reset floating root pose each step after ``mj_step`` so hinge
            position actuators behave dynamically without free-fall. Extra non-hinge position actuators
            (if any; e.g. proxy finger sliders) receive ``ctrl=0`` unless extended elsewhere.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    neutral_pose = NEUTRAL_POSTURE if neutral is None else neutral
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    hinge_names = set(build_hinge_joint_address_map(model).keys())
    if set(neutral_pose.keys()) != hinge_names:
        raise ValueError(
            "Neutral posture keys must match all hinge joints exactly; "
            f"missing={hinge_names - set(neutral_pose.keys())} "
            f"extra={set(neutral_pose.keys()) - hinge_names}"
        )
    actuator_ids = build_actuator_id_map(model)
    if len(actuator_ids) != model.nu:
        raise RuntimeError("Duplicate or missing actuator names")
    hinge_actuator_names = [a for a in actuator_ids if str(a).endswith(SUFFIX_POS)]
    if len(hinge_actuator_names) != len(hinge_names):
        raise RuntimeError(
            "Each hinge joint must have exactly one position actuator ending with "
            f"{SUFFIX_POS!r}; hinge_count={len(hinge_names)}, "
            f"matching_actuators={len(hinge_actuator_names)}, nu={model.nu}"
        )
    for aname in actuator_ids:
        if str(aname).endswith(SUFFIX_POS):
            jn = joint_name_from_actuator(str(aname))
            if jn not in neutral_pose:
                raise KeyError(f"Actuator {aname} has no neutral target for joint {jn}")

    data = mujoco.MjData(model)

    base_map = floating_base_address_map(model)

    nominal_base_qpos = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=neutral_pose,
        base_map=base_map,
    )
    command_position_actuators(model, data, neutral_pose, actuator_ids)

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def report_line(tag: str) -> None:
        if not verbose:
            return
        pos = np.asarray(data.xpos[pelvis_bid], dtype=float).copy()
        roll, pitch = pelvis_roll_pitch_deg(data, model)
        mje = max_joint_error(model, data, neutral_pose)
        rpy = "n/a"
        if roll is not None and pitch is not None:
            rpy = f"roll={np.degrees(roll):.2f}° pitch={np.degrees(pitch):.2f}°"
        print(
            f"[{tag}] t={data.time:.3f}s  pelvis_xyz={pos}  {rpy}  "
            f"max_joint_err={mje:.4f} rad ({np.degrees(mje):.2f}°)"
        )

    report_line("init")

    def physics_substep() -> None:
        command_position_actuators(model, data, neutral_pose, actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )

    if headless:
        last_print = -PRINT_INTERVAL
        while data.time < timeout:
            physics_substep()
            if verbose and data.time - last_print >= PRINT_INTERVAL:
                report_line("hold")
                last_print = data.time
        report_line("final")
    else:
        viewer_mod = importlib.import_module("mujoco.viewer")
        last_print = -PRINT_INTERVAL
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                physics_substep()
                viewer.sync()
                if verbose and data.time - last_print >= PRINT_INTERVAL:
                    report_line("hold")
                    last_print = data.time
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - step_start)))
        report_line("final")

    pos_f = np.asarray(data.xpos[pelvis_bid], dtype=float).copy()
    roll_f, pitch_f = pelvis_roll_pitch_deg(data, model)
    err_f = max_joint_error(model, data, neutral_pose)
    return {
        "model_path": str(MODEL_PATH),
        "sim_time": float(data.time),
        "pelvis_xyz": pos_f.tolist(),
        "pelvis_roll_rad": roll_f,
        "pelvis_pitch_rad": pitch_f,
        "max_joint_error_rad": float(err_f),
        "fix_base_in_world": bool(fix_base_in_world),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="G1 posture hold (position actuators).")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--pelvis-z",
        type=float,
        default=DEFAULT_PELVIS_Z,
        help=f"Initial pelvis height (m), default {DEFAULT_PELVIS_Z}",
    )
    parser.add_argument(
        "--free-base",
        action="store_true",
        help="Do not reset floating base after each step (robot can fall / drift).",
    )
    args = parser.parse_args(argv)

    try:
        run_posture_hold(
            headless=args.headless,
            timeout=args.timeout,
            initial_pelvis_z=args.pelvis_z,
            verbose=True,
            fix_base_in_world=not args.free_base,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
