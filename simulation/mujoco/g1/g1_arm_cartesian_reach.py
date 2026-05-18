"""
Shoulder-fixed arm Cartesian reach: position-only IK with explicit joint locking.

The torso/waist and non-active joints stay fixed in ``qpos``; only the requested
arm chain is solved toward a world-space target. No palm orientation tasks (avoids
wrist roll/yaw fighting the reach).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import mujoco
import numpy as np

from run_g1_right_arm_ik_demo import (
    IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    IK_MAX_DQ,
    IK_POSTURE_GAIN,
    solve_ik_q,
)

# Right arm only — waist and left arm are not in this chain.
RIGHT_ARM_SHOULDER_CHAIN: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)

RIGHT_WRIST_FREEZE_JOINTS: tuple[str, ...] = (
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)
RIGHT_WRIST_PITCH_JOINT = "right_wrist_pitch_joint"
RIGHT_WRIST_ROLL_YAW_JOINTS = RIGHT_WRIST_FREEZE_JOINTS


def build_chain_metadata(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    return _build_named(model, joint_names)


def _build_named(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    qadrs: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn!r}")
        qadrs.append(int(model.jnt_qposadr[jid]))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qadrs, np.asarray(lows), np.asarray(highs)


def hinge_joint_qpos_adrs(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm:
            out[str(nm)] = int(model.jnt_qposadr[jid])
    return out


def frozen_qpos_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    active_joint_names: Sequence[str],
) -> dict[int, float]:
    """Latch every hinge *except* the active IK chain (waist, other arm, fingers, …)."""
    active = set(active_joint_names)
    snap: dict[int, float] = {}
    for jn, adr in hinge_joint_qpos_adrs(model).items():
        if jn in active:
            continue
        snap[int(adr)] = float(data.qpos[int(adr)])
    return snap


def apply_frozen_qpos(data: mujoco.MjData, frozen: Mapping[int, float]) -> None:
    for adr, val in frozen.items():
        data.qpos[int(adr)] = float(val)


def solve_position_only(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    data: mujoco.MjData,
    *,
    ik_qpos_adrs: list[int],
    q_low: np.ndarray,
    q_high: np.ndarray,
    q_neutral: np.ndarray,
    body_id: int,
    site_id: int,
    target_xyz: np.ndarray,
    frozen_qpos: Mapping[int, float],
    posture_gain: float = IK_POSTURE_GAIN,
    max_abs_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    inner_iters: int = 14,
) -> tuple[np.ndarray, float]:
    """Position-only IK; restores frozen joints before/after solve."""
    apply_frozen_qpos(data, frozen_qpos)
    q_work, err = solve_ik_q(
        model,
        fd,
        data,
        ik_qpos_adrs,
        q_low,
        q_high,
        body_id,
        np.asarray(target_xyz, dtype=float).reshape(3),
        q_neutral,
        posture_gain=float(posture_gain),
        max_abs_joint_from_neutral=float(max_abs_joint_from_neutral),
        target_site_id=int(site_id),
        inner_iters=int(inner_iters),
        palm_axis_task_gain=0.0,
        palm_up_task_gain=0.0,
    )
    for i, adr in enumerate(ik_qpos_adrs):
        data.qpos[int(adr)] = float(q_work[i])
    apply_frozen_qpos(data, frozen_qpos)
    mujoco.mj_forward(model, data)
    return q_work, float(err)


def palm_site_xyz(
    model: mujoco.MjModel, data: mujoco.MjData, *, site_id: int, body_id: int
) -> np.ndarray:
    if site_id >= 0:
        return np.asarray(data.site_xpos[site_id, :3], dtype=float).copy()
    return np.asarray(data.xpos[body_id, :3], dtype=float).copy()


def add_joints_to_frozen(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frozen: dict[int, float],
    joint_names: Sequence[str],
) -> None:
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        frozen[adr] = float(data.qpos[adr])


def remove_joints_from_frozen(
    frozen: dict[int, float],
    joint_names: Sequence[str],
    *,
    model: mujoco.MjModel,
) -> None:
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        frozen.pop(int(model.jnt_qposadr[jid]), None)
