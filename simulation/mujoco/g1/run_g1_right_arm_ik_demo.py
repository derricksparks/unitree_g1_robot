#!/usr/bin/env python3
"""
Locked-base G1: **task-space** goal for ``right_wrist_yaw_link`` via damped least-squares IK
on a 7-DOF subset (waist yaw/pitch + right arm through wrist pitch).

Uses finite-difference Jacobian, position actuators, and base lock from ``run_g1_posture_hold``.

Examples::

    python simulation/mujoco/g1/run_g1_right_arm_ik_demo.py
    python simulation/mujoco/g1/run_g1_right_arm_ik_demo.py --headless --timeout 8
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

from run_g1_posture_hold import (  # noqa: E402
    MODEL_PATH,
    NEUTRAL_POSTURE,
    DEFAULT_PELVIS_Z,
    PRINT_INTERVAL,
    SUFFIX_POS,
    apply_neutral_pose,
    build_actuator_id_map,
    floating_base_address_map,
    joint_name_from_actuator,
    command_position_actuators,
    stabilize_floating_base,
)

TARGET_BODY_NAME = "right_wrist_yaw_link"

IK_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
)

IK_CHAIN_KP_SCALE = 0.42
BEHIND_PELVIS_THRESHOLD = -0.05

IK_FD_EPS = 8e-5
IK_LAMBDA = 6.0e-3
IK_MAX_DQ = 0.04
IK_INNER_ITERS = 28
IK_POS_TOL = 5e-4
IK_POSTURE_GAIN = 0.03
IK_MAX_ABS_JOINT_FROM_NEUTRAL = 0.9


def _apply_kp_scale_for_joint_subset(model: mujoco.MjModel, joint_names: set[str], scale: float) -> None:
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not aname or not str(aname).endswith(SUFFIX_POS):
            continue
        jn = joint_name_from_actuator(str(aname))
        if jn not in joint_names:
            continue
        model.actuator_gainprm[aid, 0] *= scale
        model.actuator_biasprm[aid, 1] *= scale


def _build_ik_metadata(model: mujoco.MjModel) -> tuple[list[int], np.ndarray, np.ndarray]:
    qadrs: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for jn in IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        qadrs.append(int(model.jnt_qposadr[jid]))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qadrs, np.asarray(lows), np.asarray(highs)


def _fk_ik_point(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    main_qpos: np.ndarray,
    ik_qpos_adrs: list[int],
    q_work: np.ndarray,
    target_body_id: int,
    target_site_id: int,
) -> np.ndarray:
    np.copyto(fd.qpos, main_qpos)
    for i, adr in enumerate(ik_qpos_adrs):
        fd.qpos[adr] = float(q_work[i])
    fd.qvel[:] = 0.0
    mujoco.mj_forward(model, fd)
    if target_site_id >= 0:
        return np.asarray(fd.site_xpos[target_site_id, :3], dtype=float).copy()
    return np.asarray(fd.xpos[target_body_id, :3], dtype=float).copy()


def _ik_jacobian_fd(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    main_qpos: np.ndarray,
    ik_qpos_adrs: list[int],
    q_work: np.ndarray,
    target_body_id: int,
    target_site_id: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(ik_qpos_adrs)
    p0 = _fk_ik_point(
        model, fd, main_qpos, ik_qpos_adrs, q_work, target_body_id, target_site_id
    )
    jac = np.zeros((3, n), dtype=float)
    for j in range(n):
        q_pert = np.array(q_work, dtype=float, copy=True)
        q_pert[j] += eps
        p1 = _fk_ik_point(
            model, fd, main_qpos, ik_qpos_adrs, q_pert, target_body_id, target_site_id
        )
        jac[:, j] = (p1 - p0) / eps
    return jac, p0


def _fk_ik_site_xmat(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    main_qpos: np.ndarray,
    ik_qpos_adrs: list[int],
    q_work: np.ndarray,
    target_site_id: int,
) -> np.ndarray:
    np.copyto(fd.qpos, main_qpos)
    for i, adr in enumerate(ik_qpos_adrs):
        fd.qpos[adr] = float(q_work[i])
    fd.qvel[:] = 0.0
    mujoco.mj_forward(model, fd)
    return np.asarray(fd.site_xmat[target_site_id], dtype=float).reshape(3, 3).copy()


def _axis_align_err(n_curr: np.ndarray, n_des: np.ndarray) -> np.ndarray:
    nc = np.asarray(n_curr, dtype=float).reshape(3,)
    nd = np.asarray(n_des, dtype=float).reshape(3,)
    nc_norm = float(np.linalg.norm(nc))
    nd_norm = float(np.linalg.norm(nd))
    if nc_norm < 1e-12 or nd_norm < 1e-12:
        return np.zeros(3, dtype=float)
    nc = nc / nc_norm
    nd = nd / nd_norm
    return np.cross(nc, nd)


def _ik_jacobian_axis_fd(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    main_qpos: np.ndarray,
    ik_qpos_adrs: list[int],
    q_work: np.ndarray,
    target_site_id: int,
    n_des: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian (3 x n) of ``cross(n_curr, n_des)`` w.r.t. q (finite differences)."""
    n = len(ik_qpos_adrs)
    R0 = _fk_ik_site_xmat(model, fd, main_qpos, ik_qpos_adrs, q_work, target_site_id)
    n0 = R0[:, 0].copy()
    e0 = _axis_align_err(n0, n_des)
    jac = np.zeros((3, n), dtype=float)
    for j in range(n):
        q_pert = np.array(q_work, dtype=float, copy=True)
        q_pert[j] += eps
        R1 = _fk_ik_site_xmat(model, fd, main_qpos, ik_qpos_adrs, q_pert, target_site_id)
        n1 = R1[:, 0]
        e1 = _axis_align_err(n1, n_des)
        jac[:, j] = (e1 - e0) / eps
    return jac, e0


def _ik_jacobian_up_fd(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    main_qpos: np.ndarray,
    ik_qpos_adrs: list[int],
    q_work: np.ndarray,
    target_site_id: int,
    z_des: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian (3 x n) of ``cross(z_curr, z_des)`` for site local z axis in world."""
    n = len(ik_qpos_adrs)
    R0 = _fk_ik_site_xmat(model, fd, main_qpos, ik_qpos_adrs, q_work, target_site_id)
    z0 = R0[:, 2].copy()
    e0 = _axis_align_err(z0, z_des)
    jac = np.zeros((3, n), dtype=float)
    for j in range(n):
        q_pert = np.array(q_work, dtype=float, copy=True)
        q_pert[j] += eps
        R1 = _fk_ik_site_xmat(model, fd, main_qpos, ik_qpos_adrs, q_pert, target_site_id)
        z1 = R1[:, 2]
        e1 = _axis_align_err(z1, z_des)
        jac[:, j] = (e1 - e0) / eps
    return jac, e0


def _dls_nullspace_posture_step(
    jac: np.ndarray,
    err: np.ndarray,
    lam: float,
    max_dq: float,
    q_work: np.ndarray,
    q_neutral: np.ndarray,
    posture_gain: float,
) -> np.ndarray:
    """Task DLS step plus nullspace posture bias toward ``q_neutral``."""
    return _dls_nullspace_posture_step_general(
        jac, err, lam, max_dq, q_work, q_neutral, posture_gain
    )


def _dls_nullspace_posture_step_general(
    jac: np.ndarray,
    err: np.ndarray,
    lam: float,
    max_dq: float,
    q_work: np.ndarray,
    q_neutral: np.ndarray,
    posture_gain: float,
) -> np.ndarray:
    """DLS for ``m×n`` task Jacobian plus nullspace posture bias."""
    m, n = jac.shape
    jjt = jac @ jac.T + lam * np.eye(m, dtype=float)
    dq_task = jac.T @ np.linalg.solve(jjt, err)
    dq_posture = -posture_gain * (q_work - q_neutral)

    rhs = np.linalg.solve(jjt, jac)
    nmat = np.eye(n, dtype=float) - jac.T @ rhs
    dq = dq_task + nmat @ dq_posture

    dq_norm = float(np.linalg.norm(dq))
    if dq_norm > max_dq and dq_norm > 1e-12:
        dq *= max_dq / dq_norm
    return dq


def _smoothstep01(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * u)


def _cartesian_waypoint(
    t: float,
    duration: float,
    p_stow: np.ndarray,
    p_fwd: np.ndarray,
    p_up: np.ndarray,
) -> tuple[np.ndarray, str]:
    if duration <= 0.0:
        return p_stow.copy(), "STOW"
    t = float(np.clip(t, 0.0, duration))
    t0, t1, t2, t3 = 0.0, duration / 3.0, 2.0 * duration / 3.0, duration

    def seg_blend(
        tc: float,
        t_a: float,
        t_b: float,
        p_a: np.ndarray,
        p_b: np.ndarray,
        label_a: str,
        label_b: str,
    ):
        u_lin = (tc - t_a) / (t_b - t_a) if t_b > t_a else 1.0
        u = _smoothstep01(u_lin)
        return (1.0 - u) * p_a + u * p_b, (label_a if u_lin < 0.5 else label_b)

    if t <= t1:
        p, ph = seg_blend(t, t0, t1, p_stow, p_fwd, "STOW", "FORWARD")
    elif t <= t2:
        p, ph = seg_blend(t, t1, t2, p_fwd, p_up, "FORWARD", "UP")
    else:
        p, ph = seg_blend(t, t2, t3, p_up, p_stow, "UP", "STOW")
    return p.astype(float), ph


def solve_ik_q(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    data: mujoco.MjData,
    ik_qpos_adrs: list[int],
    q_low: np.ndarray,
    q_high: np.ndarray,
    target_body_id: int,
    target_xyz: np.ndarray,
    q_neutral_ik: np.ndarray,
    *,
    posture_gain: float,
    max_abs_joint_from_neutral: float,
    target_site_id: int = -1,
    inner_iters: int | None = None,
    palm_axis_target_world: np.ndarray | None = None,
    palm_axis_task_gain: float = 0.0,
    palm_up_target_world: np.ndarray | None = None,
    palm_up_task_gain: float = 0.0,
    position_task_gain: float = 1.0,
) -> tuple[np.ndarray, float]:
    q_work = np.array([float(data.qpos[adr]) for adr in ik_qpos_adrs], dtype=float)
    main_qpos = np.asarray(data.qpos, dtype=float).copy()
    q_band_low = np.maximum(q_low, q_neutral_ik - float(max_abs_joint_from_neutral))
    q_band_high = np.minimum(q_high, q_neutral_ik + float(max_abs_joint_from_neutral))
    err_norm = 0.0
    n_inner = IK_INNER_ITERS if inner_iters is None else int(inner_iters)
    use_ori = (
        target_site_id >= 0
        and (
            (palm_axis_task_gain > 1e-12 and palm_axis_target_world is not None)
            or (palm_up_task_gain > 1e-12 and palm_up_target_world is not None)
        )
    )
    n_des = (
        np.asarray(palm_axis_target_world, dtype=float).reshape(3,)
        if palm_axis_target_world is not None
        else np.zeros(3, dtype=float)
    )
    z_des = (
        np.asarray(palm_up_target_world, dtype=float).reshape(3,)
        if palm_up_target_world is not None
        else np.zeros(3, dtype=float)
    )
    ga = float(palm_axis_task_gain)
    gu = float(palm_up_task_gain)

    for _ in range(max(1, n_inner)):
        jac_pos, pos = _ik_jacobian_fd(
            model,
            fd,
            main_qpos,
            ik_qpos_adrs,
            q_work,
            target_body_id,
            target_site_id,
            IK_FD_EPS,
        )
        raw_err_pos = np.asarray(target_xyz, dtype=float) - pos
        err_norm = float(np.linalg.norm(raw_err_pos))
        pg = float(position_task_gain)
        err_pos = pg * raw_err_pos

        if not use_ori:
            jac = pg * jac_pos
            err = err_pos
        else:
            rows = [pg * jac_pos]
            errs = [err_pos]
            if ga > 1e-12:
                jax_ax, e_ax = _ik_jacobian_axis_fd(
                    model,
                    fd,
                    main_qpos,
                    ik_qpos_adrs,
                    q_work,
                    target_site_id,
                    n_des,
                    IK_FD_EPS,
                )
                rows.append(ga * jax_ax)
                errs.append(ga * e_ax)
            if gu > 1e-12:
                jax_up, e_up = _ik_jacobian_up_fd(
                    model,
                    fd,
                    main_qpos,
                    ik_qpos_adrs,
                    q_work,
                    target_site_id,
                    z_des,
                    IK_FD_EPS,
                )
                rows.append(gu * jax_up)
                errs.append(gu * e_up)
            jac = np.vstack(rows)
            err = np.concatenate(errs)

        if err_norm < IK_POS_TOL and not use_ori:
            break
        dq = _dls_nullspace_posture_step_general(
            jac,
            err,
            IK_LAMBDA,
            IK_MAX_DQ,
            q_work,
            q_neutral_ik,
            posture_gain,
        )
        q_work = np.minimum(np.maximum(q_work + dq, q_band_low), q_band_high)
        if err_norm < IK_POS_TOL and use_ori:
            # Continue a few iterations for soft orientation — outer loop count caps work.
            pass
    return q_work, err_norm


def run_right_arm_ik_demo(
    *,
    headless: bool = False,
    timeout: float = 8.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
) -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if model.nu != 31:
        raise RuntimeError(f"Expected nu=31 (29 hinges + proxy finger motors), got {model.nu}")

    hinge_names = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm:
            hinge_names.add(nm)
    if set(NEUTRAL_POSTURE.keys()) != hinge_names:
        raise ValueError("NEUTRAL_POSTURE must list all hinges")

    _apply_kp_scale_for_joint_subset(model, set(IK_JOINT_NAMES), IK_CHAIN_KP_SCALE)
    _apply_kp_scale_for_joint_subset(
        model, {"right_wrist_roll_joint", "right_wrist_yaw_joint"}, IK_CHAIN_KP_SCALE
    )

    data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    actuator_ids = build_actuator_id_map(model)
    base_map = floating_base_address_map(model)
    ik_qpos_adrs, q_low, q_high = _build_ik_metadata(model)
    q_neutral_ik = np.array(
        [float(NEUTRAL_POSTURE[jn]) for jn in IK_JOINT_NAMES], dtype=float
    )

    nominal_base_qpos = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=NEUTRAL_POSTURE,
        base_map=base_map,
    )

    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    if wrist_bid < 0:
        raise RuntimeError(f"Body {TARGET_BODY_NAME} not found")

    mujoco.mj_forward(model, data)
    p_stow = np.asarray(data.xpos[wrist_bid, :3], dtype=float).copy()
    p_fwd = p_stow + np.array([0.08, 0.0, 0.03], dtype=float)
    p_up = p_fwd + np.array([0.05, 0.0, 0.13], dtype=float)

    duration = max(float(timeout), model.opt.timestep * 4)

    max_err_trace = 0.0
    max_delta_x = 0.0
    max_delta_z = 0.0
    max_cmd_mag = 0.0
    last_print = -PRINT_INTERVAL

    def full_posture_from_ik(q_work: np.ndarray) -> dict[str, float]:
        full = dict(NEUTRAL_POSTURE)
        for jn, qv in zip(IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        return full

    def report(phase: str, tgt: np.ndarray, err_norm: float, q_work: np.ndarray) -> None:
        if not verbose:
            return
        wpos = np.asarray(data.xpos[wrist_bid, :3], dtype=float)
        cmd_peak = max(
            abs(float(q_work[i]) - float(NEUTRAL_POSTURE[IK_JOINT_NAMES[i]]))
            for i in range(len(IK_JOINT_NAMES))
        )
        warn = ""
        if wpos[0] < BEHIND_PELVIS_THRESHOLD:
            warn = "  WARNING: arm behind pelvis (wrist_x < -0.05)"
        print(
            f"[{phase}] t={data.time:.3f}s  wrist_xyz={wpos.tolist()}  "
            f"target_xyz={np.asarray(tgt, dtype=float).tolist()}  "
            f"|err|={err_norm:.5f}  ik_joint_max_mag={cmd_peak:.4f}{warn}"
        )

    def physics_frame(target_xyz: np.ndarray, phase: str) -> None:
        nonlocal max_err_trace, max_delta_x, max_delta_z, max_cmd_mag
        q_work, ik_err = solve_ik_q(
            model,
            fd,
            data,
            ik_qpos_adrs,
            q_low,
            q_high,
            wrist_bid,
            target_xyz,
            q_neutral_ik,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
        )
        max_err_trace = max(max_err_trace, ik_err)
        for i, jn in enumerate(IK_JOINT_NAMES):
            max_cmd_mag = max(
                max_cmd_mag,
                abs(float(q_work[i]) - float(NEUTRAL_POSTURE[jn])),
            )
        full = full_posture_from_ik(q_work)
        command_position_actuators(model, data, full, actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        wpos = np.asarray(data.xpos[wrist_bid, :3], dtype=float)
        max_delta_x = max(max_delta_x, float(wpos[0] - p_stow[0]))
        max_delta_z = max(max_delta_z, abs(float(wpos[2] - p_stow[2])))
        if verbose and data.time - last_print >= PRINT_INTERVAL:
            report(phase, target_xyz, ik_err, q_work)

    def step_frame() -> None:
        nonlocal last_print
        tgt, phase = _cartesian_waypoint(data.time, duration, p_stow, p_fwd, p_up)
        physics_frame(tgt, phase)
        if verbose and data.time - last_print >= PRINT_INTERVAL:
            last_print = data.time

    if headless:
        while data.time < timeout:
            step_frame()
        tgt, phase = _cartesian_waypoint(min(data.time, duration), duration, p_stow, p_fwd, p_up)
        q_work, ik_err = solve_ik_q(
            model,
            fd,
            data,
            ik_qpos_adrs,
            q_low,
            q_high,
            wrist_bid,
            tgt,
            q_neutral_ik,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
        )
        max_err_trace = max(max_err_trace, ik_err)
        for i, jn in enumerate(IK_JOINT_NAMES):
            max_cmd_mag = max(
                max_cmd_mag,
                abs(float(q_work[i]) - float(NEUTRAL_POSTURE[jn])),
            )
        full = full_posture_from_ik(q_work)
        command_position_actuators(model, data, full, actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        wpos = np.asarray(data.xpos[wrist_bid, :3], dtype=float)
        max_delta_x = max(max_delta_x, float(wpos[0] - p_stow[0]))
        max_delta_z = max(max_delta_z, abs(float(wpos[2] - p_stow[2])))
        report(phase, tgt, ik_err, q_work)
    else:
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()
                step_frame()
                viewer.sync()
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - t0)))
        tgt, phase = _cartesian_waypoint(min(data.time, duration), duration, p_stow, p_fwd, p_up)
        q_work, ik_err = solve_ik_q(
            model,
            fd,
            data,
            ik_qpos_adrs,
            q_low,
            q_high,
            wrist_bid,
            tgt,
            q_neutral_ik,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
        )
        max_err_trace = max(max_err_trace, ik_err)
        for i, jn in enumerate(IK_JOINT_NAMES):
            max_cmd_mag = max(
                max_cmd_mag,
                abs(float(q_work[i]) - float(NEUTRAL_POSTURE[jn])),
            )
        full = full_posture_from_ik(q_work)
        command_position_actuators(model, data, full, actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        wpos = np.asarray(data.xpos[wrist_bid, :3], dtype=float)
        max_delta_x = max(max_delta_x, float(wpos[0] - p_stow[0]))
        max_delta_z = max(max_delta_z, abs(float(wpos[2] - p_stow[2])))
        report(phase, tgt, ik_err, q_work)

    return {
        "model_path": str(MODEL_PATH),
        "sim_time": float(data.time),
        "p_stow": p_stow.tolist(),
        "p_forward_target": p_fwd.tolist(),
        "p_up_target": p_up.tolist(),
        "max_position_error_norm": float(max_err_trace),
        "max_wrist_delta_x_from_stow": float(max_delta_x),
        "max_wrist_delta_z_from_stow": float(max_delta_z),
        "max_ik_joint_command_magnitude": float(max_cmd_mag),
        "fix_base_in_world": bool(fix_base_in_world),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="G1 locked-base right-arm Cartesian IK demo.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--pelvis-z",
        type=float,
        default=DEFAULT_PELVIS_Z,
        help=f"Locked pelvis height (m), default {DEFAULT_PELVIS_Z}",
    )
    parser.add_argument(
        "--posture-gain",
        type=float,
        default=IK_POSTURE_GAIN,
        help=f"Nullspace posture gain (bias toward neutral), default {IK_POSTURE_GAIN}",
    )
    parser.add_argument(
        "--max-joint-from-neutral",
        type=float,
        default=IK_MAX_ABS_JOINT_FROM_NEUTRAL,
        help=(
            "Max |q_i - q_neutral_i| per IK joint (also clipped to model limits), "
            f"default {IK_MAX_ABS_JOINT_FROM_NEUTRAL}"
        ),
    )
    args = parser.parse_args(argv)

    try:
        run_right_arm_ik_demo(
            headless=args.headless,
            timeout=args.timeout,
            initial_pelvis_z=args.pelvis_z,
            verbose=True,
            posture_gain=args.posture_gain,
            max_joint_from_neutral=args.max_joint_from_neutral,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
