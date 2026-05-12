#!/usr/bin/env python3
"""
Locked-base G1 reach-to-box demo (motion / IK smoke).

**Deprecated for contact work:** use ``run_g1_touch_box.py`` for a solid box contact demo with the
same pre-grasp target and stricter validation. This script keeps a lighter **stow → approach →
reach → stow** motion pattern but now aims at the **same precontact world point** as the touch demo
(not the box center).

Loads ``g1_reach_box_scene_single_arm_offset.xml`` (includes ``g1_position_actuated.xml``, floor, static box).

IK tracks ``right_palm_site`` when present; otherwise falls back to ``right_wrist_yaw_link`` body origin.

Examples::

    python simulation/mujoco/g1/run_g1_reach_box.py
    python simulation/mujoco/g1/run_g1_reach_box.py --headless --timeout 8
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

from g1_precontact import (  # noqa: E402
    APPROACH_ABOVE_OFFSET,
    TOUCH_DISTANCE_M,
    palm_precontact_target_world,
)

from run_g1_posture_hold import (  # noqa: E402
    NEUTRAL_POSTURE,
    DEFAULT_PELVIS_Z,
    PRINT_INTERVAL,
    apply_neutral_pose,
    build_actuator_id_map,
    floating_base_address_map,
    command_position_actuators,
    stabilize_floating_base,
)

from run_g1_right_arm_ik_demo import (  # noqa: E402
    IK_JOINT_NAMES,
    IK_POSTURE_GAIN,
    IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    solve_ik_q,
    _build_ik_metadata,
    _apply_kp_scale_for_joint_subset,
    _smoothstep01,
)

SCENE_PATH = _G1 / "g1_reach_box_scene_single_arm_offset.xml"
BOX_BODY_NAME = "reach_target_box"
BOX_GEOM_NAME = "box_geom"
LEGACY_BOX_GEOM_NAME = "reach_box_geom"

TARGET_SITE_NAME = "right_palm_site"
FALLBACK_TARGET_BODY_NAME = "right_wrist_yaw_link"

# Loose smoke threshold if the cyclic reach motion differs slightly from stationary touch hold.
SUCCESS_PRECONTACT_DIST_M = TOUCH_DISTANCE_M * 3.6

REACH_IK_KP_SCALE = 0.82


def _ik_qvel_adrs(model: mujoco.MjModel) -> list[int]:
    out: list[int] = []
    for jn in IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        out.append(int(model.jnt_dofadr[jid]))
    return out


def _hand_point_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_site_id: int,
    fallback_body_id: int,
) -> np.ndarray:
    if target_site_id >= 0:
        return np.asarray(data.site_xpos[target_site_id, :3], dtype=float).copy()
    return np.asarray(data.xpos[fallback_body_id, :3], dtype=float).copy()


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_scene_model() -> mujoco.MjModel:
    """Load scene XML with working directory set for mesh includes."""
    scene_dir = SCENE_PATH.resolve().parent
    with _chdir(scene_dir):
        return mujoco.MjModel.from_xml_path(SCENE_PATH.name)


def _box_geom_id(model: mujoco.MjModel) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    if gid >= 0:
        return int(gid)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEGACY_BOX_GEOM_NAME)
    if gid < 0:
        raise RuntimeError(f"Neither {BOX_GEOM_NAME} nor {LEGACY_BOX_GEOM_NAME} found")
    return int(gid)


def _box_target_point(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    gid = _box_geom_id(model)
    return np.asarray(data.geom_xpos[gid, :3], dtype=float).copy()


def _waypoint_stow_approach_reach(
    sim_t: float,
    duration: float,
    p_stow: np.ndarray,
    p_above: np.ndarray,
    p_reach: np.ndarray,
) -> tuple[np.ndarray, str]:
    if duration <= 0.0:
        return p_stow.copy(), "STOW"
    t = float(np.clip(sim_t, 0.0, duration))
    t0, t1, t2, t3 = 0.0, duration / 3.0, 2.0 * duration / 3.0, duration

    def seg_blend(
        tc: float,
        t_a: float,
        t_b: float,
        p_a: np.ndarray,
        p_b: np.ndarray,
        label_a: str,
        label_b: str,
    ) -> tuple[np.ndarray, str]:
        u_lin = (tc - t_a) / (t_b - t_a) if t_b > t_a else 1.0
        u = _smoothstep01(u_lin)
        return (1.0 - u) * p_a + u * p_b, (label_a if u_lin < 0.5 else label_b)

    if t <= t1:
        p, ph = seg_blend(t, t0, t1, p_stow, p_above, "STOW", "APPROACH_ABOVE_BOX")
    elif t <= t2:
        p, ph = seg_blend(t, t1, t2, p_above, p_reach, "APPROACH_ABOVE_BOX", "REACH_BOX")
    else:
        p, ph = seg_blend(t, t2, t3, p_reach, p_stow, "REACH_BOX", "STOW")
    return p.astype(float), ph


def _palm_precontact_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    wrist_bid: int,
    ik_target_site_id: int,
    p_precontact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    palm = _hand_point_world(
        model, data, target_site_id=ik_target_site_id, fallback_body_id=wrist_bid
    )
    box = _box_target_point(model, data)
    dist = float(np.linalg.norm(palm - p_precontact))
    return (
        palm,
        box,
        np.asarray(data.xpos[wrist_bid, :3], dtype=float).copy(),
        dist,
    )


def run_g1_reach_box(
    *,
    headless: bool = False,
    timeout: float = 8.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
) -> dict[str, Any]:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing scene file: {SCENE_PATH}")

    base_g1_xml = _G1 / "assets" / "g1_position_actuated.xml"
    if not base_g1_xml.is_file():
        raise FileNotFoundError(f"Missing packaged model: {base_g1_xml}")

    model = _load_scene_model()
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

    _apply_kp_scale_for_joint_subset(model, set(IK_JOINT_NAMES), REACH_IK_KP_SCALE)
    _apply_kp_scale_for_joint_subset(
        model, {"right_wrist_roll_joint", "right_wrist_yaw_joint"}, REACH_IK_KP_SCALE
    )

    data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    actuator_ids = build_actuator_id_map(model)
    base_map = floating_base_address_map(model)
    ik_qpos_adrs, q_low, q_high = _build_ik_metadata(model)
    ik_qvel_adrs = _ik_qvel_adrs(model)
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

    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FALLBACK_TARGET_BODY_NAME)
    if wrist_bid < 0:
        raise RuntimeError(f"Body {FALLBACK_TARGET_BODY_NAME} not found")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME) < 0:
        raise RuntimeError(f"Body {BOX_BODY_NAME} not found")

    target_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TARGET_SITE_NAME))
    ik_target_site_id = target_site_id if target_site_id >= 0 else -1
    used_target_site = TARGET_SITE_NAME if ik_target_site_id >= 0 else FALLBACK_TARGET_BODY_NAME

    mujoco.mj_forward(model, data)

    p_center = _box_target_point(model, data)
    p_precontact = palm_precontact_target_world(p_center)
    p_above = p_precontact + APPROACH_ABOVE_OFFSET
    p_reach = np.array(p_precontact, dtype=float, copy=True)

    p_stow = _hand_point_world(
        model, data, target_site_id=ik_target_site_id, fallback_body_id=wrist_bid
    )

    duration = max(float(timeout), model.opt.timestep * 4)

    max_err_trace = 0.0
    max_cmd_mag = 0.0
    min_precontact_any = float("inf")
    min_precontact_reach = float("inf")
    last_print = -PRINT_INTERVAL
    done_line_printed = False

    def full_posture_from_ik(q_work: np.ndarray) -> dict[str, float]:
        full = dict(NEUTRAL_POSTURE)
        for jn, qv in zip(IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        return full

    def report(
        phase: str,
        tgt: np.ndarray,
        ik_err: float,
        q_work: np.ndarray,
        wpos: np.ndarray,
        palm_pos: np.ndarray,
        box_pos: np.ndarray,
        palm_precontact_dist: float,
    ) -> None:
        if not verbose:
            return
        cmd_peak = max(
            abs(float(q_work[i]) - float(NEUTRAL_POSTURE[IK_JOINT_NAMES[i]]))
            for i in range(len(IK_JOINT_NAMES))
        )
        print(
            f"[{phase}] t={data.time:.3f}s  wrist_xyz={wpos.tolist()}  "
            f"palm_xyz={palm_pos.tolist()}  box_xyz={box_pos.tolist()}  "
            f"palm_precontact_distance={palm_precontact_dist:.4f}  "
            f"precontact_target={p_precontact.tolist()}  "
            f"|ik_err|={ik_err:.5f}  ik_joint_max_mag={cmd_peak:.4f}"
        )

    def track_reach_distance(sim_t: float, phase: str) -> bool:
        """True when evaluating success distance (middle segment, near reach target)."""
        if duration <= 0.0:
            return False
        if phase == "DONE":
            return False
        t = float(np.clip(sim_t, 0.0, duration))
        t1 = duration / 3.0
        t2 = 2.0 * duration / 3.0
        if not (t1 <= t <= t2):
            return False
        if phase == "REACH_BOX":
            return True
        u = (t - t1) / (t2 - t1)
        return u >= 0.42

    def physics_frame(
        target_xyz: np.ndarray,
        phase: str,
        *,
        silent: bool = False,
        force_done_report: bool = False,
    ) -> None:
        nonlocal max_err_trace, max_cmd_mag, min_precontact_any
        nonlocal min_precontact_reach, last_print, done_line_printed

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
            target_site_id=ik_target_site_id,
        )
        max_err_trace = max(max_err_trace, ik_err)
        for i, jn in enumerate(IK_JOINT_NAMES):
            max_cmd_mag = max(
                max_cmd_mag,
                abs(float(q_work[i]) - float(NEUTRAL_POSTURE[jn])),
            )
        full = full_posture_from_ik(q_work)
        for i, adr in enumerate(ik_qpos_adrs):
            data.qpos[adr] = float(q_work[i])
        for adr in ik_qvel_adrs:
            data.qvel[adr] = 0.0
        command_position_actuators(model, data, full, actuator_ids)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        mujoco.mj_forward(model, data)
        data.time += model.opt.timestep

        palm_pos, box_pos, wpos, pcdist = _palm_precontact_report(
            model,
            data,
            wrist_bid=wrist_bid,
            ik_target_site_id=ik_target_site_id,
            p_precontact=p_precontact,
        )
        min_precontact_any = min(min_precontact_any, pcdist)
        if track_reach_distance(data.time, phase):
            min_precontact_reach = min(min_precontact_reach, pcdist)

        if not verbose or silent:
            return
        if phase == "DONE":
            if force_done_report and not done_line_printed:
                report(
                    phase, target_xyz, ik_err, q_work, wpos, palm_pos, box_pos, pcdist
                )
                done_line_printed = True
            return
        if data.time - last_print >= PRINT_INTERVAL:
            report(
                phase, target_xyz, ik_err, q_work, wpos, palm_pos, box_pos, pcdist
            )
            last_print = data.time

    def target_for_time(sim_t: float) -> tuple[np.ndarray, str]:
        if sim_t >= duration:
            return p_stow, "DONE"
        return _waypoint_stow_approach_reach(sim_t, duration, p_stow, p_above, p_reach)

    mission_complete = False

    def run_headless_loop() -> None:
        nonlocal mission_complete
        while True:
            tgt, phase = target_for_time(data.time)
            if phase == "DONE":
                physics_frame(tgt, phase, force_done_report=True)
                mission_complete = True
                return
            if data.time >= timeout:
                return
            physics_frame(tgt, phase)

    def run_viewer_loop() -> None:
        nonlocal mission_complete
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_t0 = time.time()
                if mission_complete:
                    physics_frame(p_stow, "DONE", silent=True)
                else:
                    tgt, phase = target_for_time(data.time)
                    if phase == "DONE":
                        physics_frame(tgt, phase, force_done_report=True)
                        mission_complete = True
                    else:
                        physics_frame(tgt, phase)
                viewer.sync()
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - loop_t0)))

    if headless:
        run_headless_loop()
    else:
        run_viewer_loop()

    mujoco.mj_forward(model, data)
    _, _, _, final_precontact_dist = _palm_precontact_report(
        model,
        data,
        wrist_bid=wrist_bid,
        ik_target_site_id=ik_target_site_id,
        p_precontact=p_precontact,
    )

    success = np.isfinite(min_precontact_reach) and (
        min_precontact_reach < SUCCESS_PRECONTACT_DIST_M
    )

    out = {
        "model_path": str(SCENE_PATH.resolve()),
        "sim_time": float(data.time),
        "motion_duration": float(duration),
        "motion_complete": mission_complete,
        "preferred_contact_demo_script": str((_G1 / "run_g1_touch_box.py").resolve()),
        "ik_target_site_id": ik_target_site_id,
        "ik_site_name": TARGET_SITE_NAME if ik_target_site_id >= 0 else None,
        "used_target_site": used_target_site,
        "p_box_center_world": p_center.tolist(),
        "p_precontact_world": p_precontact.tolist(),
        "p_above_target": p_above.tolist(),
        "p_stow_reference": p_stow.tolist(),
        "max_position_error_norm": float(max_err_trace),
        "max_ik_joint_command_magnitude": float(max_cmd_mag),
        "min_palm_precontact_distance_reach_segment": float(min_precontact_reach),
        "min_palm_precontact_distance_any_phase": float(min_precontact_any),
        # Back-compat alias (no longer centroid distance).
        "min_palm_box_distance": float(min_precontact_reach),
        "final_palm_precontact_distance": float(final_precontact_dist),
        "success_precontact_distance_threshold_m": float(SUCCESS_PRECONTACT_DIST_M),
        "success": success,
        "fix_base_in_world": bool(fix_base_in_world),
    }

    if verbose:
        print(
            "----- reach_box summary -----\n"
            "Note: for box *contact* validation use run_g1_touch_box.py\n"
            f"success: {success}\n"
            f"min_palm_precontact_distance (reach window): "
            f"{out['min_palm_precontact_distance_reach_segment']:.5f}\n"
            f"min_palm_precontact_distance_any_phase: "
            f"{out['min_palm_precontact_distance_any_phase']:.5f}\n"
            f"final_palm_precontact_distance: {out['final_palm_precontact_distance']:.5f}\n"
            f"used_target_site: {used_target_site}\n"
            f"sim_time: {out['sim_time']:.5f}s  motion_complete: {mission_complete}"
        )

    return out


def main(argv: list[str] | None = None) -> None:
    print(
        "NOTE: run_g1_reach_box.py is a motion-only legacy demo. "
        "For solid pre-grasp contact validation, use run_g1_touch_box.py."
    )
    parser = argparse.ArgumentParser(description="G1 locked-base reach-to-static-box demo.")
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
    )
    parser.add_argument(
        "--max-joint-from-neutral",
        type=float,
        default=IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    )
    args = parser.parse_args(argv)

    try:
        run_g1_reach_box(
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
