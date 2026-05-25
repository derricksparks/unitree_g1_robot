#!/usr/bin/env python3
"""
Locked-base G1 palm touch/contact demo.

IK drives :obj:`right_palm_site` (visual only — no collisions) toward a **pre-grasp side-contact**
point from :mod:`perception.g1_box_perception` (simulated RGB-D truth: box faces, grasp height).
Penetration checks use :func:`g1_precontact.penetration_depth_m` (nominal plate outer face in +x and
a y/z corridor so overhead approach does not accumulate false ``max_penetration_depth``).

Primary contact-solid demo before grasping: prefer ``run_g1_touch_box.py`` over
``run_g1_reach_box.py`` (motion-only IK along the same precontact trajectory).

Run with the project interpreter (``python`` may be absent from :envvar:`PATH` in minimal shells)::

    ./.venv/bin/python simulation/mujoco/g1/run_g1_touch_box.py
    ./.venv/bin/python simulation/mujoco/g1/run_g1_touch_box.py --headless --timeout 8
    ./.venv/bin/python simulation/mujoco/g1/run_g1_touch_box.py --dbg-markers  # visualize perception sites
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.g1_box_perception import G1BoxPerception  # noqa: E402

from g1_precontact import (  # noqa: E402
    MAX_PENETRATION_DEPTH_OK_M,
    PENETRATION_FACE_TOLERANCE,
    TOUCH_DISTANCE_M,
    palm_in_side_contact_corridor,
    palm_plate_outer_face_x_nominal,
    penetration_depth_m,
    penetration_warning as palm_penetration_warning,
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
BASE_G1_XML = _G1 / "assets" / "g1_position_actuated.xml"

PALM_SITE_NAME = "right_palm_site"
FALLBACK_TARGET_BODY_NAME = "right_wrist_yaw_link"
BOX_BODY_NAME = "reach_target_box"
BOX_GEOM_NAME = "box_geom"
LEGACY_BOX_GEOM_NAME = "reach_box_geom"

PALM_CONTACT_GEOM_NAME = "right_palm_contact_geom"

TOUCH_IK_KP_SCALE = 0.86

_TOUCH_METRIC_PHASES = frozenset(
    {
        "DESCEND_TO_TOUCH",
        "TOUCH_HOLD",
    }
)
_EARLY_PHASES = frozenset({"STOW", "APPROACH_ABOVE_BOX"})

RIGHT_TOUCH_GEOM_NAMES = (
    "right_hand_collision",
    "right_wrist_collision",
)


class TouchPhase(Enum):
    STOW = auto()
    APPROACH_ABOVE_BOX = auto()
    DESCEND_TO_TOUCH = auto()
    TOUCH_HOLD = auto()
    RETREAT = auto()
    DONE = auto()


def _geom_names(model: mujoco.MjModel, gids: set[int]) -> list[str]:
    names: list[str] = []
    for gid in sorted(gids):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        names.append(nm if nm else f"geom[{gid}]")
    return names


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_scene_model() -> mujoco.MjModel:
    with _chdir(SCENE_PATH.resolve().parent):
        return mujoco.MjModel.from_xml_path(SCENE_PATH.name)


def _point_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_id: int,
    fallback_body_id: int,
) -> np.ndarray:
    if site_id >= 0:
        return np.asarray(data.site_xpos[site_id, :3], dtype=float).copy()
    return np.asarray(data.xpos[fallback_body_id, :3], dtype=float).copy()


def _geom_xpos_world(data: mujoco.MjData, gid: int) -> np.ndarray:
    return np.asarray(data.geom_xpos[gid, :3], dtype=float).copy()


def _box_geom_id(model: mujoco.MjModel) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    if gid >= 0:
        return int(gid)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEGACY_BOX_GEOM_NAME)
    if gid < 0:
        raise RuntimeError(f"Neither {BOX_GEOM_NAME} nor {LEGACY_BOX_GEOM_NAME} found")
    return int(gid)


def _box_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    gid = _box_geom_id(model)
    return np.asarray(data.geom_xpos[gid, :3], dtype=float).copy()


def _ik_qvel_adrs(model: mujoco.MjModel) -> list[int]:
    out: list[int] = []
    for jn in IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        out.append(int(model.jnt_dofadr[jid]))
    return out


def _contact_count(data: mujoco.MjData, box_gid: int, touch_gids: set[int]) -> int:
    count = 0
    for cid in range(data.ncon):
        contact = data.contact[cid]
        if box_gid in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == box_gid else contact.geom1
            if other in touch_gids:
                count += 1
    return count


def _hand_fallback_touch_gids(model: mujoco.MjModel) -> set[int]:
    out: set[int] = set()
    for name in RIGHT_TOUCH_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            out.add(int(gid))
    return out


def _touch_contact_gids(model: mujoco.MjModel) -> tuple[set[int], str]:
    palm_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, PALM_CONTACT_GEOM_NAME
    )
    if palm_gid >= 0:
        return {int(palm_gid)}, "palm_contact_geom"
    return _hand_fallback_touch_gids(model), "hand_geoms_fallback"


def _phase_target(
    sim_t: float,
    p_stow: np.ndarray,
    p_above: np.ndarray,
    p_touch: np.ndarray,
    *,
    timeout: float,
) -> tuple[np.ndarray, TouchPhase]:
    duration = max(float(timeout), 1e-6)
    stow_t = 0.12 * duration
    approach_t = 0.36 * duration
    descend_t = 0.58 * duration
    hold_t = 0.72 * duration
    retreat_t = 0.94 * duration

    if sim_t < stow_t:
        return p_stow.copy(), TouchPhase.STOW
    if sim_t < approach_t:
        u = _smoothstep01((sim_t - stow_t) / (approach_t - stow_t))
        return (1.0 - u) * p_stow + u * p_above, TouchPhase.APPROACH_ABOVE_BOX
    if sim_t < descend_t:
        u = _smoothstep01((sim_t - approach_t) / (descend_t - approach_t))
        return (1.0 - u) * p_above + u * p_touch, TouchPhase.DESCEND_TO_TOUCH
    if sim_t < hold_t:
        return p_touch.copy(), TouchPhase.TOUCH_HOLD
    if sim_t < retreat_t:
        u = _smoothstep01((sim_t - hold_t) / (retreat_t - hold_t))
        return (1.0 - u) * p_touch + u * p_above, TouchPhase.RETREAT
    return p_stow.copy(), TouchPhase.DONE


def run_g1_touch_box(
    *,
    headless: bool = False,
    timeout: float = 8.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    dbg_markers: bool = False,
) -> dict[str, Any]:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing scene file: {SCENE_PATH}")
    if not BASE_G1_XML.is_file():
        raise FileNotFoundError(f"Missing packaged model: {BASE_G1_XML}")

    model = _load_scene_model()
    if model.nu != 31:
        raise RuntimeError(f"Expected nu=31 (29 hinges + proxy finger motors), got {model.nu}")

    hinge_names = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            hinge_names.add(name)
    if set(NEUTRAL_POSTURE.keys()) != hinge_names:
        raise ValueError("NEUTRAL_POSTURE must list all hinges")

    _apply_kp_scale_for_joint_subset(model, set(IK_JOINT_NAMES), TOUCH_IK_KP_SCALE)
    _apply_kp_scale_for_joint_subset(
        model, {"right_wrist_roll_joint", "right_wrist_yaw_joint"}, TOUCH_IK_KP_SCALE
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
    mujoco.mj_forward(model, data)

    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FALLBACK_TARGET_BODY_NAME)
    if wrist_bid < 0:
        raise RuntimeError(f"Body {FALLBACK_TARGET_BODY_NAME} not found")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME) < 0:
        raise RuntimeError(f"Body {BOX_BODY_NAME} not found")

    palm_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE_NAME))
    box_gid = _box_geom_id(model)
    touch_geom_set, contact_detection_mode = _touch_contact_gids(model)
    palm_contact_gid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PALM_CONTACT_GEOM_NAME)
    )
    used_contact_detection = bool(touch_geom_set) and box_gid >= 0

    box_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, box_gid) or BOX_GEOM_NAME
    touch_geom_names = _geom_names(model, touch_geom_set)

    if verbose:
        print(
            "Contact detection geoms: "
            f"box={box_geom_name!r} (gid={box_gid}, contype={int(model.geom_contype[box_gid])}, "
            f"conaffinity={int(model.geom_conaffinity[box_gid])}); "
            f"touch={touch_geom_names} (mode={contact_detection_mode!r})"
        )
        for gid in sorted(touch_geom_set):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom[{gid}]"
            print(
                f"  touch geom {nm!r}: contype={int(model.geom_contype[gid])}, "
                f"conaffinity={int(model.geom_conaffinity[gid])}"
            )

    det0 = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
    p_box = np.asarray(det0["box_center"], dtype=float).reshape(3,)
    box_near_face_x_ref = float(det0["near_face_x"])
    p_touch_ref_init = np.asarray(det0["palm_pregrasp_target"], dtype=float)
    if verbose:
        print(
            "G1BoxPerception (startup, simulated vision):\n"
            f"  box_center={det0['box_center'].tolist()}\n"
            f"  box_half_size={det0['box_half_size'].tolist()}\n"
            f"  near_face_x={det0['near_face_x']:.5f}\n"
            f"  bottom_z={det0['bottom_z']:.5f}\n"
            f"  grasp_height_z={det0['grasp_height_z']:.5f}\n"
            f"  palm_pregrasp_target={det0['palm_pregrasp_target'].tolist()}\n"
            f"  palm_approach_target={det0['palm_approach_target'].tolist()}"
        )

    p_stow = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)

    initial_contact_n = _contact_count(data, box_gid, touch_geom_set)
    initial_contact_before_approach = initial_contact_n > 0
    if initial_contact_before_approach and verbose:
        print(
            "WARNING: initial contact before approach "
            f"(contacts={initial_contact_n}, mode={contact_detection_mode})"
        )
    stow_contact_warned = initial_contact_before_approach

    min_palm_touch_target_distance = float("inf")
    min_palm_box_center_distance = float("inf")
    max_contact_count = 0
    max_penetration_depth = 0.0
    early_phase_mission_violation = False
    penetration_during_touch_hold = False
    last_print = -PRINT_INTERVAL
    final_touch_target_dist = float("inf")
    final_box_center_dist = float("inf")
    final_phase = TouchPhase.STOW
    final_ik_err = 0.0
    max_joint_magnitude = 0.0
    mission_complete = False
    last_phase = TouchPhase.STOW

    def full_posture_from_ik(q_work: np.ndarray) -> dict[str, float]:
        full = dict(NEUTRAL_POSTURE)
        for jn, qv in zip(IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        return full

    def step_frame(*, force_report: bool = False, silent: bool = False) -> None:
        nonlocal min_palm_touch_target_distance, min_palm_box_center_distance
        nonlocal max_contact_count, max_penetration_depth, last_print
        nonlocal early_phase_mission_violation, penetration_during_touch_hold
        nonlocal final_touch_target_dist, final_box_center_dist
        nonlocal final_phase, final_ik_err, max_joint_magnitude
        nonlocal stow_contact_warned, last_phase

        det = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
        p_touch_now = np.asarray(det["palm_pregrasp_target"], dtype=float)
        p_above_now = np.asarray(det["palm_approach_target"], dtype=float)

        target, phase = _phase_target(
            data.time, p_stow, p_above_now, p_touch_now, timeout=timeout
        )
        q_work, ik_err = solve_ik_q(
            model,
            fd,
            data,
            ik_qpos_adrs,
            q_low,
            q_high,
            wrist_bid,
            target,
            q_neutral_ik,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
            target_site_id=palm_site_id if palm_site_id >= 0 else -1,
        )
        for i, adr in enumerate(ik_qpos_adrs):
            data.qpos[adr] = float(q_work[i])
        for adr in ik_qvel_adrs:
            data.qvel[adr] = 0.0
        mujoco.mj_forward(model, data)
        if dbg_markers:
            det_vis = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
            G1BoxPerception.sync_debug_marker_sites(
                model, data, det_vis, box_body_name=BOX_BODY_NAME
            )
            mujoco.mj_forward(model, data)
        command_position_actuators(model, data, full_posture_from_ik(q_work), actuator_ids)
        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        mujoco.mj_forward(model, data)

        palm = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)
        box = _box_center(model, data)
        box_bottom_z = float(det["bottom_z"])
        grasp_z = float(det["grasp_height_z"])
        near_face_x = float(det["near_face_x"])
        ly = float(det["left_edge_y"])
        ry = float(det["right_edge_y"])
        bz = float(det["bottom_z"])
        tz = float(det["top_z"])
        touch_target = p_touch_now
        touch_target_dist = float(np.linalg.norm(palm - touch_target))
        box_center_dist = float(np.linalg.norm(palm - box))
        contact_count = _contact_count(data, box_gid, touch_geom_set)

        if palm_contact_gid >= 0:
            pgeom = _geom_xpos_world(data, palm_contact_gid)
            palm_plate_outer_x = palm_plate_outer_face_x_nominal(float(pgeom[0]))
            depth = penetration_depth_m(
                model,
                data,
                palm_geom_id=palm_contact_gid,
                near_face_x=near_face_x,
                left_edge_y=ly,
                right_edge_y=ry,
                bottom_z=bz,
                top_z=tz,
            )
            pen_warn = palm_penetration_warning(
                model,
                data,
                palm_geom_id=palm_contact_gid,
                near_face_x=near_face_x,
                left_edge_y=ly,
                right_edge_y=ry,
                bottom_z=bz,
                top_z=tz,
            )
        else:
            pgeom = palm
            palm_plate_outer_x = palm_plate_outer_face_x_nominal(float(palm[0]))
            palm_in_corr = palm_in_side_contact_corridor(
                palm,
                left_edge_y=ly,
                right_edge_y=ry,
                bottom_z=bz,
                top_z=tz,
            )
            depth = (
                max(0.0, float(palm_plate_outer_x) - float(near_face_x))
                if palm_in_corr
                else 0.0
            )
            pen_warn = palm_in_corr and (
                float(palm_plate_outer_x) > float(near_face_x) + PENETRATION_FACE_TOLERANCE
            )

        joint_mag = max(
            abs(float(q_work[i]) - float(NEUTRAL_POSTURE[IK_JOINT_NAMES[i]]))
            for i in range(len(IK_JOINT_NAMES))
        )

        phase_name = phase.name
        last_phase = phase
        if (
            phase_name == "STOW"
            and contact_count > 0
            and verbose
            and not stow_contact_warned
        ):
            print(
                "WARNING: initial contact before approach "
                f"(contacts={contact_count}, mode={contact_detection_mode})"
            )
            stow_contact_warned = True

        if phase_name in _EARLY_PHASES and contact_count > 0:
            early_phase_mission_violation = True

        if phase_name in _TOUCH_METRIC_PHASES:
            min_palm_touch_target_distance = min(
                min_palm_touch_target_distance, touch_target_dist
            )
            min_palm_box_center_distance = min(min_palm_box_center_distance, box_center_dist)
            max_contact_count = max(max_contact_count, contact_count)
            max_penetration_depth = max(max_penetration_depth, depth)

        if phase_name == "TOUCH_HOLD" and depth > MAX_PENETRATION_DEPTH_OK_M:
            penetration_during_touch_hold = True

        max_joint_magnitude = max(max_joint_magnitude, joint_mag)
        touch_ok_mission = phase_name in _TOUCH_METRIC_PHASES and (
            touch_target_dist < TOUCH_DISTANCE_M or contact_count > 0
        ) and not (phase_name == "TOUCH_HOLD" and depth > MAX_PENETRATION_DEPTH_OK_M)
        final_touch_target_dist = touch_target_dist
        final_box_center_dist = box_center_dist
        final_phase = phase
        final_ik_err = ik_err

        if verbose and not silent and (
            force_report or data.time - last_print >= PRINT_INTERVAL
        ):
            print(
                f"phase={phase.name}  palm_xyz={palm.tolist()}  "
                f"palm_contact_geom_xyz={pgeom.tolist()}  "
                f"palm_plate_outer_face_x={palm_plate_outer_x:.5f}  "
                f"near_face_x={near_face_x:.5f}  "
                f"penetration_depth={depth:.5f}  "
                f"grasp_height_z (perception)={grasp_z:.5f}  "
                f"palm_touch_target={touch_target.tolist()}  "
                f"palm_to_touch_target_distance={touch_target_dist:.5f}  "
                f"palm_to_box_center_distance={box_center_dist:.5f}  "
                f"contact_count={contact_count}  "
                f"penetration_warning={pen_warn}  "
                f"touch_ok_mission={touch_ok_mission}  ik_error={ik_err:.5f}  "
                f"max_joint_magnitude={joint_mag:.5f}"
            )
            last_print = data.time

    def run_headless_loop() -> None:
        nonlocal mission_complete
        while data.time < timeout:
            step_frame()
            if last_phase == TouchPhase.DONE:
                mission_complete = True
                return

    def run_viewer_loop() -> None:
        nonlocal mission_complete
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_t0 = time.time()
                step_frame(silent=mission_complete)
                if last_phase == TouchPhase.DONE:
                    mission_complete = True
                viewer.sync()
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - loop_t0)))

    if headless:
        run_headless_loop()
    else:
        run_viewer_loop()

    mujoco.mj_forward(model, data)
    det_end = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
    palm_end = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)
    box_end = _box_center(model, data)
    nf_end = float(det_end["near_face_x"])
    ly_end = float(det_end["left_edge_y"])
    ry_end = float(det_end["right_edge_y"])
    bz_end = float(det_end["bottom_z"])
    tz_end = float(det_end["top_z"])
    if palm_contact_gid >= 0:
        final_penetration_depth = penetration_depth_m(
            model,
            data,
            palm_geom_id=palm_contact_gid,
            near_face_x=nf_end,
            left_edge_y=ly_end,
            right_edge_y=ry_end,
            bottom_z=bz_end,
            top_z=tz_end,
        )
        final_penetration_warning = palm_penetration_warning(
            model,
            data,
            palm_geom_id=palm_contact_gid,
            near_face_x=nf_end,
            left_edge_y=ly_end,
            right_edge_y=ry_end,
            bottom_z=bz_end,
            top_z=tz_end,
        )
    else:
        palm_ox_end = palm_plate_outer_face_x_nominal(float(palm_end[0]))
        cy_end = palm_in_side_contact_corridor(
            palm_end,
            left_edge_y=ly_end,
            right_edge_y=ry_end,
            bottom_z=bz_end,
            top_z=tz_end,
        )
        final_penetration_depth = (
            max(0.0, palm_ox_end - nf_end) if cy_end else 0.0
        )
        final_penetration_warning = bool(
            cy_end and palm_ox_end > nf_end + PENETRATION_FACE_TOLERANCE
        )
    tt_end = np.asarray(det_end["palm_pregrasp_target"], dtype=float)
    final_palm_touch_target_distance = float(np.linalg.norm(palm_end - tt_end))
    final_palm_box_center_distance = float(np.linalg.norm(palm_end - box_end))

    reach_ok = np.isfinite(min_palm_touch_target_distance) and (
        min_palm_touch_target_distance < TOUCH_DISTANCE_M or max_contact_count > 0
    )
    penetration_ok = max_penetration_depth <= MAX_PENETRATION_DEPTH_OK_M
    success = bool(
        (not initial_contact_before_approach)
        and reach_ok
        and penetration_ok
        and (not early_phase_mission_violation)
        and (not penetration_during_touch_hold)
    )

    out: dict[str, Any] = {
        "success": success,
        "min_palm_touch_target_distance": float(min_palm_touch_target_distance),
        "min_palm_box_center_distance": float(min_palm_box_center_distance),
        "max_contact_count": int(max_contact_count),
        "max_penetration_depth": float(max_penetration_depth),
        "used_contact_detection": used_contact_detection,
        "contact_detection_mode": contact_detection_mode,
        "contact_touch_geom_names": touch_geom_names,
        "contact_box_geom_name": box_geom_name,
        "initial_contact_before_approach": initial_contact_before_approach,
        "early_phase_mission_violation": early_phase_mission_violation,
        "penetration_during_touch_hold": penetration_during_touch_hold,
        "final_palm_touch_target_distance": float(final_palm_touch_target_distance),
        "final_palm_box_center_distance": float(final_palm_box_center_distance),
        "final_penetration_depth": float(final_penetration_depth),
        "final_penetration_warning": final_penetration_warning,
        "phase": final_phase.name,
        "ik_error": float(final_ik_err),
        "max_joint_magnitude": float(max_joint_magnitude),
        "model_path": str(SCENE_PATH.resolve()),
        "box_center": p_box.tolist(),
        "box_near_face_x": float(box_near_face_x_ref),
        "palm_touch_target_ref": p_touch_ref_init.tolist(),
        "touch_distance_threshold_m": TOUCH_DISTANCE_M,
        "target_is_side_pregrasp": True,
        "palm_target_not_box_center": True,
        "box_center_clearance_at_touch": float(min_palm_box_center_distance),
        "max_penetration_depth_ok_m": MAX_PENETRATION_DEPTH_OK_M,
        "penetration_face_tolerance": PENETRATION_FACE_TOLERANCE,
        "sim_time": float(data.time),
        "motion_complete": mission_complete,
    }

    if verbose:
        print(
            "----- touch_box summary -----\n"
            f"initial_contact_before_approach: {out['initial_contact_before_approach']}\n"
            f"min_palm_touch_target_distance (descend+hold): "
            f"{out['min_palm_touch_target_distance']:.5f}\n"
            f"target_is_side_pregrasp: {out['target_is_side_pregrasp']}\n"
            f"palm_target_not_box_center: {out['palm_target_not_box_center']}\n"
            f"min_palm_box_center_distance (descend+hold): "
            f"{out['min_palm_box_center_distance']:.5f}\n"
            f"box_center_clearance_at_touch: {out['box_center_clearance_at_touch']:.5f}\n"
            f"max_contact_count: {out['max_contact_count']}\n"
            f"max_penetration_depth: {out['max_penetration_depth']:.5f}\n"
            f"early_phase_mission_violation: {out['early_phase_mission_violation']}\n"
            f"penetration_during_touch_hold: {out['penetration_during_touch_hold']}\n"
            f"success: {out['success']}\n"
            f"contact_detection_mode: {contact_detection_mode}\n"
            f"used_contact_detection: {out['used_contact_detection']}\n"
            f"final_palm_touch_target_distance: {out['final_palm_touch_target_distance']:.5f}\n"
            f"final_palm_box_center_distance: {out['final_palm_box_center_distance']:.5f}\n"
            f"final_penetration_depth: {out['final_penetration_depth']:.5f}\n"
            f"sim_time: {out['sim_time']:.5f}s  motion_complete: {mission_complete}"
        )

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="G1 locked-base palm touch/contact demo.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument(
        "--pelvis-z",
        type=float,
        default=DEFAULT_PELVIS_Z,
        help=f"Locked pelvis height (m), default {DEFAULT_PELVIS_Z}",
    )
    parser.add_argument("--posture-gain", type=float, default=IK_POSTURE_GAIN)
    parser.add_argument(
        "--max-joint-from-neutral",
        type=float,
        default=IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    )
    parser.add_argument(
        "--dbg-markers",
        action="store_true",
        help="Show perception debug sites (see g1_reach_box_scene.xml).",
    )
    args = parser.parse_args(argv)

    try:
        run_g1_touch_box(
            headless=args.headless,
            timeout=args.timeout,
            initial_pelvis_z=args.pelvis_z,
            verbose=True,
            posture_gain=args.posture_gain,
            max_joint_from_neutral=args.max_joint_from_neutral,
            dbg_markers=args.dbg_markers,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
