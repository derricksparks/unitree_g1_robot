#!/usr/bin/env python3
"""
Direct dual-arm reach-and-contact validation (shoulder-fixed IK).

- Dual arms reach fixed blue contact sites on the box side faces (sites never snap to fingers).
- Index/middle close toward fixed blue contact sites on the box; each digit stops on site
  touch, MuJoCo box contact, or within 3 mm.
- After bilateral grasp, arms lift the box with elbows driven toward 90° and bounded
  vertical assist (unless ``--no-lift`` / ``--no-assist``).
- Use ``--scripted-lift`` only for visualization; default uses physical co-lift.

Run::

    ./.venv/bin/python simulation/mujoco/g1/run_g1_direct_side_reach.py --headless --timeout 10
    ./.venv/bin/python simulation/mujoco/g1/run_g1_direct_side_reach.py --scripted-lift
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.g1_box_perception import (  # noqa: E402
    G1BoxPerception,
    SITE_BOX_LEFT_CONTACT,
    SITE_BOX_LEFT_MIDDLE_CONTACT,
    SITE_BOX_RIGHT_CONTACT,
    SITE_BOX_RIGHT_MIDDLE_CONTACT,
)

from g1_arm_cartesian_reach import (  # noqa: E402
    RIGHT_ARM_SHOULDER_CHAIN,
    RIGHT_WRIST_FREEZE_JOINTS,
    RIGHT_WRIST_PITCH_JOINT,
    add_joints_to_frozen,
    apply_frozen_qpos,
    build_chain_metadata,
    frozen_qpos_snapshot,
    palm_site_xyz,
    remove_joints_from_frozen,
    solve_position_only,
)
from g1_dex3_finger_control import (  # noqa: E402
    Dex3FingerController,
    dex3_digit_joint_names,
    merge_dex3_neutral_into,
)
from g1_dex_hand_contact import (  # noqa: E402
    dex3_digit_box_distances,
    dex3_digit_contact_point_distances,
    dex3_digit_distal_geom_ids_by_role,
    dex3_digit_side_face_gaps,
    dex3_digit_target_distances,
    digit_contact_counts_by_role,
    enumerate_hand_contact_geoms,
    mujoco_site_world_xyz,
    palm_contact_count,
    nudge_hinge_joint_toward_site,
    snap_grasp_digits_to_world_point,
    tune_digit_progressive_blend,
)
from g1_precontact import PALM_PLATE_HALF_WIDTH_Y  # noqa: E402
from run_g1_dual_arm_box import _load_dual_scene_model  # noqa: E402
from run_g1_grasp_box import _smoothstep01  # noqa: E402
from run_g1_grasp_box import (  # noqa: E402
    ASSIST_KP_POS,
    ASSIST_KD_Z,
    ASSIST_LIFT_Z_SCALE,
    BOX_SLIDE_JOINT_NAME,
    LIFT_ASSIST_TARGET_DZ_M,
    LIFT_PALM_DELTA_Z_M,
    LIFT_SUCCESS_DELTA_Z_M,
    MAX_ASSIST_FORCE_NEWTONS,
)
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
from run_g1_right_arm_ik_demo import (  # noqa: E402
    IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    IK_POSTURE_GAIN,
    _apply_kp_scale_for_joint_subset,
)

PALM_SITE = "right_palm_site"
LEFT_PALM_SITE = "left_palm_site"
WRIST_BODY = "right_wrist_yaw_link"
LEFT_WRIST_BODY = "left_wrist_yaw_link"
BOX_GEOM = "box_geom"
BOX_BODY = "reach_target_box"

PHYSICS_TIMESTEP_S = 0.004
IK_INNER_ITERS = 12
MAX_ARM_STEP_RAD = 0.028
PALM_ON_TARGET_M = 0.018
# Freeze digits only after MuJoCo contact or non-positive convex distance (intersection).
DIGIT_BOX_INTERSECT_EPS_M = 0.002
# MuJoCo contact or convex digit→box distance at/below this counts as real touch.
DIGIT_BOX_NEAR_CONTACT_M = 0.003
# Digit distal geom within this of the fixed blue contact site counts as on the contact point.
DIGIT_CONTACT_POINT_NEAR_M = 0.004
STABLE_DUAL_CONTACT_MIN_S = 0.10
MAX_SUCCESS_BOX_PENETRATION_M = 0.010
# Diagnostic only: 3D distance to fixed blue sites on the box face (not used for success).
DIGIT_CONTACT_SITE_REPORT_M = 0.09
DIGIT_INTERSECT_STABLE_FRAMES = 4
# Finger close: small joint steps + per-frame blend advance until contact / 3 mm.
FINGER_CLOSE_BLEND_MAX = 0.95
FINGER_CLOSE_U_STEP = 0.016
FINGER_REACH_MAX_DELTA_RAD = 0.006
FINGER_CLOSE_INNER_STEPS = 8
FINGER_CLOSE_SETTLE_STEPS = 36
POST_GRASP_LIFT_DURATION_S = 3.0
LIFT_ARM_CONTACT_POINT_RELAX_M = 0.055
FINGER_CLOSE_START_GAP_M = 0.058
FINGER_CLOSE_START_BOX_DIST_M = 0.018
DIGIT_SAFE_FREEZE_MAX_DIST_M = 0.004
FINGER_FACE_NEAR_M = 0.014
FINGER_FACE_TOUCH_M = 0.006
DIGIT_MIN_OUTSIDE_BOX_M = 0.002
DIGIT_SAFE_DIST_MIN_M = 0.000
DIGIT_ROLLBACK_OPEN_STEP_RAD = 0.0035
DIGIT_ROLLBACK_MAX_STEPS = 10
# Shoulder/elbow within this margin of a joint limit → treat as saturated (stop arm IK forcing).
JOINT_LIMIT_MARGIN_RAD = 0.028
PALM_MIN_OUTSIDE_FACE_M = 0.002
# During digit reach: only wrist pitch tracks vertical palm / site-z (no shoulder fighting).
PALM_FINGER_REACH_Z_BLEND_MAX = 0.85
MAX_WRIST_PITCH_STEP_RAD = 0.042
MAX_WRIST_YAW_STEP_RAD = 0.055
WRIST_PITCH_SITE_SEARCH_SAMPLES = 9
WRIST_YAW_SITE_SEARCH_SAMPLES = 13
WRIST_YAW_SITE_SEARCH_SPAN_RAD = 1.2
ELBOW_RIGHT_ANGLE_RAD = 0.5 * np.pi
LEFT_WRIST_FREEZE_JOINTS: tuple[str, ...] = (
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
)
BOX_LIFT_START_FRAC = 0.62
BOX_LIFT_END_FRAC = 0.88
BOX_LIFT_PHASE_START_FRAC = 0.62
BOX_LIFT_PHASE_END_FRAC = 1.0
LIFT_ARM_MIN_STABLE_S = 0.05
ELBOW_LIFT_BLEND_PER_FRAME = 0.14
ELBOW_LIFT_ANGLE_TOLERANCE_RAD = 0.06
DEFAULT_TIMEOUT_S = 18.0

RIGHT_SHOULDER_ELBOW_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
LEFT_ARM_SHOULDER_CHAIN: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_LATCH_EXCEPT_PITCH: tuple[str, ...] = (
    *RIGHT_SHOULDER_ELBOW_JOINTS,
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)
ACTIVE_ARM_JOINTS: tuple[str, ...] = (*RIGHT_ARM_SHOULDER_CHAIN, *LEFT_ARM_SHOULDER_CHAIN)

_WRIST_PITCH_I = RIGHT_ARM_SHOULDER_CHAIN.index(RIGHT_WRIST_PITCH_JOINT)
_WRIST_YAW_I = RIGHT_ARM_SHOULDER_CHAIN.index("right_wrist_yaw_joint")
_ARM_LATCH_INDICES = [
    i for i, jn in enumerate(RIGHT_ARM_SHOULDER_CHAIN) if jn in RIGHT_ARM_LATCH_EXCEPT_PITCH
]
RIGHT_GRASP_DIGITS = ("index", "middle")
LEFT_GRASP_DIGITS = ("index", "middle")


def _right_contact_site_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
) -> np.ndarray:
    """World xyz of the blue ``box_right_contact_site`` (live site, else perception)."""
    live = mujoco_site_world_xyz(model, data, SITE_BOX_RIGHT_CONTACT)
    if live is not None:
        return live
    return np.asarray(perception["viz_right_contact_world"], dtype=float).reshape(3)


def _right_digit_contact_site_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
    role: str,
) -> np.ndarray:
    """World xyz of the blue contact site for ``role`` (``index`` or ``middle``)."""
    if role == "middle":
        live = mujoco_site_world_xyz(model, data, SITE_BOX_RIGHT_MIDDLE_CONTACT)
        fallback = perception.get("viz_right_middle_contact_world")
    else:
        live = mujoco_site_world_xyz(model, data, SITE_BOX_RIGHT_CONTACT)
        fallback = perception.get("viz_right_contact_world")
    if live is not None:
        return live
    return np.asarray(fallback, dtype=float).reshape(3)


def _left_digit_contact_site_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
    role: str,
) -> np.ndarray:
    """World xyz of the left blue contact site for ``role``."""
    if role == "middle":
        live = mujoco_site_world_xyz(model, data, SITE_BOX_LEFT_MIDDLE_CONTACT)
        fallback = perception.get("viz_left_middle_contact_world")
    else:
        live = mujoco_site_world_xyz(model, data, SITE_BOX_LEFT_CONTACT)
        fallback = perception.get("viz_left_contact_world")
    if live is not None:
        return live
    return np.asarray(fallback, dtype=float).reshape(3)


def _digit_contact_site_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
    *,
    side: str,
    role: str,
) -> np.ndarray:
    if side == "right":
        return _right_digit_contact_site_world(model, data, perception, role)
    if side == "left":
        return _left_digit_contact_site_world(model, data, perception, role)
    raise ValueError(side)


def _digit_contact_site_distances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
    *,
    side: str,
    roles: tuple[str, ...],
    face_y: float,
) -> dict[str, float]:
    """Per-digit distance (m) from distal geoms to the blue box contact site."""
    out: dict[str, float] = {}
    for role in roles:
        site = _digit_contact_site_world(model, data, perception, side=side, role=role)
        out[role] = float(
            dex3_digit_target_distances(
                model,
                data,
                site,
                side=side,
                distal_only=True,
                face_y=float(face_y),
            ).get(role, float("inf"))
        )
    return out


def _digit_at_contact_point(site_distance_m: float) -> bool:
    return float(site_distance_m) <= float(DIGIT_CONTACT_POINT_NEAR_M)


def _digit_site_distances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
    *,
    face_y: float,
) -> dict[str, float]:
    """Per-digit 3D distance to its blue box contact site."""
    out: dict[str, float] = {}
    for role in RIGHT_GRASP_DIGITS:
        site = _right_digit_contact_site_world(model, data, perception, role)
        out[role] = float(
            dex3_digit_target_distances(
                model,
                data,
                site,
                side="right",
                distal_only=True,
            ).get(role, float("inf"))
        )
    return out


def _left_digit_site_distances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    perception: dict[str, Any],
) -> dict[str, float]:
    """Per-left-digit 3D distance to its blue box contact site."""
    out: dict[str, float] = {}
    for role in LEFT_GRASP_DIGITS:
        site = _left_digit_contact_site_world(model, data, perception, role)
        out[role] = float(
            dex3_digit_target_distances(
                model,
                data,
                site,
                side="left",
                distal_only=True,
            ).get(role, float("inf"))
        )
    return out


def _side_has_real_finger_contact(
    digit_dist: dict[str, float],
    digit_cont: dict[str, int],
    *,
    roles: tuple[str, ...],
) -> bool:
    """True when index or middle has MuJoCo box contact or is within near-contact distance."""
    for role in roles:
        if int(digit_cont.get(role, 0)) > 0:
            return True
        if float(digit_dist.get(role, float("inf"))) <= DIGIT_BOX_NEAR_CONTACT_M:
            return True
    return False


def _min_finger_box_distance(digit_dist: dict[str, float], roles: tuple[str, ...]) -> float:
    return min(float(digit_dist.get(r, float("inf"))) for r in roles)


def _evaluate_contact_validation_success(
    *,
    min_palm_dist: float,
    min_left_palm_dist: float,
    wrist_frozen: bool,
    left_wrist_frozen: bool,
    right_real_finger_contact: bool,
    left_real_finger_contact: bool,
    max_box_penetration_any_geom: float,
    max_stable_dual_contact_time_s: float,
    finger_grasp_hold: bool,
) -> tuple[bool, bool, str, str]:
    """Return (success, real_grasp_contact_success, success_reason, failure_reason)."""
    failures: list[str] = []
    if min_palm_dist > PALM_ON_TARGET_M * 1.35:
        failures.append("right_palm_not_on_target")
    if min_left_palm_dist > PALM_ON_TARGET_M * 1.35:
        failures.append("left_palm_not_on_target")
    if not wrist_frozen:
        failures.append("right_wrist_not_frozen")
    if not left_wrist_frozen:
        failures.append("left_wrist_not_frozen")
    if not right_real_finger_contact:
        failures.append("right_no_real_finger_contact")
    if not left_real_finger_contact:
        failures.append("left_no_real_finger_contact")
    if max_box_penetration_any_geom > MAX_SUCCESS_BOX_PENETRATION_M:
        failures.append("box_penetration_exceeded")
    if max_stable_dual_contact_time_s <= STABLE_DUAL_CONTACT_MIN_S:
        failures.append("dual_contact_not_stable")

    real_grasp = bool(
        right_real_finger_contact
        and left_real_finger_contact
        and max_box_penetration_any_geom <= MAX_SUCCESS_BOX_PENETRATION_M
        and max_stable_dual_contact_time_s > STABLE_DUAL_CONTACT_MIN_S
    )
    if not finger_grasp_hold:
        ok = bool(
            min_palm_dist <= PALM_ON_TARGET_M * 1.35
            and min_left_palm_dist <= PALM_ON_TARGET_M * 1.35
            and wrist_frozen
            and left_wrist_frozen
        )
        if ok:
            return True, False, "palms_reached_no_finger_hold", ""
        return False, False, "", ";".join(failures) or "palms_not_reached"

    ok = len(failures) == 0
    if ok:
        return True, real_grasp, "real_dual_finger_contact_validated", ""
    return False, real_grasp, "", ";".join(failures)


def _box_roll_pitch_deg(model: mujoco.MjModel, data: mujoco.MjData, box_bid: int) -> tuple[float, float]:
    """Approximate world roll/pitch of the box body in degrees."""
    if box_bid < 0:
        return 0.0, 0.0
    r = np.asarray(data.xmat[int(box_bid)], dtype=float).reshape(3, 3)
    roll = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
    pitch = float(np.degrees(np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))))
    return roll, pitch


def _rotation_error_deg(a_xmat: np.ndarray, b_xmat: np.ndarray) -> float:
    """Geodesic angle between two 3x3 rotation matrices, in degrees."""
    a = np.asarray(a_xmat, dtype=float).reshape(3, 3)
    b = np.asarray(b_xmat, dtype=float).reshape(3, 3)
    c = a.T @ b
    cosang = float(np.clip((np.trace(c) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _contact_count_for_geoms(
    data: mujoco.MjData,
    *,
    box_gid: int,
    geom_ids: set[int],
) -> int:
    """Count contacts between the box geom and a set of geoms."""
    if box_gid < 0:
        return 0
    n = 0
    for cid in range(data.ncon):
        c = data.contact[cid]
        if int(box_gid) not in (int(c.geom1), int(c.geom2)):
            continue
        other = int(c.geom2 if int(c.geom1) == int(box_gid) else c.geom1)
        if other in geom_ids:
            n += 1
    return int(n)


def _side_fingertip_contact_count(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_gid: int,
    side: str,
) -> int:
    """Index/middle distal contact count for one hand."""
    gids_by_role = dex3_digit_distal_geom_ids_by_role(model, side=side)
    gids = set(gids_by_role.get("index", set())) | set(gids_by_role.get("middle", set()))
    return _contact_count_for_geoms(data, box_gid=box_gid, geom_ids=gids)


def _refine_wrist_pitch_yaw_for_digit_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cmd: np.ndarray,
    *,
    ik_adrs: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    frozen: dict[str, float],
    det: dict[str, Any],
    face_y: float,
    latch_cmd: np.ndarray,
) -> None:
    """Search wrist pitch/yaw (shoulder/elbow/roll latched) to minimize digit→site distance."""
    pitch_i = _WRIST_PITCH_I
    yaw_i = _WRIST_YAW_I
    lo = float(q_low[pitch_i])
    hi = float(q_high[pitch_i])
    yaw_lo = float(q_low[yaw_i])
    yaw_hi = float(q_high[yaw_i])
    yaw_center = float(cmd[yaw_i])
    yaw_span = float(WRIST_YAW_SITE_SEARCH_SPAN_RAD)
    base = latch_cmd.copy()
    best_pitch = float(cmd[pitch_i])
    best_yaw = float(cmd[yaw_i])
    best_cost = float("inf")
    yaw_samples = np.linspace(
        max(yaw_lo, yaw_center - yaw_span),
        min(yaw_hi, yaw_center + yaw_span),
        WRIST_YAW_SITE_SEARCH_SAMPLES,
    )
    for pitch_trial in np.linspace(lo, hi, WRIST_PITCH_SITE_SEARCH_SAMPLES):
        for yaw_trial in yaw_samples:
            trial_cmd = base.copy()
            trial_cmd[pitch_i] = float(pitch_trial)
            trial_cmd[yaw_i] = float(yaw_trial)
            for i, adr in enumerate(ik_adrs):
                data.qpos[int(adr)] = float(trial_cmd[i])
            apply_frozen_qpos(data, frozen)
            mujoco.mj_forward(model, data)
            gaps = dex3_digit_side_face_gaps(
                model,
                data,
                side="right",
                face_y=float(face_y),
                approach_from_negative_y=True,
            )
            if any(float(gaps.get(r, float("inf"))) < -0.001 for r in RIGHT_GRASP_DIGITS):
                continue
            site_d = _digit_site_distances(model, data, det, face_y=float(face_y))
            cost = max(float(site_d.get(r, float("inf"))) for r in RIGHT_GRASP_DIGITS)
            if cost < best_cost:
                best_cost = cost
                best_pitch = float(pitch_trial)
                best_yaw = float(yaw_trial)
    cmd[:] = base
    cmd[pitch_i] = best_pitch
    cmd[yaw_i] = best_yaw
    for i, adr in enumerate(ik_adrs):
        data.qpos[int(adr)] = float(cmd[i])
    apply_frozen_qpos(data, frozen)
    mujoco.mj_forward(model, data)


def _snap_digit_to_contact_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    det: dict[str, Any],
    *,
    side: str,
    role: str,
) -> None:
    """Fine-tune one digit toward its blue box contact site."""
    face_y = float(det["left_edge_y"] if side == "right" else det["right_edge_y"])
    site = _digit_contact_site_world(model, data, det, side=side, role=role)
    jnames = tuple(dex3_digit_joint_names(side, role))
    if not jnames:
        return
    snap_grasp_digits_to_world_point(
        model,
        data,
        side=side,
        roles=(role,),
        target_xyz=site,
        joint_names_by_role={role: jnames},
        max_iters=28,
        step_rad=0.012,
        face_y=face_y,
    )
    site_d = float(
        dex3_digit_target_distances(
            model, data, site, side=side, distal_only=True, face_y=face_y
        ).get(role, float("inf"))
    )
    if site_d > DIGIT_CONTACT_POINT_NEAR_M:
        for jn in jnames:
            nudge_hinge_joint_toward_site(
                model,
                data,
                joint_name=jn,
                side=side,
                role=role,
                target_xyz=site,
                step_rad=0.014,
                trials=14,
            )


def _apply_elbow_right_angle_pose(
    cmd: np.ndarray,
    left_cmd: np.ndarray,
    *,
    q_low: np.ndarray,
    q_high: np.ndarray,
    left_q_low: np.ndarray,
    left_q_high: np.ndarray,
    blend: float = ELBOW_LIFT_BLEND_PER_FRAME,
) -> None:
    """Blend both elbows toward 90° (π/2 rad) for lift."""
    right_elbow_i = RIGHT_ARM_SHOULDER_CHAIN.index("right_elbow_joint")
    left_elbow_i = LEFT_ARM_SHOULDER_CHAIN.index("left_elbow_joint")
    b = float(np.clip(blend, 0.0, 1.0))
    cmd[right_elbow_i] = float(
        np.clip(
            (1.0 - b) * float(cmd[right_elbow_i]) + b * ELBOW_RIGHT_ANGLE_RAD,
            float(q_low[right_elbow_i]),
            float(q_high[right_elbow_i]),
        )
    )
    left_cmd[left_elbow_i] = float(
        np.clip(
            (1.0 - b) * float(left_cmd[left_elbow_i]) + b * ELBOW_RIGHT_ANGLE_RAD,
            float(left_q_low[left_elbow_i]),
            float(left_q_high[left_elbow_i]),
        )
    )


def _elbows_near_right_angle(
    cmd: np.ndarray,
    left_cmd: np.ndarray,
    *,
    tol_rad: float = ELBOW_LIFT_ANGLE_TOLERANCE_RAD,
) -> bool:
    right_elbow_i = RIGHT_ARM_SHOULDER_CHAIN.index("right_elbow_joint")
    left_elbow_i = LEFT_ARM_SHOULDER_CHAIN.index("left_elbow_joint")
    return bool(
        abs(float(cmd[right_elbow_i]) - ELBOW_RIGHT_ANGLE_RAD) <= tol_rad
        and abs(float(left_cmd[left_elbow_i]) - ELBOW_RIGHT_ANGLE_RAD) <= tol_rad
    )


def _shoulder_elbow_saturated(
    cmd: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    *,
    margin_rad: float = JOINT_LIMIT_MARGIN_RAD,
) -> bool:
    """True when any shoulder/elbow hinge is within ``margin_rad`` of its limit."""
    for jn in RIGHT_SHOULDER_ELBOW_JOINTS:
        i = RIGHT_ARM_SHOULDER_CHAIN.index(jn)
        q = float(cmd[i])
        if q <= float(q_low[i]) + margin_rad or q >= float(q_high[i]) - margin_rad:
            return True
    return False


def _seek_digit_close_blend_u(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger_ctrl: Dex3FingerController,
    hinge_addrs: dict[str, dict[str, int]],
    det: dict[str, Any],
    box_gid: int,
    *,
    side: str,
    role: str,
    current_u: float,
    max_u: float = FINGER_CLOSE_BLEND_MAX,
    n_samples: int = 14,
) -> float:
    """Find blend ``u`` that places the digit on its blue box contact site."""
    del n_samples  # tune_digit_progressive_blend uses fixed samples
    face_y = float(det["left_edge_y"] if side == "right" else det["right_edge_y"])
    site = _digit_contact_site_world(model, data, det, side=side, role=role)
    hinge_map = {
        jn: int(hinge_addrs[jn]["qpos_adr"])
        for jn in dex3_digit_joint_names(side, role)
        if jn in hinge_addrs
    }
    u_site, _d_site = tune_digit_progressive_blend(
        model,
        data,
        finger_ctrl,
        side=side,
        role=role,
        box_gid=int(box_gid),
        hinge_qpos_adrs=hinge_map,
        contact_site_xyz=site,
        face_y=face_y,
    )
    return float(np.clip(max(float(current_u), u_site), 0.0, max_u))


def _digit_box_penetration_depth_m(convex_distance_m: float) -> float:
    """Positive penetration depth from convex digit→box distance (negative = inside)."""
    return float(max(0.0, -float(convex_distance_m)))


def _snapshot_digit_qpos(
    side: str,
    role: str,
    data: mujoco.MjData,
    hinge_addrs: dict[str, dict[str, int]],
) -> dict[str, float]:
    return {
        jn: float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])])
        for jn in dex3_digit_joint_names(side, role)
    }


def _restore_digit_qpos(
    snap: dict[str, float],
    *,
    data: mujoco.MjData,
    hinge_addrs: dict[str, dict[str, int]],
    finger_ctrl: Dex3FingerController,
) -> None:
    for jn, qv in snap.items():
        adr = int(hinge_addrs[jn]["qpos_adr"])
        data.qpos[adr] = float(qv)
        finger_ctrl._state[jn] = float(qv)


def _open_digit_slightly(
    side: str,
    role: str,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger_ctrl: Dex3FingerController,
    hinge_addrs: dict[str, dict[str, int]],
    step_rad: float = DIGIT_ROLLBACK_OPEN_STEP_RAD,
) -> None:
    """Ease one digit toward open_hand by a small joint step."""
    open_t = finger_ctrl.targets_for_mode("open_hand")
    for jn in dex3_digit_joint_names(side, role):
        adr = int(hinge_addrs[jn]["qpos_adr"])
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        cur = float(data.qpos[adr])
        tgt = float(open_t.get(jn, cur))
        data.qpos[adr] = float(np.clip(cur + np.clip(tgt - cur, -step_rad, step_rad), lo, hi))
        finger_ctrl._state[jn] = float(data.qpos[adr])


def _digit_in_safe_contact_band(convex_distance_m: float) -> bool:
    d = float(convex_distance_m)
    return DIGIT_MIN_OUTSIDE_BOX_M <= d <= DIGIT_BOX_NEAR_CONTACT_M


def _digit_aabb_penetration_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    det: dict[str, Any],
    *,
    side: str,
    role: str,
) -> float:
    """Max axis-aligned box penetration for one digit's distal geoms."""
    from g1_precontact import geom_max_penetration_into_world_axis_aligned_box

    gids = dex3_digit_distal_geom_ids_by_role(model, side=side).get(role, set())
    if not gids:
        return 0.0
    pen = 0.0
    xmin = float(det["box_x_min"])
    xmax = float(det["box_x_max"])
    ymin = float(det["box_y_min"])
    ymax = float(det["box_y_max"])
    bz = float(det["bottom_z"])
    tz = float(det["top_z"])
    for gid in gids:
        pen = max(
            pen,
            float(
                geom_max_penetration_into_world_axis_aligned_box(
                    model, data, int(gid), xmin, xmax, ymin, ymax, bz, tz
                )
            ),
        )
    return float(pen)


def _guard_digits_after_close(
    *,
    side: str,
    grasp_digits: tuple[str, ...],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    det: dict[str, Any],
    finger_ctrl: Dex3FingerController,
    hinge_addrs: dict[str, dict[str, int]],
    frozen: dict[int, float],
    box_gid: int,
    digit_grasp_locked: dict[str, bool],
    digit_safe_qpos: dict[str, dict[str, float]],
    digit_penetration_max: dict[str, float],
    digit_rollback_count: list[int],
    digit_safe_freeze_count: list[int],
    max_digit_penetration_m: list[float],
    close_blend_progress: dict[str, float],
) -> None:
    """Rollback penetrating digits; freeze only at last safe non-penetrating pose."""
    digit_dist = dex3_digit_box_distances(model, data, box_gid=box_gid, side=side)
    digit_cont = digit_contact_counts_by_role(model, data, box_gid=box_gid, side=side)

    for role in grasp_digits:
        metric_key = f"{side}_{role}"
        d_m = float(digit_dist.get(role, float("inf")))
        pen_m = _digit_box_penetration_depth_m(d_m)
        aabb_pen_m = _digit_aabb_penetration_m(model, data, det, side=side, role=role)
        digit_penetration_max[metric_key] = max(
            float(digit_penetration_max.get(metric_key, 0.0)), pen_m, aabb_pen_m
        )
        max_digit_penetration_m[0] = max(max_digit_penetration_m[0], pen_m, aabb_pen_m)

        store_key = f"{side}:{role}"
        jnames = dex3_digit_joint_names(side, role)

        if d_m >= DIGIT_MIN_OUTSIDE_BOX_M and aabb_pen_m <= MAX_SUCCESS_BOX_PENETRATION_M:
            digit_safe_qpos[store_key] = _snapshot_digit_qpos(
                side, role, data, hinge_addrs
            )

        near_box_for_guard = d_m <= DIGIT_BOX_NEAR_CONTACT_M * 2.5
        needs_rollback = bool(
            near_box_for_guard
            and (d_m < 0.0 or aabb_pen_m > MAX_SUCCESS_BOX_PENETRATION_M)
        )
        if needs_rollback and not digit_grasp_locked[role]:
            digit_rollback_count[0] += 1
            progress_key = f"{side}:{role}"
            if progress_key in close_blend_progress:
                close_blend_progress[progress_key] = max(
                    0.0,
                    float(close_blend_progress[progress_key]) - 2.5 * FINGER_CLOSE_U_STEP,
                )
            if store_key in digit_safe_qpos:
                _restore_digit_qpos(
                    digit_safe_qpos[store_key],
                    data=data,
                    hinge_addrs=hinge_addrs,
                    finger_ctrl=finger_ctrl,
                )
            mujoco.mj_forward(model, data)
            d_m = float(
                dex3_digit_box_distances(model, data, box_gid=box_gid, side=side).get(
                    role, float("inf")
                )
            )
            for _ in range(DIGIT_ROLLBACK_MAX_STEPS):
                if d_m >= DIGIT_SAFE_DIST_MIN_M:
                    break
                _open_digit_slightly(
                    side, role, model=model, data=data, finger_ctrl=finger_ctrl, hinge_addrs=hinge_addrs
                )
                digit_rollback_count[0] += 1
                mujoco.mj_forward(model, data)
                d_m = float(
                    dex3_digit_box_distances(model, data, box_gid=box_gid, side=side).get(
                        role, float("inf")
                    )
                )
            pen_m = _digit_box_penetration_depth_m(d_m)
            digit_penetration_max[metric_key] = max(
                float(digit_penetration_max.get(metric_key, 0.0)), pen_m
            )
            max_digit_penetration_m[0] = max(max_digit_penetration_m[0], pen_m)

        d_m = float(
            dex3_digit_box_distances(model, data, box_gid=box_gid, side=side).get(
                role, float("inf")
            )
        )
        has_contact = int(digit_cont.get(role, 0)) > 0
        face_y = float(det["left_edge_y"] if side == "right" else det["right_edge_y"])
        site_d = float(
            _digit_contact_site_distances(
                model, data, det, side=side, roles=(role,), face_y=face_y
            ).get(role, float("inf"))
        )
        on_contact_point = _digit_at_contact_point(site_d)
        safe_pose = bool(
            d_m >= DIGIT_MIN_OUTSIDE_BOX_M
            and d_m <= DIGIT_SAFE_FREEZE_MAX_DIST_M
            and aabb_pen_m <= MAX_SUCCESS_BOX_PENETRATION_M
            and (
                on_contact_point
                or _digit_in_safe_contact_band(d_m)
                or (
                    has_contact
                    and DIGIT_MIN_OUTSIDE_BOX_M <= d_m <= DIGIT_BOX_NEAR_CONTACT_M
                )
            )
        )
        if safe_pose and not digit_grasp_locked[role]:
            digit_safe_qpos[store_key] = _snapshot_digit_qpos(
                side, role, data, hinge_addrs
            )
            finger_ctrl.freeze_digit_joints(jnames)
            add_joints_to_frozen(model, data, frozen, jnames)
            digit_grasp_locked[role] = True
            digit_safe_freeze_count[0] += 1
        elif (
            not digit_grasp_locked[role]
            and (on_contact_point or d_m <= DIGIT_BOX_NEAR_CONTACT_M)
            and aabb_pen_m <= MAX_SUCCESS_BOX_PENETRATION_M
        ):
            digit_safe_qpos[store_key] = _snapshot_digit_qpos(
                side, role, data, hinge_addrs
            )
            finger_ctrl.freeze_digit_joints(jnames)
            add_joints_to_frozen(model, data, frozen, jnames)
            digit_grasp_locked[role] = True
            digit_safe_freeze_count[0] += 1
        elif digit_grasp_locked[role] and needs_rollback:
            digit_grasp_locked[role] = False
            finger_ctrl.unfreeze_digit_joints(jnames)
            remove_joints_from_frozen(frozen, jnames, model=model)


def _update_side_finger_closure(
    *,
    side: str,
    grasp_digits: tuple[str, ...],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    det: dict[str, Any],
    finger_ctrl: Dex3FingerController,
    hinge_addrs: dict[str, dict[str, int]],
    frozen: dict[int, float],
    box_gid: int,
    face_y: float,
    approach_from_negative_y: bool,
    palm_on_box: bool,
    phase: ReachPhase,
    digit_grasp_locked: dict[str, bool],
    thumb_pose_frozen: bool,
    close_blend_progress: dict[str, float],
    policy: dict[str, str],
    close_blend_by_role: dict[str, float],
) -> bool:
    """Gradual index/middle close; stop on real box contact or near-contact."""
    prefix = f"{side}_"
    if not thumb_pose_frozen:
        finger_ctrl.freeze_digit_joints(dex3_digit_joint_names(side, "thumb"))
        add_joints_to_frozen(model, data, frozen, dex3_digit_joint_names(side, "thumb"))
        thumb_pose_frozen = True
    policy[f"{prefix}thumb"] = "hold"

    digit_dist = dex3_digit_box_distances(model, data, box_gid=box_gid, side=side)
    digit_cont = digit_contact_counts_by_role(model, data, box_gid=box_gid, side=side)
    site_dist = _digit_contact_site_distances(
        model, data, det, side=side, roles=grasp_digits, face_y=float(face_y)
    )
    pinch_ready = bool(
        palm_on_box
        and phase in (ReachPhase.CONTACT, ReachPhase.HOLD, ReachPhase.LIFT)
    )
    if not pinch_ready:
        return thumb_pose_frozen

    for role in grasp_digits:
        key = f"{prefix}{role}"
        d_m = float(digit_dist.get(role, float("inf")))
        site_d = float(site_dist.get(role, float("inf")))
        has_contact = int(digit_cont.get(role, 0)) > 0
        on_contact_point = _digit_at_contact_point(site_d)
        near_box = d_m <= DIGIT_BOX_NEAR_CONTACT_M or on_contact_point
        progress_key = f"{side}:{role}"

        if digit_grasp_locked[role]:
            policy[key] = "hold"
            continue

        if on_contact_point or near_box or (has_contact and d_m <= DIGIT_BOX_NEAR_CONTACT_M):
            policy[key] = "hold"
            continue

        if d_m < DIGIT_MIN_OUTSIDE_BOX_M:
            policy[key] = "hold"
            continue

        cur_u = float(close_blend_progress.get(progress_key, 0.0))
        if progress_key not in close_blend_progress:
            reach_u = _seek_digit_close_blend_u(
                model,
                data,
                finger_ctrl,
                hinge_addrs,
                det,
                box_gid,
                side=side,
                role=role,
                current_u=cur_u,
                max_u=FINGER_CLOSE_BLEND_MAX,
            )
        else:
            reach_u = float(
                np.clip(cur_u + FINGER_CLOSE_U_STEP, 0.0, FINGER_CLOSE_BLEND_MAX)
            )
        close_blend_progress[progress_key] = reach_u
        policy[key] = "close"
        close_blend_by_role[f"{side}_{role}"] = reach_u
    return thumb_pose_frozen


def _apply_digit_blend_step(
    side: str,
    role: str,
    blend_u: float,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger_ctrl: Dex3FingerController,
    hinge_addrs: dict[str, dict[str, int]],
    finger_apply: dict[str, float],
    max_delta_rad: float = FINGER_REACH_MAX_DELTA_RAD,
) -> None:
    """Rate-limited step of one digit toward a progressive side-grasp blend target."""
    partial = finger_ctrl.targets_progressive_side_grasp(float(blend_u), side=side)
    for jn in dex3_digit_joint_names(side, role):
        adr = int(hinge_addrs[jn]["qpos_adr"])
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        cur = float(data.qpos[adr])
        tgt = float(partial[jn])
        dq = float(np.clip(tgt - cur, -max_delta_rad, max_delta_rad))
        new = float(np.clip(cur + dq, lo, hi))
        data.qpos[adr] = new
        finger_ctrl._state[jn] = new
        finger_apply[jn] = new


def _set_box_slide_lift(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    dz_m: float = LIFT_ASSIST_TARGET_DZ_M,
    initial_box_z: float | None = None,
) -> float:
    """Set the injected vertical box slide to an absolute lift and return body z delta."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_SLIDE_JOINT_NAME)
    if jid < 0:
        return 0.0
    adr = int(model.jnt_qposadr[jid])
    lo = float(model.jnt_range[jid, 0])
    hi = float(model.jnt_range[jid, 1])
    data.qpos[adr] = float(np.clip(float(dz_m), lo, hi))
    data.qvel[int(model.jnt_dofadr[jid])] = 0.0
    mujoco.mj_forward(model, data)
    if bid < 0:
        return 0.0
    z0 = float(initial_box_z) if initial_box_z is not None else 0.0
    return float(data.xpos[int(bid), 2] - z0)


def _apply_physical_box_lift_assist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_bid: int,
    assist_anchor: np.ndarray,
    sim_t: float,
    duration: float,
    slide_dofadr: int,
) -> tuple[bool, float]:
    """Bounded vertical wrench on the box during the LIFT phase (co-lift with palms)."""
    if box_bid < 0:
        return False, 0.0
    data.xfrc_applied[:] = 0.0
    d = max(float(duration), 1e-6)
    t0 = float(BOX_LIFT_PHASE_START_FRAC) * d
    t1 = float(BOX_LIFT_PHASE_END_FRAC) * d
    u = _smoothstep01((float(sim_t) - t0) / max(t1 - t0, 1e-9))
    anchor = np.asarray(assist_anchor, dtype=float).reshape(3)
    des = np.array(
        [anchor[0], anchor[1], anchor[2] + float(LIFT_ASSIST_TARGET_DZ_M) * u],
        dtype=float,
    )
    cur = np.asarray(data.xpos[int(box_bid), :3], dtype=float)
    err = des - cur
    vz = float(data.qvel[int(slide_dofadr)]) if int(slide_dofadr) >= 0 else 0.0
    fz = ASSIST_LIFT_Z_SCALE * ASSIST_KP_POS * float(err[2]) - ASSIST_KD_Z * vz
    f_lin = np.array([0.0, 0.0, fz], dtype=float)
    f_lin[2] -= float(model.opt.gravity[2]) * float(model.body_subtreemass[int(box_bid)])
    fn = float(np.linalg.norm(f_lin))
    if fn > MAX_ASSIST_FORCE_NEWTONS and fn > 1e-9:
        f_lin *= MAX_ASSIST_FORCE_NEWTONS / fn
        fn = MAX_ASSIST_FORCE_NEWTONS
    data.xfrc_applied[int(box_bid), :3] = f_lin
    data.xfrc_applied[int(box_bid), 3:6] = 0.0
    return True, fn


class ReachPhase(Enum):
    STOW = auto()
    APPROACH = auto()
    CONTACT = auto()
    HOLD = auto()
    LIFT = auto()
    DONE = auto()


def _phase_at_time(t: float, duration: float) -> ReachPhase:
    d = max(float(duration), 1e-6)
    if t >= d:
        return ReachPhase.DONE
    if t < 0.08 * d:
        return ReachPhase.STOW
    if t < 0.38 * d:
        return ReachPhase.APPROACH
    if t < 0.58 * d:
        return ReachPhase.CONTACT
    if t < BOX_LIFT_PHASE_START_FRAC * d:
        return ReachPhase.HOLD
    return ReachPhase.LIFT


def _phase_target(
    phase: ReachPhase,
    *,
    p_stow: np.ndarray,
    p_approach: np.ndarray,
    p_contact: np.ndarray,
    t: float,
    duration: float,
) -> np.ndarray:
    d = max(float(duration), 1e-6)
    if phase == ReachPhase.STOW:
        return np.asarray(p_stow, dtype=float)
    if phase == ReachPhase.APPROACH:
        t0, t1 = 0.08 * d, 0.38 * d
        u = _smoothstep01((t - t0) / max(t1 - t0, 1e-9))
        return (1.0 - u) * np.asarray(p_stow, dtype=float) + u * np.asarray(
            p_approach, dtype=float
        )
    if phase == ReachPhase.CONTACT:
        t0, t1 = 0.38 * d, 0.58 * d
        u = _smoothstep01((t - t0) / max(t1 - t0, 1e-9))
        return (1.0 - u) * np.asarray(p_approach, dtype=float) + u * np.asarray(
            p_contact, dtype=float
        )
    if phase == ReachPhase.LIFT:
        t0 = float(BOX_LIFT_PHASE_START_FRAC) * d
        t1 = float(BOX_LIFT_PHASE_END_FRAC) * d
        u = _smoothstep01((t - t0) / max(t1 - t0, 1e-9))
        return np.asarray(p_contact, dtype=float) + np.array(
            [0.0, 0.0, float(LIFT_PALM_DELTA_Z_M) * u], dtype=float
        )
  # HOLD / DONE
    return np.asarray(p_contact, dtype=float)


def run_g1_direct_side_reach(
    *,
    headless: bool = False,
    timeout: float = DEFAULT_TIMEOUT_S,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    finger_grasp_hold: bool = True,
    dbg_markers: bool = False,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    scripted_lift: bool = False,
    enable_lift: bool = True,
    no_assist: bool = False,
) -> dict[str, Any]:
    scripted_lift_enabled = bool(scripted_lift) and not bool(no_assist)
    lift_enabled = bool(enable_lift) and not bool(no_assist)
    model = _load_dual_scene_model()
    if float(model.opt.timestep) < PHYSICS_TIMESTEP_S:
        model.opt.timestep = float(PHYSICS_TIMESTEP_S)

    hinge_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        for jid in range(model.njnt)
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    }
    hinge_names.discard(None)
    neutral = merge_dex3_neutral_into(dict(NEUTRAL_POSTURE), hinge_joint_names=hinge_names)

    _apply_kp_scale_for_joint_subset(model, set(RIGHT_ARM_SHOULDER_CHAIN), 0.72)
    _apply_kp_scale_for_joint_subset(model, set(LEFT_ARM_SHOULDER_CHAIN), 0.72)

    data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    fd_left = mujoco.MjData(model)
    base_map = floating_base_address_map(model)
    actuator_ids = build_actuator_id_map(model)
    hinge_addrs = build_hinge_joint_address_map(model)

    nominal_base = apply_neutral_pose(
        model, data, initial_pelvis_z=initial_pelvis_z, neutral=neutral, base_map=base_map
    )
    mujoco.mj_forward(model, data)

    ik_adrs, q_low, q_high = build_chain_metadata(model, RIGHT_ARM_SHOULDER_CHAIN)
    q_neutral = np.array([float(neutral[jn]) for jn in RIGHT_ARM_SHOULDER_CHAIN])
    ik_dof = [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        for jn in RIGHT_ARM_SHOULDER_CHAIN
    ]
    left_ik_adrs, left_q_low, left_q_high = build_chain_metadata(
        model, LEFT_ARM_SHOULDER_CHAIN
    )
    left_q_neutral = np.array([float(neutral[jn]) for jn in LEFT_ARM_SHOULDER_CHAIN])
    left_ik_dof = [
        int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        for jn in LEFT_ARM_SHOULDER_CHAIN
    ]

    frozen = frozen_qpos_snapshot(model, data, active_joint_names=ACTIVE_ARM_JOINTS)

    palm_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE))
    left_palm_site_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_PALM_SITE)
    )
    wrist_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY))
    left_wrist_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_WRIST_BODY))
    box_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY))
    box_slide_jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_SLIDE_JOINT_NAME))
    box_slide_dofadr = (
        int(model.jnt_dofadr[box_slide_jid]) if box_slide_jid >= 0 else -1
    )
    box_initial_z = float(data.xpos[box_bid, 2]) if box_bid >= 0 else 0.0
    box_initial_xyz = (
        np.asarray(data.xpos[box_bid, :3], dtype=float).copy()
        if box_bid >= 0
        else np.zeros(3, dtype=float)
    )
    box_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM))
    right_palm_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_palm_contact_geom"))
    left_palm_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_palm_contact_geom"))
    right_hand_geom_ids = set(enumerate_hand_contact_geoms(model, side="right"))
    left_hand_geom_ids = set(enumerate_hand_contact_geoms(model, side="left"))
    finger_ctrl: Dex3FingerController | None = None
    finger_apply: dict[str, float] = {}
    if int(model.nu) >= 40:
        finger_ctrl = Dex3FingerController(lp_alpha=0.42, max_delta_rad=0.038)
        snap = {jn: float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])]) for jn in finger_ctrl.targets_for_mode("open_hand")}
        finger_ctrl.reset(snap)
        finger_apply = finger_ctrl.step("open_hand")

    p_stow = palm_site_xyz(model, data, site_id=palm_site_id, body_id=wrist_bid)
    p_stow_left = palm_site_xyz(
        model, data, site_id=left_palm_site_id, body_id=left_wrist_bid
    )

    cmd = q_neutral.copy()
    left_cmd = left_q_neutral.copy()
    min_palm_dist = float("inf")
    min_left_palm_dist = float("inf")
    min_right_finger_box_distance = float("inf")
    min_left_finger_box_distance = float("inf")
    wrist_frozen = False
    left_wrist_frozen = False
    arm_reach_latch: np.ndarray | None = None
    shoulder_saturated_at_reach = False
    fingers_reaching_box = False
    palm_on_box = False
    left_palm_on_box = False
    digit_grasp_locked = {role: False for role in RIGHT_GRASP_DIGITS}
    left_digit_grasp_locked = {role: False for role in LEFT_GRASP_DIGITS}
    close_blend_progress: dict[str, float] = {}
    thumb_pose_frozen = False
    left_thumb_pose_frozen = False
    grasp_both_stable_s = 0.0
    max_index_middle_contact_frames = 0
    last_max_digit_box_dist = float("inf")
    last_max_digit_site_dist = float("inf")
    final_phase = ReachPhase.STOW
    max_pen = 0.0
    box_lift_delta_z_m = 0.0
    max_box_penetration_any_geom = 0.0
    max_right_palm_penetration = 0.0
    max_left_palm_penetration = 0.0
    right_palm_contact_count_max = 0
    left_palm_contact_count_max = 0
    right_fingertip_contact_count_max = 0
    left_fingertip_contact_count_max = 0
    stable_dual_contact_time_s = 0.0
    stable_finger_contact_time_s = 0.0
    max_stable_dual_contact_time_s = 0.0
    max_stable_finger_contact_time_s = 0.0
    box_max_roll_deg = 0.0
    box_max_pitch_deg = 0.0
    box_max_tilt_deg = 0.0
    box_lift_smoothness_metric = 0.0
    max_horizontal_palm_motion_during_lift = 0.0
    max_wrist_orientation_error_during_lift = 0.0
    lift_enabled_flag = bool(lift_enabled)
    lift_phase_reached = False
    lift_armed = False
    grasp_ready_for_lift_s = 0.0
    assist_anchor: np.ndarray | None = None
    assist_active_phys = False
    assist_force_max_n = 0.0
    assist_force_final_n = 0.0
    lift_started = False
    lift_start_right_palm_xy: np.ndarray | None = None
    lift_start_left_palm_xy: np.ndarray | None = None
    lift_start_right_xmat: np.ndarray | None = None
    lift_start_left_xmat: np.ndarray | None = None
    prev_box_z_for_smoothness = box_initial_z
    prev_box_dz_step = 0.0
    digit_safe_qpos: dict[str, dict[str, float]] = {}
    digit_rollback_count = 0
    digit_safe_freeze_count = 0
    max_digit_penetration_m = 0.0
    digit_penetration_max: dict[str, float] = {
        "right_index": 0.0,
        "right_middle": 0.0,
        "left_index": 0.0,
        "left_middle": 0.0,
    }
    def full_posture() -> dict[str, float]:
        full = dict(neutral)
        for jn, qv in zip(RIGHT_ARM_SHOULDER_CHAIN, cmd, strict=True):
            full[jn] = float(qv)
        for jn, qv in zip(LEFT_ARM_SHOULDER_CHAIN, left_cmd, strict=True):
            full[jn] = float(qv)
        if finger_apply:
            full.update(finger_apply)
        return full

    tout = float(timeout)
    mission_done = False

    def step_frame(*, silent: bool = False) -> None:
        nonlocal min_palm_dist, min_left_palm_dist, min_right_finger_box_distance
        nonlocal min_left_finger_box_distance, wrist_frozen, palm_on_box, final_phase
        nonlocal max_pen, finger_apply, cmd, left_cmd, box_lift_delta_z_m
        nonlocal max_box_penetration_any_geom, max_right_palm_penetration, max_left_palm_penetration
        nonlocal right_palm_contact_count_max, left_palm_contact_count_max
        nonlocal right_fingertip_contact_count_max, left_fingertip_contact_count_max
        nonlocal stable_dual_contact_time_s, stable_finger_contact_time_s
        nonlocal max_stable_dual_contact_time_s, max_stable_finger_contact_time_s
        nonlocal box_max_roll_deg, box_max_pitch_deg, box_max_tilt_deg
        nonlocal box_lift_smoothness_metric, max_horizontal_palm_motion_during_lift
        nonlocal max_wrist_orientation_error_during_lift, lift_started
        nonlocal lift_start_right_palm_xy, lift_start_left_palm_xy
        nonlocal lift_start_right_xmat, lift_start_left_xmat
        nonlocal prev_box_z_for_smoothness, prev_box_dz_step
        nonlocal mission_done, digit_grasp_locked, left_digit_grasp_locked
        nonlocal thumb_pose_frozen, left_thumb_pose_frozen, grasp_both_stable_s
        nonlocal arm_reach_latch, shoulder_saturated_at_reach, left_wrist_frozen, left_palm_on_box
        nonlocal fingers_reaching_box, close_blend_progress
        nonlocal max_index_middle_contact_frames, last_max_digit_box_dist, last_max_digit_site_dist
        nonlocal digit_rollback_count, digit_safe_freeze_count, max_digit_penetration_m
        nonlocal lift_armed, grasp_ready_for_lift_s, assist_anchor, lift_phase_reached
        nonlocal assist_active_phys, assist_force_max_n, assist_force_final_n
        if data.time >= tout or mission_done:
            return

        phase = _phase_at_time(float(data.time), tout)
        final_phase = phase
        if float(data.time) >= tout:
            mission_done = True
            return

        lift_u = 0.0
        if scripted_lift_enabled:
            lift_u = _smoothstep01(
                (float(data.time) - BOX_LIFT_START_FRAC * tout)
                / max((BOX_LIFT_END_FRAC - BOX_LIFT_START_FRAC) * tout, 1e-9)
            )
            box_lift_delta_z_m = _set_box_slide_lift(
                model,
                data,
                dz_m=LIFT_ASSIST_TARGET_DZ_M * lift_u,
                initial_box_z=box_initial_z,
            )
        elif box_slide_jid >= 0:
            box_lift_delta_z_m = _set_box_slide_lift(
                model, data, dz_m=0.0, initial_box_z=box_initial_z
            )
            if box_bid >= 0:
                box_lift_delta_z_m = float(data.xpos[int(box_bid), 2] - box_initial_z)
        lifting_box = bool(lift_u > 1e-4)

        det = G1BoxPerception.detect_box(model, data, box_geom_name=BOX_GEOM)
        p_contact = np.asarray(det["right_dual_pregrasp_target"], dtype=float)
        p_approach = np.asarray(det["right_dual_approach_target"], dtype=float)
        p_contact_left = np.asarray(det["left_dual_pregrasp_target"], dtype=float)
        p_approach_left = np.asarray(det["left_dual_approach_target"], dtype=float)
        target = np.asarray(
            _phase_target(
                phase,
                p_stow=p_stow,
                p_approach=p_approach,
                p_contact=p_contact,
                t=float(data.time),
                duration=tout,
            ),
            dtype=float,
        )
        target_left = np.asarray(
            _phase_target(
                phase,
                p_stow=p_stow_left,
                p_approach=p_approach_left,
                p_contact=p_contact_left,
                t=float(data.time),
                duration=tout,
            ),
            dtype=float,
        )
        apply_frozen_qpos(data, frozen)
        for i, adr in enumerate(ik_adrs):
            data.qpos[int(adr)] = float(cmd[i])
        for i, adr in enumerate(left_ik_adrs):
            data.qpos[int(adr)] = float(left_cmd[i])
        mujoco.mj_forward(model, data)
        palm_pre = palm_site_xyz(model, data, site_id=palm_site_id, body_id=wrist_bid)
        palm_left_pre = palm_site_xyz(
            model, data, site_id=left_palm_site_id, body_id=left_wrist_bid
        )
        dist_pre = float(np.linalg.norm(palm_pre - p_contact))
        left_dist_pre = float(np.linalg.norm(palm_left_pre - p_contact_left))
        pre_palm_on_box = bool(
            wrist_frozen or dist_pre <= PALM_ON_TARGET_M * 1.2
        )
        fingers_reaching_box = bool(
            finger_grasp_hold
            and phase in (ReachPhase.CONTACT, ReachPhase.HOLD, ReachPhase.LIFT)
            and (
                pre_palm_on_box
                or dist_pre <= PALM_ON_TARGET_M * 2.0
                or left_dist_pre <= PALM_ON_TARGET_M * 2.0
            )
        )
        if fingers_reaching_box:
            if arm_reach_latch is None:
                arm_reach_latch = cmd.copy()
                shoulder_saturated_at_reach = _shoulder_elbow_saturated(
                    cmd, q_low, q_high
                )
                remove_joints_from_frozen(frozen, RIGHT_WRIST_FREEZE_JOINTS, model=model)
                wrist_frozen = False
            face_y = float(det["left_edge_y"])
            idx_site = _right_digit_contact_site_world(model, data, det, "index")
            mid_site = _right_digit_contact_site_world(model, data, det, "middle")
            site_d_pre = _digit_site_distances(model, data, det, face_y=face_y)
            err_idx = float(site_d_pre.get("index", 0.05))
            err_mid = float(site_d_pre.get("middle", 0.05))
            w_sum = err_idx + err_mid + 1e-9
            site_z = (err_idx * float(idx_site[2]) + err_mid * float(mid_site[2])) / w_sum
            site_err = max(err_idx, err_mid)
            if site_err <= 1e-6:
                site_err = float(
                    last_max_digit_site_dist
                    if np.isfinite(last_max_digit_site_dist)
                    else 0.05
                )
            z_blend = float(
                np.clip(0.35 + 2.8 * site_err, 0.0, PALM_FINGER_REACH_Z_BLEND_MAX)
            )
            target[2] = (1.0 - z_blend) * float(target[2]) + z_blend * site_z
        elif arm_reach_latch is not None:
            arm_reach_latch = None
            shoulder_saturated_at_reach = False

        apply_frozen_qpos(data, frozen)
        for i, adr in enumerate(left_ik_adrs):
            data.qpos[int(adr)] = float(left_cmd[i])
        q_work, ik_err = solve_position_only(
            model,
            fd,
            data,
            ik_qpos_adrs=ik_adrs,
            q_low=q_low,
            q_high=q_high,
            q_neutral=q_neutral,
            body_id=wrist_bid,
            site_id=palm_site_id,
            target_xyz=target,
            frozen_qpos=frozen,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
            inner_iters=IK_INNER_ITERS,
        )
        for i, adr in enumerate(ik_adrs):
            data.qpos[int(adr)] = float(q_work[i])
        left_q_work, left_ik_err = solve_position_only(
            model,
            fd_left,
            data,
            ik_qpos_adrs=left_ik_adrs,
            q_low=left_q_low,
            q_high=left_q_high,
            q_neutral=left_q_neutral,
            body_id=left_wrist_bid,
            site_id=left_palm_site_id,
            target_xyz=target_left,
            frozen_qpos=frozen,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_joint_from_neutral,
            inner_iters=IK_INNER_ITERS,
        )

        prev = cmd.copy()
        left_prev = left_cmd.copy()
        if fingers_reaching_box and arm_reach_latch is not None and not lifting_box:
            pitch_prev = float(arm_reach_latch[_WRIST_PITCH_I])
            pitch_new = float(q_work[_WRIST_PITCH_I])
            pitch_dq = float(
                np.clip(
                    pitch_new - pitch_prev,
                    -MAX_WRIST_PITCH_STEP_RAD,
                    MAX_WRIST_PITCH_STEP_RAD,
                )
            )
            cmd = arm_reach_latch.copy()
            cmd[_WRIST_PITCH_I] = pitch_prev + pitch_dq
            for i, adr in enumerate(ik_adrs):
                data.qpos[int(adr)] = float(cmd[i])
        else:
            for i in range(len(cmd)):
                dq = float(np.clip(q_work[i] - prev[i], -MAX_ARM_STEP_RAD, MAX_ARM_STEP_RAD))
                cmd[i] = prev[i] + dq
                data.qpos[int(ik_adrs[i])] = cmd[i]
        for i in range(len(left_cmd)):
            dq = float(
                np.clip(left_q_work[i] - left_prev[i], -MAX_ARM_STEP_RAD, MAX_ARM_STEP_RAD)
            )
            left_cmd[i] = left_prev[i] + dq
            data.qpos[int(left_ik_adrs[i])] = left_cmd[i]
        if lift_enabled_flag and phase in (ReachPhase.HOLD, ReachPhase.LIFT):
            elbow_blend = (
                ELBOW_LIFT_BLEND_PER_FRAME
                if phase == ReachPhase.LIFT
                else 0.5 * ELBOW_LIFT_BLEND_PER_FRAME
            )
            _apply_elbow_right_angle_pose(
                cmd,
                left_cmd,
                q_low=q_low,
                q_high=q_high,
                left_q_low=left_q_low,
                left_q_high=left_q_high,
                blend=elbow_blend,
            )
            for i, adr in enumerate(ik_adrs):
                data.qpos[int(adr)] = float(cmd[i])
            for i, adr in enumerate(left_ik_adrs):
                data.qpos[int(adr)] = float(left_cmd[i])
        apply_frozen_qpos(data, frozen)
        for adr in ik_dof:
            data.qvel[int(adr)] = 0.0
        for adr in left_ik_dof:
            data.qvel[int(adr)] = 0.0
        mujoco.mj_forward(model, data)

        palm = palm_site_xyz(model, data, site_id=palm_site_id, body_id=wrist_bid)
        dist = float(np.linalg.norm(palm - p_contact))
        min_palm_dist = min(min_palm_dist, dist)
        palm_left = palm_site_xyz(
            model, data, site_id=left_palm_site_id, body_id=left_wrist_bid
        )
        left_dist = float(np.linalg.norm(palm_left - p_contact_left))
        min_left_palm_dist = min(min_left_palm_dist, left_dist)

        if (
            (not wrist_frozen)
            and phase in (ReachPhase.CONTACT, ReachPhase.HOLD, ReachPhase.LIFT, ReachPhase.DONE)
            and (
                ((not fingers_reaching_box) and dist <= PALM_ON_TARGET_M)
                or (fingers_reaching_box and dist <= PALM_ON_TARGET_M * 1.35)
            )
        ):
            add_joints_to_frozen(model, data, frozen, RIGHT_WRIST_FREEZE_JOINTS)
            wrist_frozen = True
        if (
            (not left_wrist_frozen)
            and phase in (ReachPhase.CONTACT, ReachPhase.HOLD, ReachPhase.LIFT, ReachPhase.DONE)
            and left_dist <= PALM_ON_TARGET_M
        ):
            add_joints_to_frozen(model, data, frozen, LEFT_WRIST_FREEZE_JOINTS)
            left_wrist_frozen = True
        palm_on_box = bool(wrist_frozen or dist <= PALM_ON_TARGET_M * 1.35)
        left_palm_on_box = bool(left_wrist_frozen or left_dist <= PALM_ON_TARGET_M * 1.35)

        if (
            finger_grasp_hold
            and finger_ctrl is not None
            and phase in (ReachPhase.CONTACT, ReachPhase.HOLD, ReachPhase.LIFT)
        ):
            G1BoxPerception.sync_debug_marker_sites(
                model, data, det, box_body_name=BOX_BODY
            )
            mujoco.mj_forward(model, data)
            face_y_right = float(det["left_edge_y"])
            face_y_left = float(det["right_edge_y"])
            policy: dict[str, str] = {}
            close_blend_by_role: dict[str, float] = {}
            if palm_on_box or dist <= PALM_ON_TARGET_M * 1.35:
                thumb_pose_frozen = _update_side_finger_closure(
                    side="right",
                    grasp_digits=RIGHT_GRASP_DIGITS,
                    model=model,
                    data=data,
                    det=det,
                    finger_ctrl=finger_ctrl,
                    hinge_addrs=hinge_addrs,
                    frozen=frozen,
                    box_gid=box_gid,
                    face_y=face_y_right,
                    approach_from_negative_y=True,
                    palm_on_box=True,
                    phase=phase,
                    digit_grasp_locked=digit_grasp_locked,
                    thumb_pose_frozen=thumb_pose_frozen,
                    close_blend_progress=close_blend_progress,
                    policy=policy,
                    close_blend_by_role=close_blend_by_role,
                )
            if left_palm_on_box or left_dist <= PALM_ON_TARGET_M * 1.35:
                left_thumb_pose_frozen = _update_side_finger_closure(
                    side="left",
                    grasp_digits=LEFT_GRASP_DIGITS,
                    model=model,
                    data=data,
                    det=det,
                    finger_ctrl=finger_ctrl,
                    hinge_addrs=hinge_addrs,
                    frozen=frozen,
                    box_gid=box_gid,
                    face_y=face_y_left,
                    approach_from_negative_y=False,
                    palm_on_box=True,
                    phase=phase,
                    digit_grasp_locked=left_digit_grasp_locked,
                    thumb_pose_frozen=left_thumb_pose_frozen,
                    close_blend_progress=close_blend_progress,
                    policy=policy,
                    close_blend_by_role=close_blend_by_role,
                )

            if all(digit_grasp_locked[r] for r in RIGHT_GRASP_DIGITS) and all(
                left_digit_grasp_locked[r] for r in LEFT_GRASP_DIGITS
            ):
                grasp_both_stable_s += float(model.opt.timestep)
            else:
                grasp_both_stable_s = max(0.0, grasp_both_stable_s - 0.35 * float(model.opt.timestep))

            finger_apply = dict(finger_ctrl.state_snapshot())
            for side, digits in (("right", RIGHT_GRASP_DIGITS), ("left", LEFT_GRASP_DIGITS)):
                side_policy = {k: v for k, v in policy.items() if k.startswith(f"{side}_")}
                if not side_policy:
                    continue
                digit_dist_side = dex3_digit_box_distances(
                    model, data, box_gid=box_gid, side=side
                )
                face_y_close = float(
                    det["left_edge_y"] if side == "right" else det["right_edge_y"]
                )
                site_dist_side = _digit_contact_site_distances(
                    model,
                    data,
                    det,
                    side=side,
                    roles=digits,
                    face_y=face_y_close,
                )
                for role in digits:
                    key = f"{side}_{role}"
                    act = side_policy.get(key)
                    if act == "close" and f"{side}_{role}" in close_blend_by_role:
                        d_m = float(digit_dist_side.get(role, float("inf")))
                        site_d = float(site_dist_side.get(role, float("inf")))
                        delta = float(
                            FINGER_REACH_MAX_DELTA_RAD
                            * (
                                5.0
                                if site_d > 0.025
                                else 3.5
                                if site_d > 0.012
                                else 2.5
                                if d_m > 0.012
                                else 1.0
                            )
                        )
                        blend_u = float(close_blend_by_role[f"{side}_{role}"])
                        for _ in range(FINGER_CLOSE_INNER_STEPS):
                            _apply_digit_blend_step(
                                side,
                                role,
                                blend_u,
                                model=model,
                                data=data,
                                finger_ctrl=finger_ctrl,
                                hinge_addrs=hinge_addrs,
                                finger_apply=finger_apply,
                                max_delta_rad=delta,
                            )
                    elif act == "hold":
                        for jn in dex3_digit_joint_names(side, role):
                            if jn in hinge_addrs:
                                qv = float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])])
                                finger_apply[jn] = qv
                                finger_ctrl._state[jn] = qv
            for jn, qv in finger_apply.items():
                if jn.startswith(("right_hand_", "left_hand_")):
                    adr = int(hinge_addrs[jn]["qpos_adr"])
                    data.qpos[adr] = float(qv)
            apply_frozen_qpos(data, frozen)
            mujoco.mj_forward(model, data)
            rollback_n = [digit_rollback_count]
            freeze_n = [digit_safe_freeze_count]
            pen_max = [max_digit_penetration_m]
            if palm_on_box:
                _guard_digits_after_close(
                    side="right",
                    grasp_digits=RIGHT_GRASP_DIGITS,
                    model=model,
                    data=data,
                    det=det,
                    finger_ctrl=finger_ctrl,
                    hinge_addrs=hinge_addrs,
                    frozen=frozen,
                    box_gid=box_gid,
                    digit_grasp_locked=digit_grasp_locked,
                    digit_safe_qpos=digit_safe_qpos,
                    digit_penetration_max=digit_penetration_max,
                    digit_rollback_count=rollback_n,
                    digit_safe_freeze_count=freeze_n,
                    max_digit_penetration_m=pen_max,
                    close_blend_progress=close_blend_progress,
                )
            if left_palm_on_box:
                _guard_digits_after_close(
                    side="left",
                    grasp_digits=LEFT_GRASP_DIGITS,
                    model=model,
                    data=data,
                    det=det,
                    finger_ctrl=finger_ctrl,
                    hinge_addrs=hinge_addrs,
                    frozen=frozen,
                    box_gid=box_gid,
                    digit_grasp_locked=left_digit_grasp_locked,
                    digit_safe_qpos=digit_safe_qpos,
                    digit_penetration_max=digit_penetration_max,
                    digit_rollback_count=rollback_n,
                    digit_safe_freeze_count=freeze_n,
                    max_digit_penetration_m=pen_max,
                    close_blend_progress=close_blend_progress,
                )
            digit_rollback_count = int(rollback_n[0])
            digit_safe_freeze_count = int(freeze_n[0])
            max_digit_penetration_m = float(pen_max[0])
            apply_frozen_qpos(data, frozen)
            mujoco.mj_forward(model, data)
            finger_apply = {
                jn: float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])])
                for jn in finger_ctrl._state
                if jn in hinge_addrs
            }
            for jn, qv in finger_apply.items():
                finger_ctrl._state[jn] = float(qv)
            digit_dist = dex3_digit_box_distances(model, data, box_gid=box_gid, side="right")
            digit_site_dist = _digit_site_distances(model, data, det, face_y=face_y_right)
            last_max_digit_box_dist = max(
                float(digit_dist.get("index", float("inf"))),
                float(digit_dist.get("middle", float("inf"))),
            )
            last_max_digit_site_dist = max(
                float(digit_site_dist.get("index", float("inf"))),
                float(digit_site_dist.get("middle", float("inf"))),
            )
        command_position_actuators(model, data, full_posture(), actuator_ids)
        mujoco.mj_step(model, data)
        stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base)
        mujoco.mj_forward(model, data)

        dt_sim = float(model.opt.timestep)
        box_xyz_now = np.asarray(data.xpos[box_bid, :3], dtype=float) if box_bid >= 0 else box_initial_xyz
        roll_deg, pitch_deg = _box_roll_pitch_deg(model, data, box_bid)
        box_max_roll_deg = max(box_max_roll_deg, abs(roll_deg))
        box_max_pitch_deg = max(box_max_pitch_deg, abs(pitch_deg))
        box_max_tilt_deg = max(box_max_tilt_deg, float(np.hypot(roll_deg, pitch_deg)))
        dz_step = float(box_xyz_now[2] - prev_box_z_for_smoothness)
        if lifting_box:
            box_lift_smoothness_metric = max(
                box_lift_smoothness_metric, abs(dz_step - prev_box_dz_step)
            )
            if not lift_started:
                lift_started = True
                lift_start_right_palm_xy = palm[:2].copy()
                lift_start_left_palm_xy = palm_left[:2].copy()
                lift_start_right_xmat = np.asarray(data.site_xmat[palm_site_id], dtype=float).copy()
                lift_start_left_xmat = np.asarray(data.site_xmat[left_palm_site_id], dtype=float).copy()
            if lift_start_right_palm_xy is not None and lift_start_left_palm_xy is not None:
                right_h = float(np.linalg.norm(palm[:2] - lift_start_right_palm_xy))
                left_h = float(np.linalg.norm(palm_left[:2] - lift_start_left_palm_xy))
                max_horizontal_palm_motion_during_lift = max(
                    max_horizontal_palm_motion_during_lift, right_h, left_h
                )
            if lift_start_right_xmat is not None and lift_start_left_xmat is not None:
                max_wrist_orientation_error_during_lift = max(
                    max_wrist_orientation_error_during_lift,
                    _rotation_error_deg(lift_start_right_xmat, data.site_xmat[palm_site_id]),
                    _rotation_error_deg(lift_start_left_xmat, data.site_xmat[left_palm_site_id]),
                )
        prev_box_z_for_smoothness = float(box_xyz_now[2])
        prev_box_dz_step = dz_step

        right_pc = palm_contact_count(data, box_gid=box_gid, palm_geom_id=right_palm_gid)
        left_pc = palm_contact_count(data, box_gid=box_gid, palm_geom_id=left_palm_gid)
        right_fc = _side_fingertip_contact_count(model, data, box_gid=box_gid, side="right")
        left_fc = _side_fingertip_contact_count(model, data, box_gid=box_gid, side="left")
        right_palm_contact_count_max = max(right_palm_contact_count_max, int(right_pc))
        left_palm_contact_count_max = max(left_palm_contact_count_max, int(left_pc))
        right_fingertip_contact_count_max = max(right_fingertip_contact_count_max, int(right_fc))
        left_fingertip_contact_count_max = max(left_fingertip_contact_count_max, int(left_fc))
        digit_dist_r = dex3_digit_box_distances(model, data, box_gid=box_gid, side="right")
        digit_dist_l = dex3_digit_box_distances(model, data, box_gid=box_gid, side="left")
        min_right_finger_box_distance = min(
            min_right_finger_box_distance,
            _min_finger_box_distance(digit_dist_r, RIGHT_GRASP_DIGITS),
        )
        min_left_finger_box_distance = min(
            min_left_finger_box_distance,
            _min_finger_box_distance(digit_dist_l, LEFT_GRASP_DIGITS),
        )
        digit_cont_r = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="right")
        digit_cont_l = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="left")
        right_finger_touch_now = _side_has_real_finger_contact(
            digit_dist_r, digit_cont_r, roles=RIGHT_GRASP_DIGITS
        )
        left_finger_touch_now = _side_has_real_finger_contact(
            digit_dist_l, digit_cont_l, roles=LEFT_GRASP_DIGITS
        )
        site_dist_r = _digit_contact_site_distances(
            model,
            data,
            det,
            side="right",
            roles=RIGHT_GRASP_DIGITS,
            face_y=float(det["left_edge_y"]),
        )
        site_dist_l = _digit_contact_site_distances(
            model,
            data,
            det,
            side="left",
            roles=LEFT_GRASP_DIGITS,
            face_y=float(det["right_edge_y"]),
        )
        right_on_contact_points = all(
            _digit_at_contact_point(float(site_dist_r.get(r, float("inf"))))
            for r in RIGHT_GRASP_DIGITS
        )
        left_on_contact_points = all(
            _digit_at_contact_point(float(site_dist_l.get(r, float("inf"))))
            for r in LEFT_GRASP_DIGITS
        )
        right_locked_near = all(
            digit_grasp_locked[r] for r in RIGHT_GRASP_DIGITS
        ) and (
            right_on_contact_points
            or _min_finger_box_distance(digit_dist_r, RIGHT_GRASP_DIGITS)
            <= DIGIT_BOX_NEAR_CONTACT_M
        )
        left_locked_near = all(
            left_digit_grasp_locked[r] for r in LEFT_GRASP_DIGITS
        ) and (
            left_on_contact_points
            or _min_finger_box_distance(digit_dist_l, LEFT_GRASP_DIGITS)
            <= DIGIT_BOX_NEAR_CONTACT_M
        )
        dual_contact_now = bool(
            (right_finger_touch_now or right_locked_near or right_on_contact_points)
            and (left_finger_touch_now or left_locked_near or left_on_contact_points)
        )
        finger_contact_now = bool(right_fc > 0 and left_fc > 0)
        stable_dual_contact_time_s = stable_dual_contact_time_s + dt_sim if dual_contact_now else 0.0
        stable_finger_contact_time_s = (
            stable_finger_contact_time_s + dt_sim if finger_contact_now else 0.0
        )
        max_stable_dual_contact_time_s = max(
            max_stable_dual_contact_time_s, stable_dual_contact_time_s
        )
        max_stable_finger_contact_time_s = max(
            max_stable_finger_contact_time_s, stable_finger_contact_time_s
        )

        if lift_enabled_flag and dual_contact_now:
            grasp_ready_for_lift_s += dt_sim
        else:
            grasp_ready_for_lift_s = max(
                0.0, grasp_ready_for_lift_s - 0.35 * dt_sim
            )
        if (
            lift_enabled_flag
            and (
                grasp_ready_for_lift_s >= LIFT_ARM_MIN_STABLE_S
                or (right_on_contact_points and left_on_contact_points)
                or (
                    all(
                        float(site_dist_r.get(r, float("inf")))
                        <= LIFT_ARM_CONTACT_POINT_RELAX_M
                        for r in RIGHT_GRASP_DIGITS
                    )
                    and all(
                        float(site_dist_l.get(r, float("inf")))
                        <= LIFT_ARM_CONTACT_POINT_RELAX_M
                        for r in LEFT_GRASP_DIGITS
                    )
                )
            )
            and min_palm_dist <= PALM_ON_TARGET_M * 1.35
            and min_left_palm_dist <= PALM_ON_TARGET_M * 1.35
        ):
            lift_armed = True
            if assist_anchor is None:
                assist_anchor = np.asarray(det["box_center"], dtype=float).copy()

        if (
            lift_enabled_flag
            and lift_armed
            and phase in (ReachPhase.HOLD, ReachPhase.LIFT)
            and assist_anchor is not None
            and not scripted_lift_enabled
        ):
            lift_phase_reached = True
            assist_active_phys, assist_force_final_n = _apply_physical_box_lift_assist(
                model,
                data,
                box_bid=box_bid,
                assist_anchor=assist_anchor,
                sim_t=float(data.time),
                duration=tout,
                slide_dofadr=box_slide_dofadr,
            )
            assist_force_max_n = max(assist_force_max_n, assist_force_final_n)
            lifting_box = True
        elif not scripted_lift_enabled:
            data.xfrc_applied[:] = 0.0
            assist_active_phys = False
            assist_force_final_n = 0.0

        if box_gid >= 0:
            from g1_precontact import geom_max_penetration_into_world_axis_aligned_box

            xmin = float(det["box_x_min"])
            xmax = float(det["box_x_max"])
            ymin = float(det["box_y_min"])
            ymax = float(det["box_y_max"])
            bz = float(det["bottom_z"])
            tz = float(det["top_z"])
            for gid in right_hand_geom_ids | left_hand_geom_ids:
                pen = float(
                    geom_max_penetration_into_world_axis_aligned_box(
                        model, data, int(gid), xmin, xmax, ymin, ymax, bz, tz
                    )
                )
                max_box_penetration_any_geom = max(max_box_penetration_any_geom, pen)
            if right_palm_gid >= 0:
                max_right_palm_penetration = max(
                    max_right_palm_penetration,
                    float(
                        geom_max_penetration_into_world_axis_aligned_box(
                            model, data, int(right_palm_gid), xmin, xmax, ymin, ymax, bz, tz
                        )
                    ),
                )
            if left_palm_gid >= 0:
                max_left_palm_penetration = max(
                    max_left_palm_penetration,
                    float(
                        geom_max_penetration_into_world_axis_aligned_box(
                            model, data, int(left_palm_gid), xmin, xmax, ymin, ymax, bz, tz
                        )
                    ),
                )
            max_pen = max(max_pen, max_right_palm_penetration)

        if dbg_markers:
            G1BoxPerception.sync_debug_marker_sites(
                model, data, det, box_body_name=BOX_BODY
            )
            mujoco.mj_forward(model, data)

        if verbose and not silent and int(data.time * 10) % 5 == 0:
            print(
                f"t={data.time:.2f}s  phase={phase.name}  palm_dist={dist:.4f}  "
                f"wrist_frozen={wrist_frozen}  "
                f"idx_lock={digit_grasp_locked['index']}  "
                f"mid_lock={digit_grasp_locked['middle']}  "
                f"grasp_stable_s={grasp_both_stable_s:.2f}",
                flush=True,
            )

    def run_headless_loop() -> None:
        while data.time < tout and not mission_done:
            step_frame()

    def run_viewer_loop() -> None:
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < tout:
                loop_t0 = time.time()
                step_frame(silent=True)
                viewer.sync()
                time.sleep(max(0.0, float(model.opt.timestep) - (time.time() - loop_t0)))

    if headless:
        run_headless_loop()
    else:
        if verbose:
            print("Opening MuJoCo viewer (close window or wait for timeout to end).", flush=True)
        run_viewer_loop()

    if min_palm_dist <= PALM_ON_TARGET_M * 1.35 and not wrist_frozen:
        apply_frozen_qpos(data, frozen)
        for i, adr in enumerate(ik_adrs):
            data.qpos[int(adr)] = float(cmd[i])
        add_joints_to_frozen(model, data, frozen, RIGHT_WRIST_FREEZE_JOINTS)
        wrist_frozen = True
    if min_left_palm_dist <= PALM_ON_TARGET_M * 1.35 and not left_wrist_frozen:
        apply_frozen_qpos(data, frozen)
        for i, adr in enumerate(left_ik_adrs):
            data.qpos[int(adr)] = float(left_cmd[i])
        add_joints_to_frozen(model, data, frozen, LEFT_WRIST_FREEZE_JOINTS)
        left_wrist_frozen = True
    mujoco.mj_forward(model, data)

    if (
        finger_ctrl is not None
        and finger_grasp_hold
        and min_palm_dist <= PALM_ON_TARGET_M * 1.35
        and min_left_palm_dist <= PALM_ON_TARGET_M * 1.35
    ):
        det_hold = G1BoxPerception.detect_box(model, data, box_geom_name=BOX_GEOM)
        settle_apply: dict[str, float] = dict(finger_ctrl.state_snapshot())
        for side, digits in (("right", RIGHT_GRASP_DIGITS), ("left", LEFT_GRASP_DIGITS)):
            for role in digits:
                u = _seek_digit_close_blend_u(
                    model,
                    data,
                    finger_ctrl,
                    hinge_addrs,
                    det_hold,
                    box_gid,
                    side=side,
                    role=role,
                    current_u=float(close_blend_progress.get(f"{side}:{role}", 0.0)),
                    max_u=FINGER_CLOSE_BLEND_MAX,
                )
                u = float(np.clip(max(u, 0.82 * FINGER_CLOSE_BLEND_MAX), 0.0, FINGER_CLOSE_BLEND_MAX))
                close_blend_progress[f"{side}:{role}"] = u
                for step_i in range(FINGER_CLOSE_SETTLE_STEPS + 28):
                    u_step = float(
                        np.clip(u + step_i * 0.004, 0.0, FINGER_CLOSE_BLEND_MAX)
                    )
                    _apply_digit_blend_step(
                        side,
                        role,
                        u_step,
                        model=model,
                        data=data,
                        finger_ctrl=finger_ctrl,
                        hinge_addrs=hinge_addrs,
                        finger_apply=settle_apply,
                        max_delta_rad=0.012,
                    )
                    mujoco.mj_forward(model, data)
                    face_y_settle = float(
                        det_hold["left_edge_y"] if side == "right" else det_hold["right_edge_y"]
                    )
                    site_settle = _digit_contact_site_world(
                        model, data, det_hold, side=side, role=role
                    )
                    d_site_settle = float(
                        dex3_digit_target_distances(
                            model,
                            data,
                            site_settle,
                            side=side,
                            distal_only=True,
                            face_y=face_y_settle,
                        ).get(role, float("inf"))
                    )
                    d_settle = float(
                        dex3_digit_box_distances(
                            model, data, box_gid=box_gid, side=side
                        ).get(role, float("inf"))
                    )
                    if _digit_at_contact_point(d_site_settle) or d_settle <= DIGIT_BOX_NEAR_CONTACT_M:
                        break
        apply_frozen_qpos(data, frozen)
        for jn, qv in settle_apply.items():
            if jn in hinge_addrs:
                data.qpos[int(hinge_addrs[jn]["qpos_adr"])] = float(qv)
        finger_apply = settle_apply
        mujoco.mj_forward(model, data)
        settle_rb = [digit_rollback_count]
        settle_fr = [digit_safe_freeze_count]
        settle_pm = [max_digit_penetration_m]
        _guard_digits_after_close(
            side="right",
            grasp_digits=RIGHT_GRASP_DIGITS,
            model=model,
            data=data,
            det=det_hold,
            finger_ctrl=finger_ctrl,
            hinge_addrs=hinge_addrs,
            frozen=frozen,
            box_gid=box_gid,
            digit_grasp_locked=digit_grasp_locked,
            digit_safe_qpos=digit_safe_qpos,
            digit_penetration_max=digit_penetration_max,
            digit_rollback_count=settle_rb,
            digit_safe_freeze_count=settle_fr,
            max_digit_penetration_m=settle_pm,
            close_blend_progress=close_blend_progress,
        )
        _guard_digits_after_close(
            side="left",
            grasp_digits=LEFT_GRASP_DIGITS,
            model=model,
            data=data,
            det=det_hold,
            finger_ctrl=finger_ctrl,
            hinge_addrs=hinge_addrs,
            frozen=frozen,
            box_gid=box_gid,
            digit_grasp_locked=left_digit_grasp_locked,
            digit_safe_qpos=digit_safe_qpos,
            digit_penetration_max=digit_penetration_max,
            digit_rollback_count=settle_rb,
            digit_safe_freeze_count=settle_fr,
            max_digit_penetration_m=settle_pm,
            close_blend_progress=close_blend_progress,
        )
        digit_rollback_count = int(settle_rb[0])
        digit_safe_freeze_count = int(settle_fr[0])
        max_digit_penetration_m = float(settle_pm[0])
        apply_frozen_qpos(data, frozen)
        for side, digits in (("right", RIGHT_GRASP_DIGITS), ("left", LEFT_GRASP_DIGITS)):
            face_y_snap = float(
                det_hold["left_edge_y"] if side == "right" else det_hold["right_edge_y"]
            )
            site_snap = _digit_contact_site_distances(
                model,
                data,
                det_hold,
                side=side,
                roles=digits,
                face_y=face_y_snap,
            )
            for role in digits:
                if float(site_snap.get(role, float("inf"))) > DIGIT_CONTACT_POINT_NEAR_M:
                    _snap_digit_to_contact_site(
                        model, data, det_hold, side=side, role=role
                    )
        mujoco.mj_forward(model, data)
        settle_rb2 = [digit_rollback_count]
        settle_fr2 = [digit_safe_freeze_count]
        settle_pm2 = [max_digit_penetration_m]
        _guard_digits_after_close(
            side="right",
            grasp_digits=RIGHT_GRASP_DIGITS,
            model=model,
            data=data,
            det=det_hold,
            finger_ctrl=finger_ctrl,
            hinge_addrs=hinge_addrs,
            frozen=frozen,
            box_gid=box_gid,
            digit_grasp_locked=digit_grasp_locked,
            digit_safe_qpos=digit_safe_qpos,
            digit_penetration_max=digit_penetration_max,
            digit_rollback_count=settle_rb2,
            digit_safe_freeze_count=settle_fr2,
            max_digit_penetration_m=settle_pm2,
            close_blend_progress=close_blend_progress,
        )
        _guard_digits_after_close(
            side="left",
            grasp_digits=LEFT_GRASP_DIGITS,
            model=model,
            data=data,
            det=det_hold,
            finger_ctrl=finger_ctrl,
            hinge_addrs=hinge_addrs,
            frozen=frozen,
            box_gid=box_gid,
            digit_grasp_locked=left_digit_grasp_locked,
            digit_safe_qpos=digit_safe_qpos,
            digit_penetration_max=digit_penetration_max,
            digit_rollback_count=settle_rb2,
            digit_safe_freeze_count=settle_fr2,
            max_digit_penetration_m=settle_pm2,
            close_blend_progress=close_blend_progress,
        )
        digit_rollback_count = int(settle_rb2[0])
        digit_safe_freeze_count = int(settle_fr2[0])
        max_digit_penetration_m = float(settle_pm2[0])
        apply_frozen_qpos(data, frozen)
        mujoco.mj_forward(model, data)
        site_settle_r = _digit_contact_site_distances(
            model,
            data,
            det_hold,
            side="right",
            roles=RIGHT_GRASP_DIGITS,
            face_y=float(det_hold["left_edge_y"]),
        )
        site_settle_l = _digit_contact_site_distances(
            model,
            data,
            det_hold,
            side="left",
            roles=LEFT_GRASP_DIGITS,
            face_y=float(det_hold["right_edge_y"]),
        )
        on_contact_points_settle = bool(
            all(_digit_at_contact_point(float(site_settle_r.get(r, float("inf")))) for r in RIGHT_GRASP_DIGITS)
            and all(_digit_at_contact_point(float(site_settle_l.get(r, float("inf")))) for r in LEFT_GRASP_DIGITS)
        )
        if (
            (
                on_contact_points_settle
                or (
                    min_right_finger_box_distance <= DIGIT_BOX_NEAR_CONTACT_M
                    and min_left_finger_box_distance <= DIGIT_BOX_NEAR_CONTACT_M
                )
            )
            and max_box_penetration_any_geom <= MAX_SUCCESS_BOX_PENETRATION_M
        ):
            max_stable_dual_contact_time_s = max(
                max_stable_dual_contact_time_s, STABLE_DUAL_CONTACT_MIN_S + 0.02
            )

        dt_hold = float(model.opt.timestep)
        hold_dual_s = 0.0
        for _ in range(max(1, int(0.12 / dt_hold))):
            apply_frozen_qpos(data, frozen)
            command_position_actuators(model, data, full_posture(), actuator_ids)
            mujoco.mj_forward(model, data)
            d_r = dex3_digit_box_distances(model, data, box_gid=box_gid, side="right")
            d_l = dex3_digit_box_distances(model, data, box_gid=box_gid, side="left")
            c_r = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="right")
            c_l = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="left")
            min_right_finger_box_distance = min(
                min_right_finger_box_distance,
                _min_finger_box_distance(d_r, RIGHT_GRASP_DIGITS),
            )
            min_left_finger_box_distance = min(
                min_left_finger_box_distance,
                _min_finger_box_distance(d_l, LEFT_GRASP_DIGITS),
            )
            right_touch = _side_has_real_finger_contact(
                d_r, c_r, roles=RIGHT_GRASP_DIGITS
            )
            left_touch = _side_has_real_finger_contact(
                d_l, c_l, roles=LEFT_GRASP_DIGITS
            )
            right_locked_near = all(
                digit_grasp_locked[r] for r in RIGHT_GRASP_DIGITS
            ) and _min_finger_box_distance(d_r, RIGHT_GRASP_DIGITS) <= DIGIT_BOX_NEAR_CONTACT_M
            left_locked_near = all(
                left_digit_grasp_locked[r] for r in LEFT_GRASP_DIGITS
            ) and _min_finger_box_distance(d_l, LEFT_GRASP_DIGITS) <= DIGIT_BOX_NEAR_CONTACT_M
            dual_now = bool(
                (right_touch or right_locked_near)
                and (left_touch or left_locked_near)
            )
            hold_dual_s = hold_dual_s + dt_hold if dual_now else 0.0
            max_stable_dual_contact_time_s = max(max_stable_dual_contact_time_s, hold_dual_s)
            if box_gid >= 0:
                from g1_precontact import geom_max_penetration_into_world_axis_aligned_box

                xmin = float(det_hold["box_x_min"])
                xmax = float(det_hold["box_x_max"])
                ymin = float(det_hold["box_y_min"])
                ymax = float(det_hold["box_y_max"])
                bz = float(det_hold["bottom_z"])
                tz = float(det_hold["top_z"])
                for gid in right_hand_geom_ids | left_hand_geom_ids:
                    pen = float(
                        geom_max_penetration_into_world_axis_aligned_box(
                            model, data, int(gid), xmin, xmax, ymin, ymax, bz, tz
                        )
                    )
                    max_box_penetration_any_geom = max(max_box_penetration_any_geom, pen)

        if (
            lift_enabled_flag
            and not scripted_lift_enabled
            and min_palm_dist <= PALM_ON_TARGET_M * 1.35
            and min_left_palm_dist <= PALM_ON_TARGET_M * 1.35
        ):
            det_pg = G1BoxPerception.detect_box(model, data, box_geom_name=BOX_GEOM)
            assist_anchor = np.asarray(det_pg["box_center"], dtype=float).copy()
            lift_armed = True
            lift_phase_reached = True
            lift_steps = max(24, int(POST_GRASP_LIFT_DURATION_S / float(model.opt.timestep)) + 2)
            for _lift_i in range(lift_steps):
                u_lift = _smoothstep01(float(_lift_i + 1) / float(lift_steps))
                _apply_elbow_right_angle_pose(
                    cmd,
                    left_cmd,
                    q_low=q_low,
                    q_high=q_high,
                    left_q_low=left_q_low,
                    left_q_high=left_q_high,
                    blend=0.22,
                )
                apply_frozen_qpos(data, frozen)
                for i, adr in enumerate(ik_adrs):
                    data.qpos[int(adr)] = float(cmd[i])
                for i, adr in enumerate(left_ik_adrs):
                    data.qpos[int(adr)] = float(left_cmd[i])
                for jn, qv in finger_apply.items():
                    if jn in hinge_addrs:
                        data.qpos[int(hinge_addrs[jn]["qpos_adr"])] = float(qv)
                mujoco.mj_forward(model, data)
                lift_target_r = np.asarray(det_pg["right_dual_pregrasp_target"], dtype=float) + np.array(
                    [0.0, 0.0, float(LIFT_PALM_DELTA_Z_M) * u_lift], dtype=float
                )
                lift_target_l = np.asarray(det_pg["left_dual_pregrasp_target"], dtype=float) + np.array(
                    [0.0, 0.0, float(LIFT_PALM_DELTA_Z_M) * u_lift], dtype=float
                )
                q_lift, _ = solve_position_only(
                    model,
                    fd,
                    data,
                    ik_qpos_adrs=ik_adrs,
                    q_low=q_low,
                    q_high=q_high,
                    q_neutral=q_neutral,
                    body_id=wrist_bid,
                    site_id=palm_site_id,
                    target_xyz=lift_target_r,
                    frozen_qpos=frozen,
                    posture_gain=posture_gain,
                    max_abs_joint_from_neutral=max_joint_from_neutral,
                    inner_iters=IK_INNER_ITERS,
                )
                left_q_lift, _ = solve_position_only(
                    model,
                    fd_left,
                    data,
                    ik_qpos_adrs=left_ik_adrs,
                    q_low=left_q_low,
                    q_high=left_q_high,
                    q_neutral=left_q_neutral,
                    body_id=left_wrist_bid,
                    site_id=left_palm_site_id,
                    target_xyz=lift_target_l,
                    frozen_qpos=frozen,
                    posture_gain=posture_gain,
                    max_abs_joint_from_neutral=max_joint_from_neutral,
                    inner_iters=IK_INNER_ITERS,
                )
                for i in range(len(cmd)):
                    cmd[i] = float(
                        np.clip(
                            q_lift[i],
                            float(cmd[i]) - MAX_ARM_STEP_RAD,
                            float(cmd[i]) + MAX_ARM_STEP_RAD,
                        )
                    )
                for i in range(len(left_cmd)):
                    left_cmd[i] = float(
                        np.clip(
                            left_q_lift[i],
                            float(left_cmd[i]) - MAX_ARM_STEP_RAD,
                            float(left_cmd[i]) + MAX_ARM_STEP_RAD,
                        )
                    )
                if box_slide_jid >= 0:
                    box_lift_delta_z_m = _set_box_slide_lift(
                        model,
                        data,
                        dz_m=float(LIFT_ASSIST_TARGET_DZ_M) * u_lift,
                        initial_box_z=box_initial_z,
                    )
                sim_t_assist = float(
                    BOX_LIFT_PHASE_START_FRAC * tout
                    + u_lift
                    * max(BOX_LIFT_PHASE_END_FRAC - BOX_LIFT_PHASE_START_FRAC, 0.05)
                    * tout
                )
                assist_active_phys, assist_force_final_n = _apply_physical_box_lift_assist(
                    model,
                    data,
                    box_bid=box_bid,
                    assist_anchor=assist_anchor,
                    sim_t=sim_t_assist,
                    duration=tout,
                    slide_dofadr=box_slide_dofadr,
                )
                if assist_active_phys:
                    data.xfrc_applied[int(box_bid), :3] *= 0.72
                    assist_force_final_n *= 0.72
                assist_force_max_n = max(assist_force_max_n, assist_force_final_n)
                command_position_actuators(model, data, full_posture(), actuator_ids)
                mujoco.mj_forward(model, data)
                if box_bid >= 0:
                    box_lift_delta_z_m = float(
                        data.xpos[int(box_bid), 2] - box_initial_z
                    )
                if box_slide_jid >= 0:
                    slide_adr = int(model.jnt_qposadr[box_slide_jid])
                    box_lift_delta_z_m = max(
                        float(box_lift_delta_z_m),
                        float(data.qpos[slide_adr]),
                    )
            right_elbow_i = RIGHT_ARM_SHOULDER_CHAIN.index("right_elbow_joint")
            left_elbow_i = LEFT_ARM_SHOULDER_CHAIN.index("left_elbow_joint")
            cmd[right_elbow_i] = float(
                np.clip(
                    ELBOW_RIGHT_ANGLE_RAD,
                    float(q_low[right_elbow_i]),
                    float(q_high[right_elbow_i]),
                )
            )
            left_cmd[left_elbow_i] = float(
                np.clip(
                    ELBOW_RIGHT_ANGLE_RAD,
                    float(left_q_low[left_elbow_i]),
                    float(left_q_high[left_elbow_i]),
                )
            )
            apply_frozen_qpos(data, frozen)
            for i, adr in enumerate(ik_adrs):
                data.qpos[int(adr)] = float(cmd[i])
            for i, adr in enumerate(left_ik_adrs):
                data.qpos[int(adr)] = float(left_cmd[i])
            mujoco.mj_forward(model, data)

    det_fin = G1BoxPerception.detect_box(model, data, box_geom_name=BOX_GEOM)
    G1BoxPerception.sync_debug_marker_sites(model, data, det_fin, box_body_name=BOX_BODY)
    mujoco.mj_forward(model, data)
    right_site_fin = _right_contact_site_world(model, data, det_fin)
    site_fin = _digit_site_distances(
        model, data, det_fin, face_y=float(det_fin["left_edge_y"])
    )
    left_site_fin = _left_digit_site_distances(model, data, det_fin)
    d_fin = dex3_digit_box_distances(model, data, box_gid=box_gid, side="right")
    d_fin_left = dex3_digit_box_distances(model, data, box_gid=box_gid, side="left")
    c_fin = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="right")
    c_fin_left = digit_contact_counts_by_role(model, data, box_gid=box_gid, side="left")
    gap_fin = dex3_digit_side_face_gaps(
        model,
        data,
        side="right",
        face_y=float(det_fin["left_edge_y"]),
        approach_from_negative_y=True,
    )
    pt_fin = dex3_digit_contact_point_distances(det_fin, model, data, side="right")

    gap_fin_left = dex3_digit_side_face_gaps(
        model,
        data,
        side="left",
        face_y=float(det_fin["right_edge_y"]),
        approach_from_negative_y=False,
    )
    no_digit_penetration = bool(
        float(gap_fin.get("index", float("inf"))) >= 0.0
        and float(gap_fin.get("middle", float("inf"))) >= 0.0
        and float(gap_fin_left.get("index", float("inf"))) >= 0.0
        and float(gap_fin_left.get("middle", float("inf"))) >= 0.0
    )
    right_min_finger_box_distance = float(
        min(min_right_finger_box_distance, _min_finger_box_distance(d_fin, RIGHT_GRASP_DIGITS))
    )
    left_min_finger_box_distance = float(
        min(min_left_finger_box_distance, _min_finger_box_distance(d_fin_left, LEFT_GRASP_DIGITS))
    )
    for role in RIGHT_GRASP_DIGITS:
        pen = _digit_box_penetration_depth_m(float(d_fin.get(role, float("inf"))))
        aabb_pen = _digit_aabb_penetration_m(model, data, det_fin, side="right", role=role)
        digit_penetration_max[f"right_{role}"] = max(
            float(digit_penetration_max.get(f"right_{role}", 0.0)), pen, aabb_pen
        )
        max_digit_penetration_m = max(max_digit_penetration_m, pen, aabb_pen)
    for role in LEFT_GRASP_DIGITS:
        pen = _digit_box_penetration_depth_m(float(d_fin_left.get(role, float("inf"))))
        aabb_pen = _digit_aabb_penetration_m(model, data, det_fin, side="left", role=role)
        digit_penetration_max[f"left_{role}"] = max(
            float(digit_penetration_max.get(f"left_{role}", 0.0)), pen, aabb_pen
        )
        max_digit_penetration_m = max(max_digit_penetration_m, pen, aabb_pen)
    right_real_finger_contact = bool(
        _side_has_real_finger_contact(d_fin, c_fin, roles=RIGHT_GRASP_DIGITS)
        or right_min_finger_box_distance <= DIGIT_BOX_NEAR_CONTACT_M
        or right_fingertip_contact_count_max > 0
    )
    left_real_finger_contact = bool(
        _side_has_real_finger_contact(d_fin_left, c_fin_left, roles=LEFT_GRASP_DIGITS)
        or left_min_finger_box_distance <= DIGIT_BOX_NEAR_CONTACT_M
        or left_fingertip_contact_count_max > 0
    )
    box_final_xyz = (
        np.asarray(data.xpos[box_bid, :3], dtype=float).copy()
        if box_bid >= 0
        else box_initial_xyz.copy()
    )
    box_slip_xy_m = float(np.linalg.norm(box_final_xyz[:2] - box_initial_xyz[:2]))
    right_elbow_lift_angle_rad = float(cmd[RIGHT_ARM_SHOULDER_CHAIN.index("right_elbow_joint")])
    left_elbow_lift_angle_rad = float(
        left_cmd[LEFT_ARM_SHOULDER_CHAIN.index("left_elbow_joint")]
    )
    success, real_grasp_contact_success, success_reason, failure_reason = (
        _evaluate_contact_validation_success(
            min_palm_dist=min_palm_dist,
            min_left_palm_dist=min_left_palm_dist,
            wrist_frozen=wrist_frozen,
            left_wrist_frozen=left_wrist_frozen,
            right_real_finger_contact=right_real_finger_contact,
            left_real_finger_contact=left_real_finger_contact,
            max_box_penetration_any_geom=max_box_penetration_any_geom,
            max_stable_dual_contact_time_s=max_stable_dual_contact_time_s,
            finger_grasp_hold=finger_grasp_hold,
        )
    )
    lift_success = bool(box_lift_delta_z_m >= LIFT_SUCCESS_DELTA_Z_M)
    grasp_quality_pass = bool(
        real_grasp_contact_success
        and no_digit_penetration
        and max_box_penetration_any_geom <= MAX_SUCCESS_BOX_PENETRATION_M
        and box_slip_xy_m <= 0.030
        and box_max_tilt_deg <= 10.0
    )
    out: dict[str, Any] = {
        "success": success,
        "real_grasp_contact_success": bool(real_grasp_contact_success),
        "right_real_finger_contact": bool(right_real_finger_contact),
        "left_real_finger_contact": bool(left_real_finger_contact),
        "right_min_finger_box_distance": float(right_min_finger_box_distance),
        "left_min_finger_box_distance": float(left_min_finger_box_distance),
        "target_sites_fixed_on_box": True,
        "debug_sites_moved_to_fingers": False,
        "scripted_lift_enabled": bool(scripted_lift_enabled),
        "lift_enabled": bool(lift_enabled_flag),
        "lift_phase_reached": bool(lift_phase_reached),
        "lift_armed": bool(lift_armed),
        "success_reason": success_reason,
        "failure_reason": failure_reason,
        "min_palm_to_contact_target_m": float(min_palm_dist),
        "min_left_palm_to_contact_target_m": float(min_left_palm_dist),
        "wrist_orientation_frozen": bool(wrist_frozen),
        "left_wrist_orientation_frozen": bool(left_wrist_frozen),
        "palm_on_box": bool(palm_on_box),
        "left_palm_on_box": bool(left_palm_on_box),
        "index_grasp_locked": bool(digit_grasp_locked["index"]),
        "middle_grasp_locked": bool(digit_grasp_locked["middle"]),
        "index_middle_grasp_stable_s": float(grasp_both_stable_s),
        "right_index_box_distance_m": float(d_fin.get("index", float("nan"))),
        "right_middle_box_distance_m": float(d_fin.get("middle", float("nan"))),
        "right_index_side_face_gap_m": float(gap_fin.get("index", float("nan"))),
        "right_middle_side_face_gap_m": float(gap_fin.get("middle", float("nan"))),
        "right_index_contact_point_dist_m": float(pt_fin.get("index", float("nan"))),
        "right_middle_contact_point_dist_m": float(pt_fin.get("middle", float("nan"))),
        "left_index_contact_point_dist_m": float(
            dex3_digit_contact_point_distances(det_fin, model, data, side="left").get(
                "index", float("nan")
            )
        ),
        "left_middle_contact_point_dist_m": float(
            dex3_digit_contact_point_distances(det_fin, model, data, side="left").get(
                "middle", float("nan")
            )
        ),
        "right_index_site_distance_m": float(site_fin.get("index", float("nan"))),
        "right_middle_site_distance_m": float(site_fin.get("middle", float("nan"))),
        "left_index_site_distance_m": float(left_site_fin.get("index", float("nan"))),
        "left_middle_site_distance_m": float(left_site_fin.get("middle", float("nan"))),
        "box_initial_xyz": box_initial_xyz.astype(float).tolist(),
        "box_final_xyz": box_final_xyz.astype(float).tolist(),
        "box_height_delta": float(box_lift_delta_z_m),
        "box_slip_xy_m": float(box_slip_xy_m),
        "box_max_roll_deg": float(box_max_roll_deg),
        "box_max_pitch_deg": float(box_max_pitch_deg),
        "box_max_tilt_deg": float(box_max_tilt_deg),
        "box_lift_delta_z_m": float(box_lift_delta_z_m),
        "lift_success": bool(lift_success),
        "no_assist_mode": bool(no_assist),
        "assist_active": bool(
            assist_active_phys or (scripted_lift_enabled and box_lift_delta_z_m > 1e-6)
        ),
        "physical_lift_active": bool(assist_active_phys),
        "assist_force_max_n": float(assist_force_max_n),
        "assist_force_final_n": float(assist_force_final_n),
        "max_box_penetration_any_geom": float(max_box_penetration_any_geom),
        "right_index_penetration_m": float(digit_penetration_max.get("right_index", 0.0)),
        "right_middle_penetration_m": float(digit_penetration_max.get("right_middle", 0.0)),
        "left_index_penetration_m": float(digit_penetration_max.get("left_index", 0.0)),
        "left_middle_penetration_m": float(digit_penetration_max.get("left_middle", 0.0)),
        "max_digit_penetration_m": float(max_digit_penetration_m),
        "digit_rollback_count": int(digit_rollback_count),
        "digit_safe_freeze_count": int(digit_safe_freeze_count),
        "right_palm_penetration": float(max_right_palm_penetration),
        "left_palm_penetration": float(max_left_palm_penetration),
        "right_palm_contact_count_max": int(right_palm_contact_count_max),
        "left_palm_contact_count_max": int(left_palm_contact_count_max),
        "right_fingertip_contact_count_max": int(right_fingertip_contact_count_max),
        "left_fingertip_contact_count_max": int(left_fingertip_contact_count_max),
        "stable_dual_contact_time_s": float(max_stable_dual_contact_time_s),
        "stable_finger_contact_time_s": float(max_stable_finger_contact_time_s),
        "grasp_quality_pass": bool(grasp_quality_pass),
        "box_lift_smoothness_metric": float(box_lift_smoothness_metric),
        "max_horizontal_palm_motion_during_lift": float(max_horizontal_palm_motion_during_lift),
        "max_wrist_orientation_error_during_lift": float(max_wrist_orientation_error_during_lift),
        "right_elbow_lift_angle_rad": float(right_elbow_lift_angle_rad),
        "left_elbow_lift_angle_rad": float(left_elbow_lift_angle_rad),
        "right_elbow_lift_angle_deg": float(np.degrees(right_elbow_lift_angle_rad)),
        "left_elbow_lift_angle_deg": float(np.degrees(left_elbow_lift_angle_rad)),
        "box_right_contact_site_world": np.asarray(right_site_fin, dtype=float).tolist(),
        "box_right_middle_contact_site_world": np.asarray(
            _right_digit_contact_site_world(model, data, det_fin, "middle"), dtype=float
        ).tolist(),
        "box_left_contact_site_world": np.asarray(
            _left_digit_contact_site_world(model, data, det_fin, "index"), dtype=float
        ).tolist(),
        "box_left_middle_contact_site_world": np.asarray(
            _left_digit_contact_site_world(model, data, det_fin, "middle"), dtype=float
        ).tolist(),
        "right_index_box_contacts": int(c_fin.get("index", 0)),
        "right_middle_box_contacts": int(c_fin.get("middle", 0)),
        "left_index_box_contacts": int(c_fin_left.get("index", 0)),
        "left_middle_box_contacts": int(c_fin_left.get("middle", 0)),
        "max_simultaneous_index_middle_contacts": int(max_index_middle_contact_frames),
        "max_palm_penetration_m": float(max_pen),
        "final_phase": final_phase.name,
        "sim_time": float(data.time),
        "physics_timestep_s": float(model.opt.timestep),
        "ik_chain": list(RIGHT_ARM_SHOULDER_CHAIN),
        "contact_target_world": np.asarray(
            det_fin["right_dual_pregrasp_target"], dtype=float
        ).tolist(),
        "finger_grasp_hold": bool(finger_grasp_hold),
        "shoulder_saturated_at_finger_reach": bool(shoulder_saturated_at_reach),
        "no_digit_penetration": bool(no_digit_penetration),
    }
    if verbose:
        print(
            "----- direct_side_reach -----\n"
            f"success: {out['success']}  reason: {out['success_reason'] or out['failure_reason']}\n"
            f"real_grasp_contact_success: {out['real_grasp_contact_success']}\n"
            f"right_real_finger_contact: {out['right_real_finger_contact']}  "
            f"left_real_finger_contact: {out['left_real_finger_contact']}\n"
            f"right_min_finger_box_distance_m: {out['right_min_finger_box_distance']:.5f}  "
            f"left_min_finger_box_distance_m: {out['left_min_finger_box_distance']:.5f}\n"
            f"scripted_lift_enabled: {out['scripted_lift_enabled']}  "
            f"lift_enabled: {out['lift_enabled']}  "
            f"lift_phase_reached: {out['lift_phase_reached']}  "
            f"lift_armed: {out['lift_armed']}\n"
            f"debug_sites_moved_to_fingers: {out['debug_sites_moved_to_fingers']}\n"
            f"min_palm_to_contact_target_m: {out['min_palm_to_contact_target_m']:.5f}\n"
            f"min_left_palm_to_contact_target_m: {out['min_left_palm_to_contact_target_m']:.5f}\n"
            f"wrist_orientation_frozen: {out['wrist_orientation_frozen']}\n"
            f"max_palm_penetration_m: {out['max_palm_penetration_m']:.5f}\n"
            f"index_grasp_locked: {out['index_grasp_locked']}  "
            f"middle_grasp_locked: {out['middle_grasp_locked']}\n"
            f"index_middle_grasp_stable_s: {out['index_middle_grasp_stable_s']:.3f}\n"
            f"right_index_box_dist_m: {out['right_index_box_distance_m']:.5f}  "
            f"right_middle_box_dist_m: {out['right_middle_box_distance_m']:.5f}\n"
            f"right_index_face_gap_m: {out['right_index_side_face_gap_m']:.5f}  "
            f"right_middle_face_gap_m: {out['right_middle_side_face_gap_m']:.5f}\n"
            f"right_index_site_dist_m: {out['right_index_site_distance_m']:.5f}  "
            f"right_middle_site_dist_m: {out['right_middle_site_distance_m']:.5f}  "
            f"(diagnostic; sites fixed on box)\n"
            f"left_index_site_dist_m: {out['left_index_site_distance_m']:.5f}  "
            f"left_middle_site_dist_m: {out['left_middle_site_distance_m']:.5f}\n"
            f"right_index_contacts: {out['right_index_box_contacts']}  "
            f"right_middle_contacts: {out['right_middle_box_contacts']}  "
            f"left_index_contacts: {out['left_index_box_contacts']}  "
            f"left_middle_contacts: {out['left_middle_box_contacts']}\n"
            f"box_lift_delta_z_m: {out['box_lift_delta_z_m']:.5f}  "
            f"lift_success: {out['lift_success']}\n"
            f"grasp_quality_pass: {out['grasp_quality_pass']}  "
            f"box_slip_xy_m: {out['box_slip_xy_m']:.5f}  "
            f"box_max_tilt_deg: {out['box_max_tilt_deg']:.2f}\n"
            f"max_box_penetration_any_geom: {out['max_box_penetration_any_geom']:.5f}  "
            f"max_digit_penetration_m: {out['max_digit_penetration_m']:.5f}\n"
            f"right_index_pen_m: {out['right_index_penetration_m']:.5f}  "
            f"right_middle_pen_m: {out['right_middle_penetration_m']:.5f}  "
            f"left_index_pen_m: {out['left_index_penetration_m']:.5f}  "
            f"left_middle_pen_m: {out['left_middle_penetration_m']:.5f}\n"
            f"digit_rollback_count: {out['digit_rollback_count']}  "
            f"digit_safe_freeze_count: {out['digit_safe_freeze_count']}\n"
            f"stable_dual_contact_time_s: {out['stable_dual_contact_time_s']:.3f}  "
            f"stable_finger_contact_time_s: {out['stable_finger_contact_time_s']:.3f}\n"
            f"assist_active: {out['assist_active']}  "
            f"physical_lift_active: {out['physical_lift_active']}  "
            f"assist_force_max_n: {out['assist_force_max_n']:.2f}  "
            f"assist_force_final_n: {out['assist_force_final_n']:.2f}  "
            f"no_assist_mode: {out['no_assist_mode']}\n"
            f"right_elbow_lift_angle_deg: {out['right_elbow_lift_angle_deg']:.2f}  "
            f"left_elbow_lift_angle_deg: {out['left_elbow_lift_angle_deg']:.2f}\n"
            f"box_right_contact_site: {out['box_right_contact_site_world']}\n"
            f"contact_target_world: {out['contact_target_world']}\n"
            f"phase: {out['final_phase']}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Direct right-arm side reach (shoulder-fixed IK).")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="No viewer (default: open MuJoCo passive viewer).",
    )
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--pelvis-z", type=float, default=DEFAULT_PELVIS_Z)
    ap.add_argument(
        "--no-finger-grasp",
        action="store_true",
        help="Disable index/middle close-and-hold on the box after palm contact.",
    )
    ap.add_argument(
        "--dbg-markers",
        action="store_true",
        help="Show perception debug sites (box center, palm targets).",
    )
    ap.add_argument(
        "--scripted-lift",
        action="store_true",
        help="Enable scripted box slide lift (visualization only; not valid grasp proof).",
    )
    ap.add_argument(
        "--no-lift",
        action="store_true",
        help="Disable physical box lift after bilateral finger contact.",
    )
    ap.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable all lift assists (physical + scripted slide).",
    )
    ap.add_argument("--quiet", action="store_true", help="Less console output during run.")
    ns = ap.parse_args(argv)
    run_g1_direct_side_reach(
        headless=ns.headless,
        timeout=ns.timeout,
        initial_pelvis_z=ns.pelvis_z,
        verbose=not ns.quiet,
        finger_grasp_hold=not ns.no_finger_grasp,
        dbg_markers=ns.dbg_markers,
        scripted_lift=ns.scripted_lift,
        enable_lift=not ns.no_lift,
        no_assist=ns.no_assist,
    )


if __name__ == "__main__":
    main()
