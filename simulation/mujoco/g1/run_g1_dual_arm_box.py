#!/usr/bin/env python3
"""
Locked-base **dual-arm** box coordination (IK + dual-side contact + bounded ``xfrc_applied`` assist).

Uses ``g1_reach_box_scene_dex3.xml`` (dual-milestone centered box + **Dex3-1** hands), :class:`~perception.g1_box_perception.G1BoxPerception`, and the
same vertical box slide injection as ``run_g1_grasp_box.py``. Coordination is **primary-right / support-left**: the right arm
solves the full waist+arm chain until the right palm enters a tight pregrasp band, then the waist is latched and both arms use
**arm-only** IK (left at reduced gains); the left arm never owns waist DOFs. While the waist is still free, **left arm IK is gated**:
the right palm must stay within ``SEQUENTIAL_IK_RIGHT_TO_GOAL_M`` of its active Cartesian goal for ``SEQUENTIAL_IK_RIGHT_POSE_HOLD_FRAMES``
consecutive steps (or right–box contact) before the left arm IK runs, so the arms do not fight the same torso DOFs during approach.

**Assist / lift gating** requires shallow bilateral surface contact (MuJoCo contacts on both ±y sides), palm centers strictly outside the box AABB, palm plates on the correct sides of the torso midplane, conservative palm/wrist separation (Dex3) or legacy proxy–proxy checks, penetration ≤ ``MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M``, and continuous bilateral stability for ``DUAL_CONTACT_SETTLE_TIME_S`` before bounded ``xfrc_applied`` engages or the grasp-hold segment advances.
Palm–IK norms (**right < 7 cm**, **left < 10 cm**) define ``strict_dual_pose`` for timeline squeeze clearance.

The timeline is **staged**: nominal approach/descend, then **DUAL_CONTACT_HOLD** until **both**
arms satisfy palm–target thresholds; only **afterwards** does the scripted ``CLOSE_PROXY_HANDS``
segment begin (name retained for compatibility; **Dex3** finger closure runs there).

**Dex3 path** (``nu == 43``): real finger collision meshes, :mod:`g1_dex_hand_contact` summaries, and :class:`g1_dex3_finger_control.Dex3FingerController` (rate-limited, low-passed). Legacy proxy sliders are disabled. See :mod:`g1_proxy_deprecated` for the old proxy-only contract.

Assisted lift remains a bounded wrench on ``reach_target_box`` after bilateral contact cues.

Run::

    python simulation/mujoco/g1/run_g1_dual_arm_box.py
    python simulation/mujoco/g1/run_g1_dual_arm_box.py --headless --timeout 24 --dbg-markers

"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator, Mapping

import mujoco
import numpy as np

_G1 = Path(__file__).resolve().parent
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perception.g1_box_perception import (  # noqa: E402
    DUAL_SURFACE_CONTACT_CLEARANCE_Y_M,
    G1BoxPerception,
)

from g1_precontact import (  # noqa: E402
    DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M,
    MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M,
    PALM_PLATE_HALF_THICKNESS_X,
    PALM_PLATE_HALF_WIDTH_Y,
    PALM_SURFACE_CLEARANCE,
    geom_max_penetration_into_world_axis_aligned_box,
    palm_plate_max_world_x_extent,
)
from g1_dex3_finger_control import (  # noqa: E402
    Dex3FingerController,
    dex3_digit_joint_names,
    merge_dex3_neutral_into,
)
from g1_dex_hand_contact import (  # noqa: E402
    digit_contact_counts_by_role,
    dex3_digit_box_distances,
    dex3_finger_clearance_metrics,
    dex3_per_hand_lift_ready_relaxed,
    enumerate_hand_contact_geoms,
    hand_box_contact_summary,
)
from run_g1_grasp_box import (  # noqa: E402 — shared injector + actuator merge
    ASSIST_KP_POS,
    ASSIST_KD_Z,
    ASSIST_LIFT_Z_SCALE,
    BOX_SLIDE_JOINT_NAME,
    GraspPhase,
    G1_DUAL_ARM_NU_DEX3,
    LIFT_PALM_DELTA_Z_M,
    LIFT_SUCCESS_DELTA_Z_M,
    LIFT_ASSIST_TARGET_DZ_M,
    MAX_ASSIST_FORCE_NEWTONS,
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    LEGACY_BOX_GEOM_NAME,
    PIPELINE_G1_DEX3_HANDS_XML,
    REACH_BOX_DUAL_MILESTONE_SCENE_PATH,
    _command_grasp_actuators,
    _geom_names,
    _has_actuated_finger_joints,
    _inject_box_vertical_slide_for_grasp,
    _proxy_finger_slide_target,
    _smoothstep01,
    _touch_contact_gids as _right_touch_contact_gids_grasp,
)
from run_g1_posture_hold import (  # noqa: E402
    NEUTRAL_POSTURE,
    DEFAULT_PELVIS_Z,
    PRINT_INTERVAL,
    apply_neutral_pose,
    build_actuator_id_map,
    build_hinge_joint_address_map,
    floating_base_address_map,
    stabilize_floating_base,
)
from run_g1_right_arm_ik_demo import (  # noqa: E402
    IK_POSTURE_GAIN,
    IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    IK_JOINT_NAMES as RIGHT_IK_JOINT_NAMES_ONLY,
    _apply_kp_scale_for_joint_subset,
    solve_ik_q,
)

RIGHT_SITE = "right_palm_site"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_SITE = "left_palm_site"
LEFT_WRIST_BODY = "left_wrist_yaw_link"

RIGHT_PALM_GEOM = "right_palm_contact_geom"
LEFT_PALM_GEOM = "left_palm_contact_geom"
# Legacy proxy geom names (``g1_position_actuated`` / old reach scene). Absent in Dex3 MJCF.
LEFT_PROXY_GEOMS = ("left_proxy_left_finger_geom", "left_proxy_right_finger_geom")
RIGHT_PROXY_GEOMS = ("right_proxy_left_finger_geom", "right_proxy_right_finger_geom")
DUAL_IK_KP_SCALE = 0.68

# Alternating IK is Jacobian-heavy; fewer inner NR steps keep headless timelines practical.
DUAL_ALT_IK_INNER_ITERS = 7
DUAL_HOLD_IK_INNER_ITERS = 9

# Palm–IK targets (instantaneous norms after mj_forward during the physics step).
RIGHT_PALM_CONTACT_GOAL_M = 0.070
LEFT_PALM_CONTACT_GOAL_M = 0.100

# Dual milestone: outside ±y palm strips + capped proxy slide travel (see MJCF slide ranges).
DUAL_PROXY_MAX_SLIDE_M = 0.0045
# Allowed ±y deviation from nominal outside-edge palm strips (faces at ymin / ymax).
EDGE_CONTACT_Y_PAD_INBOARD_M = 0.026
EDGE_CONTACT_Y_PAD_OUTBOARD_M = 0.036
# Separation threshold for mj_geomDistance (negative slight overlap allowed; capped by penetration checks).
GEOM_PAIR_TOUCH_DIST_OK_M = 0.0045
# Bilateral surface-contact persistence before grasp_hold → lift progression / assist arming.
DUAL_CONTACT_SETTLE_TIME_S = 0.25
# Dex3: sticky real-contact / alignment seconds before lift assist (target band ~0.25–0.35 s).
DEX3_LIFT_CONTACT_SETTLE_TIME_S = 0.30
DEX3_STICKY_CONTACT_LEAK = 0.48
DEX3_PALM_NORMAL_OK = 0.18
DEX3_PALM_NORMAL_CLOSE_OK = 0.14
DEX3_PALM_UP_OK = 0.42
DEX3_CLEAN_PENETRATION_CAP_M = 0.010
DEX3_DIGIT_HOLD_NEAR_BOX_M = 0.010
# Freeze digit joints once MuJoCo contact or within this distance (m) of the box.
DEX3_DIGIT_CONTACT_FREEZE_DIST_M = 0.012
DEX3_DIGIT_APPROACH_MAX_DELTA_RAD = 0.011
DEX3_DIGIT_OPEN_PENETRATION_M = 0.0085
# Stop closing fingers if distal geoms from opposite hands approach this close (m).
DEX3_CROSS_HAND_FINGER_STOP_M = 0.034
DEX3_FINGERTIP_NEAR_BOX_M = 0.040
# Soft palm orientation — heavily damped (was fighting wrists); authority scaled ~72% down vs legacy.
PALM_AXIS_GAIN_BASE = 0.62
PALM_UP_GAIN_BASE = 0.28
ORI_AUTHORITY_SCALE = 0.34
RIGHT_ORI_GAIN_MUL = 1.0
LEFT_ORI_GAIN_MUL = 0.62
RIGHT_POSITION_TASK_GAIN = 0.92
LEFT_POSITION_TASK_GAIN = 0.58

# Primary-right / support-left coordination (no alternating ``equal'' full chains).
RIGHT_PREGRASP_STABLE_FOR_WAIST_FREEZE_M = 0.048
LEFT_PREGRASP_LOOSE_FOR_WAIST_FREEZE_M = 0.185
WAIST_FREEZE_DEBOUNCE_STEPS = 5
SUPPORT_LEFT_POSTURE_SCALE = 0.46
SUPPORT_LEFT_POSITION_GAIN_MUL = 0.45
SUPPORT_LEFT_ORI_MUL = 0.40
SUPPORT_LEFT_TARGET_BLEND = 0.62
IK_ITERS_RIGHT_PRIMARY_FULL = 11
IK_ITERS_RIGHT_ARM_LOCKED = 7
IK_ITERS_LEFT_SUPPORT = 6
IK_ERR_REJECT_LEFT_SUPPORT_M = 0.042
# Sequential IK (waist-free phase): left arm IK only after right palm holds near its **active** goal for N steps.
SEQUENTIAL_IK_RIGHT_TO_GOAL_M = 0.050
SEQUENTIAL_IK_RIGHT_POSE_HOLD_FRAMES = 8

FREEZE_WAIST_AFTER_CONTACT = True
WAIST_RATE_LIMIT_RAD_PER_STEP = 0.008
WRIST_LP_ALPHA = 0.30
WRIST_MAX_DELTA_RAD = 0.022
MAX_ARM_JOINT_STEP_RAD = 0.016
MAX_WAIST_JOINT_STEP_RAD = 0.007
IK_ERR_REJECT_M = 0.03

# Command-space low-pass after IK step limits (reduces right-arm motor chatter before ``ctrl``).
RIGHT_ARM_CMD_SMOOTH_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEFT_ARM_CMD_SMOOTH_JOINTS: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_CMD_LP_ALPHA = 0.09
LEFT_ARM_CMD_LP_ALPHA = 0.07
RIGHT_ARM_CMD_MAX_DELTA_RAD = 0.012
LEFT_ARM_CMD_MAX_DELTA_RAD = 0.014
# Raw right palm goal low-pass (approach / descend only).
PR_GOAL_FILTER_ALPHA = 0.20
# Palm speed averaging window: pre-step palm–goal distance must be below this (m).
NEAR_CONTACT_PALM_DIST_M = 0.10
# IK reject: require this many consecutive over-threshold solves before counting a debounced recovery.
IK_REJECT_DEBOUNCE_FRAMES = 3
# Palm orientation gain multiplier during stable contact hold (approach/descend = position-only IK).
ORI_GAIN_MUL_CONTACT_HOLD_UNSTABLE = 0.68
ORI_GAIN_MUL_CONTACT_HOLD_STABLE = 0.90
ORI_CONTACT_HOLD_STABLE_BILATERAL_S = 0.12

DEFAULT_DUAL_ARM_TIMEOUT_S = 24.0
FAST_DUAL_ARM_TIMEOUT_CAP_S = 12.0
LEFT_ARM_NEUTRAL_DRIFT_ALPHA = 0.055

# Runtime timeline compression (>1 reaches contact / finger phases sooner in sim time).
_ACTIVE_TIMELINE_SPEED = 1.0


@dataclass(frozen=True)
class SimMotionProfile:
    """Joint / finger rate limits and IK budget for a run (wall-clock + apparent motion)."""

    physics_timestep_s: float
    max_arm_joint_step_rad: float
    max_waist_joint_step_rad: float
    right_cmd_max_delta_rad: float
    left_cmd_max_delta_rad: float
    right_cmd_lp_alpha: float
    left_cmd_lp_alpha: float
    pr_goal_filter_alpha: float
    finger_max_delta_rad: float
    finger_lp_alpha: float
    digit_approach_max_delta_rad: float
    ik_iter_scale: float
    timeline_speed: float
    viewer_playback_mul: float
    sequential_right_hold_frames: int


def _sim_motion_profile(*, fast: bool) -> SimMotionProfile:
    if fast:
        return SimMotionProfile(
            physics_timestep_s=0.005,
            max_arm_joint_step_rad=0.032,
            max_waist_joint_step_rad=0.012,
            right_cmd_max_delta_rad=0.024,
            left_cmd_max_delta_rad=0.026,
            right_cmd_lp_alpha=0.18,
            left_cmd_lp_alpha=0.14,
            pr_goal_filter_alpha=0.32,
            finger_max_delta_rad=0.034,
            finger_lp_alpha=0.34,
            digit_approach_max_delta_rad=0.022,
            ik_iter_scale=0.72,
            timeline_speed=1.48,
            viewer_playback_mul=2.8,
            sequential_right_hold_frames=5,
        )
    return SimMotionProfile(
        physics_timestep_s=0.004,
        max_arm_joint_step_rad=0.023,
        max_waist_joint_step_rad=0.010,
        right_cmd_max_delta_rad=0.018,
        left_cmd_max_delta_rad=0.020,
        right_cmd_lp_alpha=0.14,
        left_cmd_lp_alpha=0.11,
        pr_goal_filter_alpha=0.28,
        finger_max_delta_rad=0.027,
        finger_lp_alpha=0.30,
        digit_approach_max_delta_rad=0.017,
        ik_iter_scale=0.86,
        timeline_speed=1.24,
        viewer_playback_mul=1.85,
        sequential_right_hold_frames=6,
    )


def _apply_physics_timestep(model: mujoco.MjModel, target_dt_s: float) -> float:
    """Coarsen integrator step for fewer steps per simulated second (returns applied dt)."""
    dt0 = float(model.opt.timestep)
    tgt = float(max(dt0, target_dt_s))
    model.opt.timestep = tgt
    return tgt

# IK iteration budgets (hierarchical stages).
IK_ITERS_RIGHT_LEAD = 11
IK_ITERS_RIGHT_SOFT = 5
IK_ITERS_LEFT_PRIMARY = 11
IK_ITERS_LEFT_COOP = 5
IK_ITERS_DESCEND_RIGHT = 10
IK_ITERS_DESCEND_LEFT = 6
IK_ITERS_FROZEN_EACH = 6
# Lift-phase inward squeeze bias on palm IK targets (toward box center, shallow).
DUAL_LIFT_INWARD_SQUEEZE_Y_M = 0.00075
# Proxy–proxy separation alert threshold (negative ⇒ geometric overlap).
PROXY_CROSS_ARM_SEP_WARN_M = 0.004
# Dual motion: defer left IK until right is nearly on the scripted near-face palm target.
_RIGHT_ENGAGE_LEFT_ARM_M = 0.110

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
RIGHT_PALM_AXIS_TARGET = np.array([0.0, 1.0, 0.0], dtype=float)
LEFT_PALM_AXIS_TARGET = np.array([0.0, -1.0, 0.0], dtype=float)

# Post-contact-hold fractions of remaining sim budget (after squeeze gate clears).
_POST_GATE_FRACS: tuple[tuple[str, float], ...] = (
    ("close", 0.045),
    ("grasp_hold", 0.028),
    ("lift", 0.465),
    ("lower", 0.148),
    ("open", 0.058),
    ("retreat", 0.256),
)
# Finger placement only: long close + hold on box; no lift / lower.
_POST_GATE_FRACS_FINGER_PLACE: tuple[tuple[str, float], ...] = (
    ("close", 0.14),
    ("grasp_hold", 0.76),
    ("open", 0.05),
    ("retreat", 0.05),
)


def _post_gate_fracs(*, finger_place_only: bool) -> tuple[tuple[str, float], ...]:
    return _POST_GATE_FRACS_FINGER_PLACE if finger_place_only else _POST_GATE_FRACS

DUAL_IK_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
)

LEFT_TOUCH_FALLBACK = ("left_hand_collision", "left_wrist_collision")

LEFT_IK_JOINT_NAMES_ONLY: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
)

DUAL_RIGHT_IK_JOINT_NAMES_ONLY: tuple[str, ...] = RIGHT_IK_JOINT_NAMES_ONLY + (
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
)
DUAL_LEFT_IK_JOINT_NAMES_ONLY: tuple[str, ...] = LEFT_IK_JOINT_NAMES_ONLY + (
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
)

WAIST_IK_JOINT_NAMES: tuple[str, ...] = ("waist_yaw_joint", "waist_pitch_joint")
RIGHT_ARM_IK_JOINT_NAMES_ONLY: tuple[str, ...] = tuple(
    j for j in DUAL_RIGHT_IK_JOINT_NAMES_ONLY if j not in WAIST_IK_JOINT_NAMES
)
LEFT_ARM_IK_JOINT_NAMES_ONLY: tuple[str, ...] = tuple(
    j for j in DUAL_LEFT_IK_JOINT_NAMES_ONLY if j not in WAIST_IK_JOINT_NAMES
)
WRIST_JOINT_NAMES_DUAL: tuple[str, ...] = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_WRIST_JOINT_NAMES_DUAL = tuple(j for j in WRIST_JOINT_NAMES_DUAL if j.startswith("right_"))
LEFT_WRIST_JOINT_NAMES_DUAL = tuple(j for j in WRIST_JOINT_NAMES_DUAL if j.startswith("left_"))
RIGHT_WRIST_ROLL_YAW_JOINTS = ("right_wrist_roll_joint", "right_wrist_yaw_joint")
LEFT_WRIST_ROLL_YAW_JOINTS = ("left_wrist_roll_joint", "left_wrist_yaw_joint")
RIGHT_WRIST_PITCH_JOINT = "right_wrist_pitch_joint"
LEFT_WRIST_PITCH_JOINT = "left_wrist_pitch_joint"


class DualPhase(Enum):
    STOW = auto()
    DUAL_APPROACH = auto()
    DUAL_DESCEND_TO_PREGRASP = auto()
    DUAL_CONTACT_HOLD = auto()
    CLOSE_PROXY_HANDS = auto()
    DUAL_GRASP_HOLD = auto()
    DUAL_LIFT_TEST = auto()
    DUAL_LOWER_BACK = auto()
    OPEN_PROXY_HANDS = auto()
    RETREAT = auto()
    DONE = auto()


FINGER_REACH_WRIST_PHASES = frozenset(
    {DualPhase.DUAL_CONTACT_HOLD, DualPhase.CLOSE_PROXY_HANDS}
)


class CoordinationMode(Enum):
    """Monotonic dual-arm policy: right owns waist until freeze, then locked torso + arm-only solves."""

    IDLE = auto()
    RIGHT_PRIMARY_WAIST_FREE = auto()
    WAIST_LOCKED_DUAL_ARM = auto()


def _dex3_effective_finger_mode(
    phase: DualPhase,
    *,
    safety_stop: bool,
    closure_block: str | None,
    palms_aligned_for_close: bool,
    finger_place_only: bool = False,
) -> str:
    """Dex3 finger posture vs timeline (enum still uses *PROXY* for legacy compatibility)."""
    if safety_stop:
        return "open_hand"
    if phase == DualPhase.STOW:
        return "open_hand"
    if phase == DualPhase.DUAL_APPROACH:
        return "pregrasp_spread"
    if phase == DualPhase.DUAL_DESCEND_TO_PREGRASP:
        return "light_side_prepare"
    if phase == DualPhase.DUAL_CONTACT_HOLD:
        if finger_place_only:
            return "light_side_prepare"
        return "side_support_partial"
    if phase == DualPhase.CLOSE_PROXY_HANDS:
        if closure_block and not finger_place_only:
            return "pregrasp_spread"
        if finger_place_only or palms_aligned_for_close:
            return "side_support_grasp"
        return "side_support_partial"
    if phase in (DualPhase.OPEN_PROXY_HANDS,):
        return "release"
    if phase in (DualPhase.RETREAT, DualPhase.DONE):
        return "open_hand"
    if phase == DualPhase.DUAL_GRASP_HOLD:
        if closure_block:
            return "pregrasp_spread"
        if finger_place_only:
            return "side_support_grasp"
        return "lift_hold_grasp"
    if phase in (DualPhase.DUAL_LIFT_TEST, DualPhase.DUAL_LOWER_BACK):
        if finger_place_only:
            return "side_support_grasp"
        return "lift_hold_grasp"
    return "open_hand"


ORIENTATION_SOFT_PHASES = frozenset(
    {
        DualPhase.DUAL_APPROACH,
        DualPhase.DUAL_DESCEND_TO_PREGRASP,
        DualPhase.DUAL_CONTACT_HOLD,
        DualPhase.CLOSE_PROXY_HANDS,
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
    }
)

# Freeze palm/wrist orientation targets during side grasp + close + lift (no re-aiming).
WRIST_ORIENTATION_LOCK_PHASES = frozenset(
    {
        DualPhase.DUAL_CONTACT_HOLD,
        DualPhase.CLOSE_PROXY_HANDS,
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK,
    }
)
WRIST_LOCK_PALM_SETTLE_S = 0.10
WRIST_LOCK_INNER_ITERS_SCALE = 0.52
WRIST_LOCK_MAX_WRIST_CMD_DELTA_RAD = 0.0045
WRIST_LOCK_ORI_GAIN_MUL = 0.11
WRIST_LOCK_DURATION_TRACK = True


def _locked_site_axis_snapshot(
    data: mujoco.MjData,
    site_id: int,
    nominal_axis_world: np.ndarray,
) -> np.ndarray:
    """Pick palm-site axis column best aligned with nominal; sign toward box interaction."""
    nom = np.asarray(nominal_axis_world, dtype=float).reshape(3)
    nn = float(np.linalg.norm(nom))
    if nn < 1e-12:
        return _site_axis_world(data, site_id, axis_col=0)
    nom = nom / nn
    best_ax = _site_axis_world(data, site_id, axis_col=0)
    best_s = abs(float(np.dot(best_ax, nom)))
    for col in (1, 2):
        ax = _site_axis_world(data, site_id, axis_col=col)
        s = abs(float(np.dot(ax, nom)))
        if s > best_s:
            best_s = s
            best_ax = ax
    if float(np.dot(best_ax, nom)) < 0.0:
        best_ax = -best_ax
    nax = float(np.linalg.norm(best_ax))
    return best_ax / nax if nax > 1e-12 else nom


def _locked_up_from_axis(locked_axis: np.ndarray) -> np.ndarray:
    a = np.asarray(locked_axis, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    if na < 1e-12:
        return WORLD_UP.copy()
    a = a / na
    up = WORLD_UP - float(np.dot(WORLD_UP, a)) * a
    nu = float(np.linalg.norm(up))
    return up / nu if nu > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)


def _latch_wrist_roll_yaw(
    data: mujoco.MjData,
    joint_qpos_adr: Mapping[str, int],
    joint_names: tuple[str, ...],
) -> dict[str, float]:
    """Snapshot roll/yaw hinge qpos before finger reach (orientation unfreeze)."""
    out: dict[str, float] = {}
    for jn in joint_names:
        out[jn] = float(data.qpos[int(joint_qpos_adr[jn])])
    return out


def _wrist_lock_palm_ori_kw(axis_world: np.ndarray, up_world: np.ndarray) -> dict[str, Any]:
    """Low-gain orientation hold (no dynamic retargeting)."""
    mul = float(WRIST_LOCK_ORI_GAIN_MUL)
    ga = PALM_AXIS_GAIN_BASE * ORI_AUTHORITY_SCALE * mul
    gu = PALM_UP_GAIN_BASE * ORI_AUTHORITY_SCALE * mul
    return {
        "palm_axis_target_world": np.asarray(axis_world, dtype=float).reshape(3).copy(),
        "palm_axis_task_gain": float(ga),
        "palm_up_target_world": np.asarray(up_world, dtype=float).reshape(3).copy(),
        "palm_up_task_gain": float(gu),
    }


def _palm_axis_tracking_error(
    data: mujoco.MjData,
    site_id: int,
    lock_axis_world: np.ndarray,
) -> float:
    """0 = perfect alignment of best palm normal with locked axis; ~1 = opposite."""
    la = np.asarray(lock_axis_world, dtype=float).reshape(3)
    nl = float(np.linalg.norm(la))
    if nl < 1e-12:
        return 0.0
    la = la / nl
    best = -1.0
    for col in (0, 1, 2):
        ax = _site_axis_world(data, site_id, axis_col=col)
        best = max(best, abs(float(np.dot(ax, la))))
    return float(max(0.0, 1.0 - best))


def _dex3_timeline_phase_display(phase: DualPhase, *, use_dex3: bool) -> str:
    """Human-readable phase for logs (Dex3 renames proxy-era close segment)."""
    if use_dex3 and phase == DualPhase.CLOSE_PROXY_HANDS:
        return "CLOSE_DEX3_HANDS"
    return phase.name


PRE_RIGHT_ONLY_PHASES = frozenset(
    {
        DualPhase.STOW,
        DualPhase.DUAL_APPROACH,
        DualPhase.DUAL_DESCEND_TO_PREGRASP,
    }
)
PRE_GATE = frozenset(
    {DualPhase.STOW, DualPhase.DUAL_APPROACH, DualPhase.DUAL_DESCEND_TO_PREGRASP}
)
PALM_IK_PULLBACK_PHASES = frozenset(
    {
        DualPhase.DUAL_APPROACH,
        DualPhase.DUAL_DESCEND_TO_PREGRASP,
        DualPhase.DUAL_CONTACT_HOLD,
        DualPhase.CLOSE_PROXY_HANDS,
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK,
    }
)

# Penetration diagnostics: peak overlap into box AABB per episode segment (not timeline retuning).
PENETRATION_DIAG_PHASES = frozenset(
    {
        DualPhase.DUAL_DESCEND_TO_PREGRASP,
        DualPhase.DUAL_CONTACT_HOLD,
        DualPhase.CLOSE_PROXY_HANDS,
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
    }
)
ARM_ASSIST_PHASES = frozenset({DualPhase.DUAL_CONTACT_HOLD, DualPhase.CLOSE_PROXY_HANDS})
ACTIVE_ASSIST_PHASES = frozenset(
    {
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK,
    }
)


def _dual_phase_boundaries(duration: float) -> dict[str, float]:
    """Timeline boundaries with lengthened approach/descend (+ contact hold slack)."""
    d = max(float(duration), 1e-6)
    spd = float(max(_ACTIVE_TIMELINE_SPEED, 1.0))
    k_appr = 1.52 * spd
    k_desc = 1.58 * spd
    k_hold = 1.22 * spd
    t_stow_end = 0.09 * d
    t_approach_end = t_stow_end + 0.19 * d * k_appr
    t_descend_end = t_approach_end + 0.16 * d * k_desc
    t_contact_hold_end = t_descend_end + 0.14 * d * k_hold
    tail_scale = (d - t_contact_hold_end) / max((1.0 - 0.58) * d, 1e-9)

    def _tail(old_frac: float) -> float:
        return float(t_contact_hold_end + (old_frac - 0.58) * d * tail_scale)

    return {
        "t_stow_end": float(t_stow_end),
        "t_approach_end": float(t_approach_end),
        "t_descend_end": float(t_descend_end),
        "t_contact_hold_end": float(t_contact_hold_end),
        "t_close_proxy_end": _tail(0.62),
        "t_grasp_hold_end": _tail(0.64),
        "t_lift_end": _tail(0.84),
        "t_lower_end": _tail(0.90),
        "t_open_end": _tail(0.92),
        "t_retreat_end": _tail(0.973),
    }


def _coordination_mode(
    phase: DualPhase,
    *,
    waist_frozen_xy: np.ndarray | None,
) -> CoordinationMode:
    if phase in (DualPhase.DONE, DualPhase.STOW):
        return CoordinationMode.IDLE
    if waist_frozen_xy is not None:
        return CoordinationMode.WAIST_LOCKED_DUAL_ARM
    return CoordinationMode.RIGHT_PRIMARY_WAIST_FREE


def _dual_phase_effective(
    sim_t: float,
    duration: float,
    squeeze_cleared_t: float | None,
    *,
    post_gate_virtual_lag_s: float = 0.0,
    finger_place_only: bool = False,
) -> DualPhase:
    """Scheduling: nominal stow→approach→descend→hold(gated), then fractional post-close timeline.

    ``post_gate_virtual_lag_s`` stalls the post–squeeze-branch clock (advances slower than ``sim_t``)
    so ``CLOSE_PROXY_HANDS`` does not consume wall time while palm–IK distances exceed thresholds.
    """
    d = max(float(duration), 1e-9)
    t = float(sim_t)
    if t + 1e-12 >= d:
        return DualPhase.DONE
    bd = _dual_phase_boundaries(d)

    if squeeze_cleared_t is None:
        if t < bd["t_stow_end"]:
            return DualPhase.STOW
        if t < bd["t_approach_end"]:
            return DualPhase.DUAL_APPROACH
        if t < bd["t_descend_end"]:
            return DualPhase.DUAL_DESCEND_TO_PREGRASP
        return DualPhase.DUAL_CONTACT_HOLD

    pb = max(d - float(squeeze_cleared_t), 1e-9)
    t_post = t - float(post_gate_virtual_lag_s)
    dt = max(0.0, t_post - float(squeeze_cleared_t))
    return _post_gate_budget_and_phase(dt, pb, finger_place_only=finger_place_only)


def _post_gate_abs_spans(
    squeeze_t: float,
    duration: float,
    *,
    finger_place_only: bool = False,
) -> dict[str, tuple[float, float]]:
    dur = max(float(duration), 1e-9)
    t0 = float(squeeze_t)
    pb = max(dur - t0, 1e-9)
    fracs = _post_gate_fracs(finger_place_only=finger_place_only)
    wt = sum(w for _, w in fracs)
    t = t0
    out: dict[str, tuple[float, float]] = {}
    for key, frac in fracs:
        seg = pb * frac / wt
        out[key] = (t, t + seg)
        t += seg
    return out


_POST_GATE_KEY_TO_DUAL: dict[str, DualPhase] = {
    "close": DualPhase.CLOSE_PROXY_HANDS,
    "grasp_hold": DualPhase.DUAL_GRASP_HOLD,
    "lift": DualPhase.DUAL_LIFT_TEST,
    "lower": DualPhase.DUAL_LOWER_BACK,
    "open": DualPhase.OPEN_PROXY_HANDS,
    "retreat": DualPhase.RETREAT,
}


def _post_gate_budget_and_phase(
    dt_after_gate: float,
    post_budget: float,
    *,
    finger_place_only: bool = False,
) -> DualPhase:
    """Map elapsed time since squeeze gate clears into post-contact timeline."""
    dt = float(max(0.0, dt_after_gate))
    pb = float(max(post_budget, 1e-6))
    fracs = _post_gate_fracs(finger_place_only=finger_place_only)
    wt = sum(w for _, w in fracs)
    acc = 0.0
    for key, frac in fracs:
        seg = pb * (frac / wt)
        if dt < acc + seg - 1e-11:
            return _POST_GATE_KEY_TO_DUAL[key]
        acc += seg
    return DualPhase.DONE


def _grasp_phase_for_proxy(p: DualPhase) -> GraspPhase:
    return {
        DualPhase.STOW: GraspPhase.STOW,
        DualPhase.DUAL_APPROACH: GraspPhase.APPROACH_ABOVE_BOX,
        DualPhase.DUAL_DESCEND_TO_PREGRASP: GraspPhase.DESCEND_TO_PREGRASP,
        DualPhase.DUAL_CONTACT_HOLD: GraspPhase.PRE_GRASP_HOLD,
        DualPhase.CLOSE_PROXY_HANDS: GraspPhase.CLOSE_HAND,
        DualPhase.DUAL_GRASP_HOLD: GraspPhase.GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST: GraspPhase.LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK: GraspPhase.LOWER_BACK,
        DualPhase.OPEN_PROXY_HANDS: GraspPhase.OPEN_HAND,
        DualPhase.RETREAT: GraspPhase.RETREAT,
        DualPhase.DONE: GraspPhase.DONE,
    }[p]


def _dual_right_palm_ik_goal(
    gp: GraspPhase,
    *,
    p_stow: np.ndarray,
    p_above: np.ndarray,
    p_touch: np.ndarray,
    sim_t: float,
    duration: float,
    post_spans: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Piecewise Cartesian blend; ``post_spans`` time-stamps gated post-contact segments."""
    b = _dual_phase_boundaries(duration)
    if gp == GraspPhase.STOW:
        return p_stow.copy()
    if gp == GraspPhase.APPROACH_ABOVE_BOX:
        u = _smoothstep01(
            (sim_t - b["t_stow_end"]) / max(b["t_approach_end"] - b["t_stow_end"], 1e-9)
        )
        return (1.0 - u) * p_stow + u * p_above
    if gp == GraspPhase.DESCEND_TO_PREGRASP:
        u = _smoothstep01(
            (sim_t - b["t_approach_end"])
            / max(b["t_descend_end"] - b["t_approach_end"], 1e-9)
        )
        return (1.0 - u) * p_above + u * p_touch
    if gp in (GraspPhase.PRE_GRASP_HOLD, GraspPhase.CLOSE_HAND, GraspPhase.GRASP_HOLD):
        return p_touch.copy()
    if gp == GraspPhase.LIFT_TEST:
        if post_spans is not None:
            ts0, ts1 = post_spans["lift"]
            dt = float(max(ts1 - ts0, 1e-9))
            u = _smoothstep01((sim_t - ts0) / dt)
        else:
            u = _smoothstep01(
                (sim_t - b["t_grasp_hold_end"])
                / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
            )
        return p_touch + np.array([0.0, 0.0, LIFT_PALM_DELTA_Z_M * u], dtype=float)
    if gp == GraspPhase.LOWER_BACK:
        if post_spans is not None:
            tl0, tl1 = post_spans["lower"]
            dtl = float(max(tl1 - tl0, 1e-9))
            u_lo = _smoothstep01((sim_t - tl0) / dtl)
            dz_peak = LIFT_PALM_DELTA_Z_M
            dz = dz_peak * (1.0 - u_lo)
            return p_touch + np.array([0.0, 0.0, dz], dtype=float)
        u_lo = _smoothstep01(
            (sim_t - b["t_lift_end"]) / max(b["t_lower_end"] - b["t_lift_end"], 1e-9)
        )
        u_hi = _smoothstep01(
            (sim_t - b["t_grasp_hold_end"])
            / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
        )
        dz_end = LIFT_PALM_DELTA_Z_M * u_hi
        dz = dz_end * (1.0 - u_lo)
        return p_touch + np.array([0.0, 0.0, dz], dtype=float)
    if gp == GraspPhase.OPEN_HAND:
        return p_touch.copy()
    if gp == GraspPhase.RETREAT:
        if post_spans is not None:
            tr0, tr1 = post_spans["retreat"]
            dtr = float(max(tr1 - tr0, 1e-9))
            u = _smoothstep01((sim_t - tr0) / dtr)
        else:
            u = _smoothstep01(
                (sim_t - b["t_open_end"])
                / max(b["t_retreat_end"] - b["t_open_end"], 1e-9)
            )
        return (1.0 - u) * p_touch + u * p_above
    return p_stow.copy()


LEFT_TARGET_MODE_DUAL_OUTSIDE_EDGES = "dual_outside_y_faces_near_strip_v2"


def _left_palm_ik_goal(
    phase: DualPhase,
    *,
    p_stow: np.ndarray,
    p_above: np.ndarray,
    p_touch: np.ndarray,
    sim_t: float,
    duration: float,
    post_spans: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    gp = _grasp_phase_for_proxy(phase)
    b = _dual_phase_boundaries(duration)
    gb = {
        "t_stow_end": b["t_stow_end"],
        "t_approach_end": b["t_approach_end"],
        "t_descend_end": b["t_descend_end"],
        "t_prehold_end": b["t_contact_hold_end"],
        "t_close_hand_end": b["t_close_proxy_end"],
        "t_grasp_hold_end": b["t_grasp_hold_end"],
        "t_lift_end": b["t_lift_end"],
        "t_lower_end": b["t_lower_end"],
        "t_open_end": b["t_open_end"],
        "t_retreat_end": b["t_retreat_end"],
    }

    if gp == GraspPhase.STOW:
        return p_stow.copy()
    if gp == GraspPhase.APPROACH_ABOVE_BOX:
        u = _smoothstep01(
            (sim_t - gb["t_stow_end"])
            / max(gb["t_approach_end"] - gb["t_stow_end"], 1e-9)
        )
        return (1.0 - u) * p_stow + u * p_above
    if gp == GraspPhase.DESCEND_TO_PREGRASP:
        u = _smoothstep01(
            (sim_t - gb["t_approach_end"])
            / max(gb["t_descend_end"] - gb["t_approach_end"], 1e-9)
        )
        return (1.0 - u) * p_above + u * p_touch
    if gp in (GraspPhase.PRE_GRASP_HOLD, GraspPhase.CLOSE_HAND, GraspPhase.GRASP_HOLD):
        return p_touch.copy()
    if gp == GraspPhase.LIFT_TEST:
        if post_spans is not None:
            ts0, ts1 = post_spans["lift"]
            dt = float(max(ts1 - ts0, 1e-9))
            u = _smoothstep01((sim_t - ts0) / dt)
        else:
            u = _smoothstep01(
                (sim_t - gb["t_grasp_hold_end"])
                / max(gb["t_lift_end"] - gb["t_grasp_hold_end"], 1e-9)
            )
        return p_touch + np.array([0.0, 0.0, LIFT_PALM_DELTA_Z_M * u], dtype=float)
    if gp == GraspPhase.LOWER_BACK:
        if post_spans is not None:
            tl0, tl1 = post_spans["lower"]
            dtl = float(max(tl1 - tl0, 1e-9))
            u_lo = _smoothstep01((sim_t - tl0) / dtl)
            dz_peak = LIFT_PALM_DELTA_Z_M
            dz = dz_peak * (1.0 - u_lo)
            return p_touch + np.array([0.0, 0.0, dz], dtype=float)
        u_lo = _smoothstep01(
            (sim_t - gb["t_lift_end"])
            / max(gb["t_lower_end"] - gb["t_lift_end"], 1e-9)
        )
        u_hi = _smoothstep01(
            (sim_t - gb["t_grasp_hold_end"])
            / max(gb["t_lift_end"] - gb["t_grasp_hold_end"], 1e-9)
        )
        dz_end = LIFT_PALM_DELTA_Z_M * u_hi
        dz = dz_end * (1.0 - u_lo)
        return p_touch + np.array([0.0, 0.0, dz], dtype=float)
    if gp == GraspPhase.OPEN_HAND:
        return p_touch.copy()
    if gp == GraspPhase.RETREAT:
        if post_spans is not None:
            tr0, tr1 = post_spans["retreat"]
            dtr = float(max(tr1 - tr0, 1e-9))
            u = _smoothstep01((sim_t - tr0) / dtr)
        else:
            u = _smoothstep01(
                (sim_t - gb["t_open_end"])
                / max(gb["t_retreat_end"] - gb["t_open_end"], 1e-9)
            )
        return (1.0 - u) * p_touch + u * p_above
    return p_stow.copy()


def _assist_desired_xyz_dual(
    phase: DualPhase,
    *,
    anchor: np.ndarray,
    sim_t: float,
    duration: float,
    post_spans: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Like ``run_g1_grasp_box._assist_desired_xyz`` but uses dual phase time boundaries."""
    gp = _grasp_phase_for_proxy(phase)
    b = _dual_phase_boundaries(duration)
    xy = anchor[:2]
    az = float(anchor[2])
    dz_max = float(LIFT_ASSIST_TARGET_DZ_M)
    if gp == GraspPhase.GRASP_HOLD:
        return np.array([xy[0], xy[1], az], dtype=float)
    if gp == GraspPhase.LIFT_TEST:
        if post_spans is not None:
            ts0, ts1 = post_spans["lift"]
            dt = float(max(ts1 - ts0, 1e-9))
            u = _smoothstep01((sim_t - ts0) / dt)
        else:
            u = _smoothstep01(
                (sim_t - b["t_grasp_hold_end"])
                / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
            )
        return np.array([xy[0], xy[1], az + dz_max * u], dtype=float)
    if gp == GraspPhase.LOWER_BACK:
        if post_spans is not None:
            tl0, tl1 = post_spans["lower"]
            dtl = float(max(tl1 - tl0, 1e-9))
            u_lo = _smoothstep01((sim_t - tl0) / dtl)
            z_top = az + dz_max
            return np.array([xy[0], xy[1], (1.0 - u_lo) * z_top + u_lo * az], dtype=float)
        u_lo = _smoothstep01(
            (sim_t - b["t_lift_end"]) / max(b["t_lower_end"] - b["t_lift_end"], 1e-9)
        )
        z_top = az + dz_max
        return np.array([xy[0], xy[1], (1.0 - u_lo) * z_top + u_lo * az], dtype=float)
    return np.asarray(anchor, dtype=float).copy()


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_dual_scene_model() -> mujoco.MjModel:
    if not REACH_BOX_DUAL_MILESTONE_SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing scene file: {REACH_BOX_DUAL_MILESTONE_SCENE_PATH}")
    xml_text = _inject_box_vertical_slide_for_grasp(REACH_BOX_DUAL_MILESTONE_SCENE_PATH.read_text())
    scene_root = REACH_BOX_DUAL_MILESTONE_SCENE_PATH.resolve().parent
    fd, tmp_name = tempfile.mkstemp(suffix=".xml", dir=scene_root, text=True)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(xml_text)
        with _chdir(scene_root):
            model = mujoco.MjModel.from_xml_path(tmp_path.name)
    finally:
        tmp_path.unlink(missing_ok=True)

    return model


def _box_geom_id(model: mujoco.MjModel) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOX_GEOM_NAME)
    if gid >= 0:
        return int(gid)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEGACY_BOX_GEOM_NAME)
    if gid < 0:
        raise RuntimeError(f"Neither {BOX_GEOM_NAME} nor {LEGACY_BOX_GEOM_NAME} found")
    return int(gid)


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


def _build_dual_ik_metadata(model: mujoco.MjModel) -> tuple[list[int], np.ndarray, np.ndarray]:
    qadrs: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for jn in DUAL_IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        qadrs.append(int(model.jnt_qposadr[jid]))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qadrs, np.asarray(lows), np.asarray(highs)


def _build_named_chain_metadata(
    model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    qadrs: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        qadrs.append(int(model.jnt_qposadr[jid]))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qadrs, np.asarray(lows), np.asarray(highs)


def _dual_ik_dof_adrs(model: mujoco.MjModel) -> list[int]:
    out: list[int] = []
    for jn in DUAL_IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        out.append(int(model.jnt_dofadr[jid]))
    return out


def _apply_chain_positions(
    model: mujoco.MjModel, data: mujoco.MjData, adrs: list[int], vals: np.ndarray
) -> None:
    for i, adr in enumerate(adrs):
        data.qpos[adr] = float(vals[i])


def _enforce_waist_qpos(
    data: mujoco.MjData,
    waist_qpos_adrs: tuple[int, int],
    lock_xy: np.ndarray | None,
) -> None:
    if lock_xy is None:
        return
    data.qpos[int(waist_qpos_adrs[0])] = float(lock_xy[0])
    data.qpos[int(waist_qpos_adrs[1])] = float(lock_xy[1])


def _snapshot_qpos(data: mujoco.MjData, adrs: list[int]) -> dict[int, float]:
    return {int(a): float(data.qpos[int(a)]) for a in adrs}


def _restore_qpos(data: mujoco.MjData, snap: dict[int, float]) -> None:
    for a, v in snap.items():
        data.qpos[int(a)] = float(v)


def _solve_chain_with_recovery(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    data: mujoco.MjData,
    *,
    ik_adrs: list[int],
    q_low: np.ndarray,
    q_high: np.ndarray,
    q_neutral: np.ndarray,
    body_id: int,
    target_xyz: np.ndarray,
    site_id: int,
    posture_gain: float,
    max_abs_dn: float,
    inner_iters: int,
    ori_kw: dict[str, Any],
    position_gain: float,
    err_reject: float,
    reject_streak: list[int] | None = None,
    recovery_event_counter: list[int] | None = None,
    rejected_update_counter: list[int] | None = None,
) -> tuple[float, bool]:
    snap = _snapshot_qpos(data, ik_adrs)
    q_new, err = solve_ik_q(
        model,
        fd,
        data,
        ik_adrs,
        q_low,
        q_high,
        body_id,
        target_xyz,
        q_neutral,
        posture_gain=posture_gain,
        max_abs_joint_from_neutral=max_abs_dn,
        target_site_id=site_id if site_id >= 0 else -1,
        inner_iters=inner_iters,
        position_task_gain=position_gain,
        **ori_kw,
    )
    rec = False
    err_f = float(err)
    thr = float(err_reject)
    if reject_streak is None:
        if err_f > thr:
            _restore_qpos(data, snap)
            rec = True
        else:
            _apply_chain_positions(model, data, ik_adrs, q_new)
    else:
        if err_f <= thr:
            _apply_chain_positions(model, data, ik_adrs, q_new)
            reject_streak[0] = 0
        else:
            _restore_qpos(data, snap)
            if rejected_update_counter is not None:
                rejected_update_counter[0] = int(rejected_update_counter[0]) + 1
            reject_streak[0] = int(reject_streak[0]) + 1
            if reject_streak[0] >= IK_REJECT_DEBOUNCE_FRAMES:
                reject_streak[0] = 0
                if recovery_event_counter is not None:
                    recovery_event_counter[0] = int(recovery_event_counter[0]) + 1
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(err), rec


def _scale_palm_ori_kw(kw: dict[str, Any], mul: float) -> dict[str, Any]:
    if not kw:
        return kw
    out = dict(kw)
    for key in ("palm_axis_task_gain", "palm_up_task_gain"):
        if key in out:
            out[key] = float(out[key]) * float(mul)
    return out


def _drift_left_arm_to_neutral(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    l_arm_adrs: list[int],
    q_neutral_l_arm_vals: np.ndarray,
    alpha: float = LEFT_ARM_NEUTRAL_DRIFT_ALPHA,
) -> None:
    for adr, nv in zip(l_arm_adrs, q_neutral_l_arm_vals, strict=True):
        qv = float(data.qpos[int(adr)])
        data.qpos[int(adr)] = (1.0 - alpha) * qv + alpha * float(nv)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _posture_scale_coord(mode: CoordinationMode) -> tuple[float, float]:
    """Returns (right_scale, left_scale) posture multipliers."""
    if mode == CoordinationMode.RIGHT_PRIMARY_WAIST_FREE:
        return 1.0, 1.0
    if mode == CoordinationMode.WAIST_LOCKED_DUAL_ARM:
        return 0.82, 0.72
    return 1.0, 1.0


def _solve_primary_support_dual_arms(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    data: mujoco.MjData,
    *,
    coordination_mode: CoordinationMode,
    phase: DualPhase,
    bilateral_stable_s: float,
    waist_qpos_adrs: tuple[int, int],
    waist_lock_xy: np.ndarray | None,
    engage_left: bool,
    r_full_adrs: list[int],
    r_full_low: np.ndarray,
    r_full_high: np.ndarray,
    q_neutral_r_full: np.ndarray,
    r_arm_adrs: list[int],
    r_arm_low: np.ndarray,
    r_arm_high: np.ndarray,
    q_neutral_r_arm: np.ndarray,
    l_arm_adrs: list[int],
    l_arm_low: np.ndarray,
    l_arm_high: np.ndarray,
    q_neutral_l_arm: np.ndarray,
    q_neutral_l_arm_vals: np.ndarray,
    target_right: np.ndarray,
    target_left: np.ndarray,
    right_site_id: int,
    right_body_id: int,
    left_site_id: int,
    left_body_id: int,
    posture_gain: float,
    max_abs_dn: float,
    ori_recovery_scale: float,
    palm_ori_box_center: np.ndarray | None = None,
    palm_right_pre_ik_xyz: np.ndarray | None = None,
    palm_left_pre_ik_xyz: np.ndarray | None = None,
    wrist_orientation_lock: bool = False,
    wrist_frozen_kw_r: dict[str, Any] | None = None,
    wrist_frozen_kw_l: dict[str, Any] | None = None,
    ik_inner_iters_scale: float = 1.0,
    right_reject_streak: list[int],
    left_reject_streak: list[int],
    right_recovery_events: list[int],
    left_recovery_events: list[int],
    right_ik_rejects: list[int],
    left_ik_rejects: list[int],
) -> tuple[float, float, bool, bool]:
    """Right arm primary (full chain until waist freeze), left arm support without waist ownership."""
    sr, sl_base = _posture_scale_coord(coordination_mode)
    pg_r = RIGHT_POSITION_TASK_GAIN
    pg_l = LEFT_POSITION_TASK_GAIN * SUPPORT_LEFT_POSITION_GAIN_MUL
    sl = sl_base * SUPPORT_LEFT_POSTURE_SCALE
    err_r = err_l = 0.0
    rec_r = rec_l = False
    it_scale = float(max(0.35, min(float(ik_inner_iters_scale), 1.0)))

    def okw(side: str) -> dict[str, Any]:
        if wrist_orientation_lock:
            fk = wrist_frozen_kw_r if side == "right" else wrist_frozen_kw_l
            if fk:
                base = dict(fk)
                if side == "left":
                    return _scale_palm_ori_kw(base, SUPPORT_LEFT_ORI_MUL)
                return base
        pw = palm_right_pre_ik_xyz if side == "right" else palm_left_pre_ik_xyz
        kw = _palm_ori_kw_scaled(
            phase,
            side=side,
            ori_recovery_scale=ori_recovery_scale,
            bilateral_stable_s=bilateral_stable_s,
            box_center=palm_ori_box_center,
            palm_site_xyz=pw,
        )
        if side == "left":
            return _scale_palm_ori_kw(kw, SUPPORT_LEFT_ORI_MUL)
        return kw

    if coordination_mode == CoordinationMode.IDLE:
        return 0.0, 0.0, False, False

    def _solve_left_support() -> tuple[float, bool]:
        ori_l = okw("left")
        return _solve_chain_with_recovery(
            model,
            fd,
            data,
            ik_adrs=l_arm_adrs,
            q_low=l_arm_low,
            q_high=l_arm_high,
            q_neutral=q_neutral_l_arm,
            body_id=left_body_id,
            target_xyz=target_left,
            site_id=left_site_id,
            posture_gain=posture_gain * sl,
            max_abs_dn=max_abs_dn,
            inner_iters=max(3, int(float(IK_ITERS_LEFT_SUPPORT) * it_scale)),
            ori_kw=ori_l,
            position_gain=pg_l,
            err_reject=IK_ERR_REJECT_LEFT_SUPPORT_M,
            reject_streak=left_reject_streak,
            recovery_event_counter=left_recovery_events,
            rejected_update_counter=left_ik_rejects,
        )

    if coordination_mode == CoordinationMode.RIGHT_PRIMARY_WAIST_FREE:
        err_r, rec_r = _solve_chain_with_recovery(
            model,
            fd,
            data,
            ik_adrs=r_full_adrs,
            q_low=r_full_low,
            q_high=r_full_high,
            q_neutral=q_neutral_r_full,
            body_id=right_body_id,
            target_xyz=target_right,
            site_id=right_site_id,
            posture_gain=posture_gain * sr,
            max_abs_dn=max_abs_dn,
            inner_iters=max(3, int(float(IK_ITERS_RIGHT_PRIMARY_FULL) * it_scale)),
            ori_kw=okw("right"),
            position_gain=pg_r,
            err_reject=IK_ERR_REJECT_M,
            reject_streak=right_reject_streak,
            recovery_event_counter=right_recovery_events,
            rejected_update_counter=right_ik_rejects,
        )
        if engage_left:
            err_l, rec_l = _solve_left_support()
        else:
            _drift_left_arm_to_neutral(
                model,
                data,
                l_arm_adrs=l_arm_adrs,
                q_neutral_l_arm_vals=q_neutral_l_arm_vals,
            )
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return float(err_r), float(err_l), rec_r, rec_l

    if coordination_mode == CoordinationMode.WAIST_LOCKED_DUAL_ARM:
        if waist_lock_xy is None:
            raise RuntimeError("WAIST_LOCKED_DUAL_ARM requires waist_lock_xy")
        _enforce_waist_qpos(data, waist_qpos_adrs, waist_lock_xy)
        mujoco.mj_forward(model, data)
        err_r, rec_r = _solve_chain_with_recovery(
            model,
            fd,
            data,
            ik_adrs=r_arm_adrs,
            q_low=r_arm_low,
            q_high=r_arm_high,
            q_neutral=q_neutral_r_arm,
            body_id=right_body_id,
            target_xyz=target_right,
            site_id=right_site_id,
            posture_gain=posture_gain * sr,
            max_abs_dn=max_abs_dn,
            inner_iters=max(3, int(float(IK_ITERS_RIGHT_ARM_LOCKED) * it_scale)),
            ori_kw=okw("right"),
            position_gain=pg_r,
            err_reject=IK_ERR_REJECT_M,
            reject_streak=right_reject_streak,
            recovery_event_counter=right_recovery_events,
            rejected_update_counter=right_ik_rejects,
        )
        _enforce_waist_qpos(data, waist_qpos_adrs, waist_lock_xy)
        mujoco.mj_forward(model, data)
        if engage_left:
            err_l, rec_l = _solve_left_support()
        else:
            _drift_left_arm_to_neutral(
                model,
                data,
                l_arm_adrs=l_arm_adrs,
                q_neutral_l_arm_vals=q_neutral_l_arm_vals,
            )
        _enforce_waist_qpos(data, waist_qpos_adrs, waist_lock_xy)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return float(err_r), float(err_l), rec_r, rec_l

    return float(err_r), float(err_l), rec_r, rec_l


def _contact_count_set(data: mujoco.MjData, box_gid: int, touch_gids: set[int]) -> int:
    n = 0
    for cid in range(data.ncon):
        contact = data.contact[cid]
        if box_gid in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == box_gid else contact.geom1
            if other in touch_gids:
                n += 1
    return n


def _min_geom_distance_to_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_gid: int,
    touch_gids: set[int],
    dist_cap: float = 0.25,
) -> float:
    """Minimum MuJoCo convex distance between ``box_geom`` and any monitored manipulator geom."""
    fromto = np.zeros(6, dtype=np.float64)
    best = float("inf")
    bg = int(box_gid)
    for gid in touch_gids:
        if gid < 0:
            continue
        d = float(
            mujoco.mj_geomDistance(model, data, int(gid), bg, float(dist_cap), fromto)
        )
        best = min(best, d)
    return float(best)


def _model_uses_legacy_proxy_fingers(model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_proxy_left_finger_geom") >= 0


def _dual_overlap_monitor_geom_ids(model: mujoco.MjModel, *, side: str) -> set[int]:
    """Palm + wrist (+ hand capsule if present) for cross-arm clearance; not every fingertip mesh."""
    if _model_uses_legacy_proxy_fingers(model):
        names = LEFT_PROXY_GEOMS if side == "left" else RIGHT_PROXY_GEOMS
        return _proxy_geom_ids(model, names)
    out: set[int] = set()
    for nm in (
        f"{side}_palm_contact_geom",
        f"{side}_wrist_collision",
        f"{side}_hand_collision",
    ):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g >= 0:
            out.add(int(g))
    return out


def _left_touch_geom_ids(model: mujoco.MjModel) -> set[int]:
    if not _model_uses_legacy_proxy_fingers(model):
        return set(enumerate_hand_contact_geoms(model, side="left"))
    gids: set[int] = set()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_PALM_GEOM)
    if gid >= 0:
        gids.add(int(gid))
    for nm in LEFT_PROXY_GEOMS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g >= 0:
            gids.add(int(g))
    if not gids:
        for nm in LEFT_TOUCH_FALLBACK:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if g >= 0:
                gids.add(int(g))
    return gids


def _is_palm_inside_box_volume(
    palm_xyz: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    eps: float = 1e-5,
) -> bool:
    x, y, z = float(palm_xyz[0]), float(palm_xyz[1]), float(palm_xyz[2])
    return bool(
        xmin + eps < x < xmax - eps
        and ymin + eps < y < ymax - eps
        and zmin + eps < z < zmax - eps
    )


def _is_proxy_crossing_box_center(
    palm_xyz: np.ndarray,
    box_center_y: float,
    *,
    side: str,
    eps: float = 1e-5,
) -> bool:
    y = float(palm_xyz[1])
    cy = float(box_center_y)
    if side == "right":
        return bool(y > cy + eps)
    return bool(y < cy - eps)


def _is_cross_arm_intersection_risk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    left_gids: set[int],
    right_gids: set[int],
    dist_cap: float = 0.12,
) -> bool:
    fromto = np.zeros(6, dtype=np.float64)
    best = float("inf")
    for gl in left_gids:
        for gr in right_gids:
            if gl < 0 or gr < 0:
                continue
            d = float(
                mujoco.mj_geomDistance(model, data, int(gl), int(gr), float(dist_cap), fromto)
            )
            best = min(best, d)
    return bool(best < PROXY_CROSS_ARM_SEP_WARN_M)


def _site_axis_world(data: mujoco.MjData, site_id: int, *, axis_col: int = 0) -> np.ndarray:
    R = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    v = np.asarray(R[:, int(axis_col)], dtype=float).reshape(3,)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-12 else v


def _normal_alignment_score(axis_world: np.ndarray, desired_world: np.ndarray) -> float:
    a = np.asarray(axis_world, dtype=float).reshape(3,)
    b = np.asarray(desired_world, dtype=float).reshape(3,)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))


def _palm_normal_alignment_score(
    data: mujoco.MjData,
    site_id: int,
    desired_world: np.ndarray,
) -> float:
    """Score whichever palm-site ±axis best represents the side-facing palm normal."""
    desired = np.asarray(desired_world, dtype=float).reshape(3,)
    best = -1.0
    for col in (0, 1, 2):
        ax = _site_axis_world(data, site_id, axis_col=col)
        best = max(best, _normal_alignment_score(ax, desired))
        best = max(best, _normal_alignment_score(-ax, desired))
    return float(best)


def _palm_up_alignment_score(data: mujoco.MjData, site_id: int) -> float:
    """How well some palm-local axis aligns with world +Z (prefer palm ``up'' not pitching down)."""
    best = 0.0
    for col in (1, 2):
        ax = _site_axis_world(data, site_id, axis_col=col)
        best = max(best, abs(_normal_alignment_score(ax, WORLD_UP)))
    return float(best)


def _palm_ori_phase_gain_mul(phase: DualPhase, *, bilateral_stable_s: float) -> float:
    if phase == DualPhase.DUAL_CONTACT_HOLD:
        if float(bilateral_stable_s) >= ORI_CONTACT_HOLD_STABLE_BILATERAL_S:
            return float(ORI_GAIN_MUL_CONTACT_HOLD_STABLE)
        return float(ORI_GAIN_MUL_CONTACT_HOLD_UNSTABLE)
    if phase in (DualPhase.DUAL_APPROACH, DualPhase.DUAL_DESCEND_TO_PREGRASP):
        return 0.74
    return 1.0


def _palm_ori_kw_scaled(
    phase: DualPhase,
    *,
    side: str,
    ori_recovery_scale: float = 1.0,
    bilateral_stable_s: float = 0.0,
    box_center: np.ndarray | None = None,
    palm_site_xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    if phase not in ORIENTATION_SOFT_PHASES:
        return {}
    default_axis = RIGHT_PALM_AXIS_TARGET if side == "right" else LEFT_PALM_AXIS_TARGET
    axis_target = np.asarray(default_axis, dtype=float).copy()
    if (
        box_center is not None
        and palm_site_xyz is not None
        and phase
        in (
            DualPhase.DUAL_APPROACH,
            DualPhase.DUAL_DESCEND_TO_PREGRASP,
            DualPhase.DUAL_CONTACT_HOLD,
        )
    ):
        bc = np.asarray(box_center, dtype=float).reshape(3)
        pw = np.asarray(palm_site_xyz, dtype=float).reshape(3)
        v = bc - pw
        v[2] *= 0.20
        nv = float(np.linalg.norm(v))
        if nv > 1e-6:
            face_n = v / nv
            if float(np.dot(face_n, default_axis)) < 0.0:
                face_n = -face_n
            blend = 0.58 * face_n + 0.42 * default_axis
            nb = float(np.linalg.norm(blend))
            if nb > 1e-9:
                axis_target = blend / nb
    ori_mul = RIGHT_ORI_GAIN_MUL if side == "right" else LEFT_ORI_GAIN_MUL
    scl = float(ori_recovery_scale)
    phase_mul = float(_palm_ori_phase_gain_mul(phase, bilateral_stable_s=bilateral_stable_s))
    if phase in (DualPhase.DUAL_APPROACH, DualPhase.DUAL_DESCEND_TO_PREGRASP):
        axis_soft = 0.40
        up_soft = 0.62
    elif phase == DualPhase.DUAL_CONTACT_HOLD:
        axis_soft = 0.52
        up_soft = 0.72
    else:
        axis_soft = 0.58
        up_soft = 0.66
    ga = PALM_AXIS_GAIN_BASE * ORI_AUTHORITY_SCALE * ori_mul * scl * phase_mul * axis_soft
    gu = PALM_UP_GAIN_BASE * ORI_AUTHORITY_SCALE * ori_mul * scl * phase_mul * up_soft
    return {
        "palm_axis_target_world": np.asarray(axis_target, dtype=float).copy(),
        "palm_axis_task_gain": float(ga),
        "palm_up_target_world": WORLD_UP.copy(),
        "palm_up_task_gain": float(gu),
    }


def _dex3_digit_policy(
    *,
    phase: DualPhase,
    side: str,
    side_geometry_ok: bool,
    palm_aligned: bool,
    palm_up_ok: bool,
    palm_outside: bool,
    palm_has_contact: bool,
    wrist_orientation_locked: bool,
    penetration_m: float,
    penetration_prev_m: float,
    digit_distances: dict[str, float],
    digit_contacts: dict[str, int],
    safety_stop: bool,
    finger_place_only: bool = False,
) -> dict[str, str]:
    """Per-digit decisions: approach box contact points, then hold (no wrist re-aim)."""
    out: dict[str, str] = {}
    freeze_dist = float(DEX3_DIGIT_CONTACT_FREEZE_DIST_M)
    active_phase = phase in (
        DualPhase.DUAL_CONTACT_HOLD,
        DualPhase.CLOSE_PROXY_HANDS,
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK,
    )
    close_allowed = bool(
        active_phase
        and side_geometry_ok
        and palm_aligned
        and palm_up_ok
        and palm_outside
        and not safety_stop
        and float(penetration_m) <= float(DEX3_DIGIT_OPEN_PENETRATION_M)
    )
    pen_rising = float(penetration_m) > float(penetration_prev_m) + 0.0012

    def _digit_touching(role: str) -> bool:
        return int(digit_contacts.get(role, 0)) > 0 or float(
            digit_distances.get(role, float("inf"))
        ) <= freeze_dist

    def _emit_finger_place() -> None:
        """Slow approach; lock each digit at first box contact (stay on contact point)."""
        may_close = bool(
            close_allowed
            or (
                side_geometry_ok
                and palm_outside
                and not safety_stop
                and float(penetration_m) <= float(DEX3_DIGIT_OPEN_PENETRATION_M)
            )
        )
        for role in ("thumb", "index", "middle"):
            key = f"{side}_{role}"
            if safety_stop or float(penetration_m) > float(DEX3_CLEAN_PENETRATION_CAP_M):
                out[key] = "open"
                continue
            if _digit_touching(role):
                out[key] = "hold"
                continue
            d = float(digit_distances.get(role, float("inf")))
            if pen_rising and d < 0.025:
                out[key] = "hold"
            elif may_close and d > float(DEX3_DIGIT_HOLD_NEAR_BOX_M) * 0.85:
                out[key] = "close"
            else:
                out[key] = "hold" if active_phase else "open"

    def _emit_open_hold_close_spread() -> None:
        """Stage A: bring digits toward the box without an early pinch (thumb lags)."""
        for role in ("thumb", "index", "middle"):
            key = f"{side}_{role}"
            if safety_stop or float(penetration_m) > float(DEX3_CLEAN_PENETRATION_CAP_M) or pen_rising:
                out[key] = "open"
                continue
            if int(digit_contacts.get(role, 0)) > 0:
                out[key] = "hold"
                continue
            d = float(digit_distances.get(role, float("inf")))
            near = float(DEX3_DIGIT_HOLD_NEAR_BOX_M)
            if role == "middle":
                near *= 1.35
            if d <= near:
                out[key] = "hold"
            elif close_allowed:
                if role == "thumb" and d > 0.085:
                    out[key] = "close"
                elif role == "thumb":
                    out[key] = "hold"
                else:
                    out[key] = "close"
            else:
                out[key] = "hold" if active_phase else "open"

    def _emit_stage_b() -> None:
        """Stage B: thumb stabilizes side, middle supports, index does final closure."""
        for role in ("thumb", "index", "middle"):
            key = f"{side}_{role}"
            if safety_stop or float(penetration_m) > float(DEX3_CLEAN_PENETRATION_CAP_M) or pen_rising:
                out[key] = "open"
                continue
            d = float(digit_distances.get(role, float("inf")))
            dc = int(digit_contacts.get(role, 0))
            if role == "thumb":
                if dc > 0 or d <= 0.022:
                    out[key] = "hold"
                elif close_allowed:
                    out[key] = "close"
                else:
                    out[key] = "hold"
            elif role == "middle":
                if dc > 0 and wrist_orientation_locked:
                    out[key] = "hold"
                elif close_allowed:
                    out[key] = "close"
                else:
                    out[key] = "hold" if active_phase else "open"
            else:  # index — allow refinement even under mild penetration slack
                if close_allowed or (active_phase and d < 0.09):
                    out[key] = "close"
                elif dc > 0:
                    out[key] = "hold"
                else:
                    out[key] = "hold" if active_phase else "open"

    if not active_phase:
        for role in ("thumb", "index", "middle"):
            out[f"{side}_{role}"] = "open"
        return out

    if finger_place_only and active_phase:
        _emit_finger_place()
        return out

    if palm_has_contact:
        _emit_stage_b()
        return out

    if close_allowed or active_phase:
        _emit_open_hold_close_spread()
        return out

    for role in ("thumb", "index", "middle"):
        out[f"{side}_{role}"] = "open"
    return out


def _proxy_geom_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> set[int]:
    gids: set[int] = set()
    for nm in names:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g >= 0:
            gids.add(int(g))
    return gids


def _right_touch_geom_ids_box(model: mujoco.MjModel) -> tuple[set[int], str]:
    if not _model_uses_legacy_proxy_fingers(model):
        return (
            set(enumerate_hand_contact_geoms(model, side="right")),
            "dex3_hand_collision_geoms",
        )
    gids, mode = _right_touch_contact_gids_grasp(model)
    return gids, mode


_ORDERED_PENETRATION_DIAG_KEYS = (
    "DUAL_DESCEND_TO_PREGRASP",
    "DUAL_CONTACT_HOLD",
    "CLOSE_PROXY_HANDS",
    "DUAL_GRASP_HOLD",
    "DUAL_LIFT_TEST",
)


def _print_penetration_phase_report(
    phase_peak: dict[str, dict[str, Any]],
    *,
    dual_extra_clearance_x_m: float,
) -> None:
    """Human-readable peak penetration snapshot per DualPhase (debug only)."""
    print(
        "\n----- penetration diagnostics -----\n"
        f"DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M (dual IK −x margin): {dual_extra_clearance_x_m:.5f}\n"
        "Peak box-AABB penetration per phase (leader geom = argmax monitored geoms that frame):\n",
        flush=True,
    )
    for key in _ORDERED_PENETRATION_DIAG_KEYS:
        rec = phase_peak.get(key)
        if rec is None:
            print(f"  {key}: (no samples)", flush=True)
            continue
        nm = str(rec["leader_geom"])
        d_ik = float(rec["right_site_to_target_m"])
        rsite = rec["palm_site_xyz"]
        gctr = rec["leader_geom_center_xyz"]
        nfx = float(rec["near_face_x"])
        plat_x = rec.get("right_plate_max_world_x")
        plat_s = f"{float(plat_x):.5f}" if plat_x is not None and np.isfinite(float(plat_x)) else "n/a"
        lx = rec["geom_local_x_world_unit"]
        hint = ""
        if nm == RIGHT_PALM_GEOM:
            if d_ik < 0.025 and float(rec["max_penetration_m"]) > 0.003:
                hint = "likely_orientation_or_plate_geometry (small IK error)"
            elif d_ik >= 0.025:
                hint = "possible_IK_position_overshoot"
            else:
                hint = "check_thickness_and_corners"
        print(
            f"  {key}:\n"
            f"    max_penetration_any_geom_m={float(rec['max_penetration_m']):.6f}  "
            f"leader={nm}  leader_pen_m={float(rec['leader_penetration_m']):.6f}  t={float(rec['sim_time']):.3f}s\n"
            f"    palm_site_xyz={rsite}  geom_center_xyz={gctr}\n"
            f"    near_face_x={nfx:.5f}  box_x_min={float(rec['box_x_min']):.5f}  "
            f"right_plate_max_world_x={plat_s}\n"
            f"    geom_local+x_world_unit={lx}\n"
            f"    right_site_to_ik_target_m={d_ik:.5f}  left_site_to_ik_target_m="
            f"{float(rec['left_site_to_target_m']):.5f}  right_normal_alignment={float(rec['right_normal_alignment']):.5f}\n"
            f"    site_minus_geom_center={rec['site_minus_geom_center']}  hint={hint}",
            flush=True,
        )


def _detect_box_dual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_geom_name: str,
) -> dict[str, Any]:
    """Ground-truth box pose with dual-arm palm centers biased farther −x for geom-corner penetration margin."""
    return G1BoxPerception.detect_box(
        model,
        data,
        box_geom_name=box_geom_name,
        dual_near_face_extra_clearance_x_m=DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M,
    )


def _print_episode_stability_summary(out: dict[str, Any]) -> None:
    """Short post-run stability metrics (stdout)."""
    ar = out.get("avg_right_palm_vel_near_contact_m_per_s")
    al = out.get("avg_left_palm_vel_near_contact_m_per_s")
    ar_s = f"{float(ar):.4f}" if ar is not None and np.isfinite(float(ar)) else "n/a"
    al_s = f"{float(al):.4f}" if al is not None and np.isfinite(float(al)) else "n/a"
    print(
        "\n----- stability summary -----\n"
        f"max_joint_vel_mag_episode: {float(out.get('max_joint_vel_mag_episode', 0.0)):.4f} rad/s\n"
        f"max_per_frame_dual_cmd_delta_episode: {float(out.get('max_per_frame_dual_cmd_delta_episode', 0.0)):.5f} rad\n"
        f"ik_rejected_updates_total: {int(out.get('ik_rejected_updates_total', 0))}\n"
        f"ik_debounce_recovery_events: {int(out.get('ik_debounce_recovery_events', 0))}\n"
        f"max_waist_delta_step_rad_episode: {float(out.get('max_waist_delta_step_rad_episode', 0.0)):.5f}\n"
        f"waist_yaw_range_episode_rad: {float(out.get('waist_yaw_range_episode_rad', 0.0)):.5f}  "
        f"waist_pitch_range_episode_rad: {float(out.get('waist_pitch_range_episode_rad', 0.0)):.5f}\n"
        f"avg_right_palm_vel_near_contact_m_per_s: {ar_s}  "
        f"avg_left_palm_vel_near_contact_m_per_s: {al_s}\n"
        f"legacy_ik_recovery_count_key: {int(out.get('ik_recovery_count', 0))}",
        flush=True,
    )


def run_g1_dual_arm_box(
    *,
    headless: bool = False,
    timeout: float = DEFAULT_DUAL_ARM_TIMEOUT_S,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    quiet: bool = False,
    fast: bool = False,
    max_steps: int | None = None,
    print_every_s: float | None = None,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL + 0.40,
    dbg_markers: bool = False,
    dbg_penetration: bool = False,
    finger_place_only: bool = True,
    no_assist: bool = False,
) -> dict[str, Any]:
    if not REACH_BOX_DUAL_MILESTONE_SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing scene: {REACH_BOX_DUAL_MILESTONE_SCENE_PATH}")
    if not PIPELINE_G1_DEX3_HANDS_XML.is_file():
        raise FileNotFoundError(f"Missing Dex3 MJCF: {PIPELINE_G1_DEX3_HANDS_XML}")

    tout_req = float(timeout)
    timeout = min(tout_req, float(FAST_DUAL_ARM_TIMEOUT_CAP_S)) if fast else tout_req
    silent = bool(quiet)
    print_interval_s = float(print_every_s) if print_every_s is not None else float(PRINT_INTERVAL)
    max_steps_limit = max_steps

    model = _load_dual_scene_model()
    motion = _sim_motion_profile(fast=bool(fast))
    global _ACTIVE_TIMELINE_SPEED
    _ACTIVE_TIMELINE_SPEED = float(motion.timeline_speed)
    applied_dt = _apply_physics_timestep(model, motion.physics_timestep_s)
    use_dex3_hands = int(model.nu) == int(G1_DUAL_ARM_NU_DEX3)
    if not use_dex3_hands and int(model.nu) != 31:
        raise RuntimeError(f"Expected nu=31 (29 hinge + 2 right proxy sliders), got {model.nu}")

    hinge_names = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm:
            hinge_names.add(nm)
    neutral_eff = merge_dex3_neutral_into(dict(NEUTRAL_POSTURE), hinge_joint_names=hinge_names)
    if set(neutral_eff.keys()) != hinge_names:
        raise ValueError("Neutral posture dict must list every hinge (including Dex3 fingers).")

    hinge_addrs = build_hinge_joint_address_map(model)

    finger_joints = _has_actuated_finger_joints(model)
    assist_finger_actuators_unsupported = bool(finger_joints and not use_dex3_hands)
    finger_controller: Dex3FingerController | None = None
    if use_dex3_hands:
        finger_controller = Dex3FingerController(
            lp_alpha=float(motion.finger_lp_alpha),
            max_delta_rad=float(motion.finger_max_delta_rad),
        )
    pen_episode_cap_m = (
        float(DEX3_CLEAN_PENETRATION_CAP_M)
        if use_dex3_hands
        else float(MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M)
    )
    _apply_kp_scale_for_joint_subset(model, set(DUAL_IK_JOINT_NAMES), DUAL_IK_KP_SCALE)
    _apply_kp_scale_for_joint_subset(
        model,
        {
            "right_wrist_roll_joint",
            "right_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_yaw_joint",
        },
        DUAL_IK_KP_SCALE,
    )

    data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    actuator_ids = build_actuator_id_map(model)
    base_map = floating_base_address_map(model)
    ik_qpos_adrs, _, _ = _build_dual_ik_metadata(model)
    ik_dof_adrs = _dual_ik_dof_adrs(model)
    joint_qpos_adr = {jn: ik_qpos_adrs[i] for i, jn in enumerate(DUAL_IK_JOINT_NAMES)}
    waist_qpos_tuple = (
        int(joint_qpos_adr["waist_yaw_joint"]),
        int(joint_qpos_adr["waist_pitch_joint"]),
    )
    r_qpos_adrs, r_low, r_high = _build_named_chain_metadata(model, DUAL_RIGHT_IK_JOINT_NAMES_ONLY)
    q_neutral_r = np.array([float(neutral_eff[jn]) for jn in DUAL_RIGHT_IK_JOINT_NAMES_ONLY])
    r_arm_adrs, r_arm_low, r_arm_high = _build_named_chain_metadata(
        model, RIGHT_ARM_IK_JOINT_NAMES_ONLY
    )
    q_neutral_r_arm = np.array([float(neutral_eff[jn]) for jn in RIGHT_ARM_IK_JOINT_NAMES_ONLY])
    l_arm_adrs, l_arm_low, l_arm_high = _build_named_chain_metadata(
        model, LEFT_ARM_IK_JOINT_NAMES_ONLY
    )
    q_neutral_l_arm = np.array([float(neutral_eff[jn]) for jn in LEFT_ARM_IK_JOINT_NAMES_ONLY])
    q_neutral_l_arm_vals = q_neutral_l_arm.copy()
    l_full_adrs, l_full_low, l_full_high = _build_named_chain_metadata(
        model, DUAL_LEFT_IK_JOINT_NAMES_ONLY
    )
    q_neutral_l_full = np.array(
        [float(neutral_eff[jn]) for jn in DUAL_LEFT_IK_JOINT_NAMES_ONLY]
    )

    nominal_base_qpos = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=neutral_eff,
        base_map=base_map,
    )
    mujoco.mj_forward(model, data)

    if finger_controller is not None:
        snap = {
            jn: float(data.qpos[int(hinge_addrs[jn]["qpos_adr"])])
            for jn in finger_controller.targets_for_mode("open_hand")
        }
        finger_controller.reset(snap)

    neutral_dual_vec = np.array(
        [float(data.qpos[int(joint_qpos_adr[jn])]) for jn in DUAL_IK_JOINT_NAMES], dtype=float
    )
    cmd_dual_prev = neutral_dual_vec.copy()
    pr_goal_filt_ref: list[np.ndarray | None] = [None]
    right_reject_streak = [0]
    left_reject_streak = [0]
    right_recovery_events = [0]
    left_recovery_events = [0]
    right_ik_rejects = [0]
    left_ik_rejects = [0]
    max_joint_vel_mag_episode = 0.0
    max_per_frame_dual_cmd_delta_episode = 0.0
    min_waist_yaw_episode = float("inf")
    max_waist_yaw_episode = float("-inf")
    min_waist_pitch_episode = float("inf")
    max_waist_pitch_episode = float("-inf")
    prev_palm_r_phys: np.ndarray | None = None
    prev_palm_l_phys: np.ndarray | None = None
    right_palm_vel_near_sum = 0.0
    right_palm_vel_near_n = 0
    left_palm_vel_near_sum = 0.0
    left_palm_vel_near_n = 0
    right_arm_cmd_prev = {
        jn: float(data.qpos[int(joint_qpos_adr[jn])]) for jn in RIGHT_ARM_CMD_SMOOTH_JOINTS
    }
    left_arm_cmd_prev = {
        jn: float(data.qpos[int(joint_qpos_adr[jn])]) for jn in LEFT_ARM_CMD_SMOOTH_JOINTS
    }
    max_filtered_pr_goal_dist_episode = 0.0
    max_right_joint_command_delta_episode = 0.0
    max_right_bad_ik_streak_episode = 0
    waist_frozen_xy: np.ndarray | None = None
    right_stable_for_freeze_frames = 0
    sequential_left_arm_frames = 0
    joint_jumps_by_phase: defaultdict[str, int] = defaultdict(int)
    max_waist_delta_step_rad_episode = 0.0
    max_palm_convergence_speed_r = 0.0
    max_palm_convergence_speed_l = 0.0
    max_dual_goal_asymmetry_m = 0.0
    prev_dr_track = float("nan")
    prev_dl_track = float("nan")
    max_wrist_command_delta_episode = 0.0
    lift_phase_reached = False
    lift_assist_applied_in_lift_phase = False
    large_joint_jump_events = 0
    max_abs_joint_step_applied_episode = 0.0

    dex3_pen_right_gid_list: list[int] = (
        list(enumerate_hand_contact_geoms(model, side="right")) if use_dex3_hands else []
    )
    dex3_pen_left_gid_list: list[int] = (
        list(enumerate_hand_contact_geoms(model, side="left")) if use_dex3_hands else []
    )

    right_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_WRIST_BODY))
    left_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_WRIST_BODY))
    if right_bid < 0 or left_bid < 0:
        raise RuntimeError("Expected right/left wrist bodies")

    box_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME))
    if box_bid < 0:
        raise RuntimeError(f"Missing {BOX_BODY_NAME}")

    right_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_SITE))
    left_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_SITE))

    box_gid = _box_geom_id(model)
    palm_r_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_PALM_GEOM))
    palm_l_gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_PALM_GEOM))

    touch_r, detect_mode_r = _right_touch_geom_ids_box(model)
    touch_l = _left_touch_geom_ids(model)
    overlap_left_gids = _dual_overlap_monitor_geom_ids(model, side="left")
    overlap_right_gids = _dual_overlap_monitor_geom_ids(model, side="right")
    left_proxy_gids = overlap_left_gids
    right_proxy_gids = overlap_right_gids

    slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_SLIDE_JOINT_NAME)
    slide_dofadr = int(model.jnt_dofadr[slide_jid]) if slide_jid >= 0 else -1

    box_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, box_gid) or BOX_GEOM_NAME

    if verbose and (not silent):
        _pipe = "Dex3-1 articulated hands + real finger/box contacts" if use_dex3_hands else (
            "legacy proxy finger sliders (deprecated for dual-arm; see g1_proxy_deprecated)"
        )
        print(
            "\nNOTE: Dual-arm coordinated assist uses **bounded** ``xfrc_applied`` on the box.\n"
            f"Pipeline: {_pipe}.\n"
            f"sim_timestep={applied_dt:.4f}s  timeline_speed={motion.timeline_speed:.2f}  "
            f"fast_profile={bool(fast)}\n"
            f"right_touch_detection={detect_mode_r!r}  slide_joint="
            f"{BOX_SLIDE_JOINT_NAME if slide_jid >= 0 else 'MISSING'}\n"
            f"touches: right_geoms={_geom_names(model, touch_r)}  left_geoms={_geom_names(model, touch_l)}\n",
            flush=True,
        )

    det0 = _detect_box_dual(model, data, box_geom_name=box_geom_name)
    box0 = np.asarray(det0["box_center"])
    box_initial_z = float(box0[2])
    box_max_z = box_initial_z

    right_pre0 = np.asarray(det0["right_dual_pregrasp_target"], dtype=float)
    left_pre0 = np.asarray(det0["left_dual_pregrasp_target"], dtype=float)
    if verbose and (not silent):
        print(
            "dual_arm geometric targets / box frame:\n"
            f"  box_center={det0['box_center']!s}\n"
            f"  box_x_min={det0['box_x_min']:.5f}  box_x_max={det0['box_x_max']:.5f}\n"
            f"  box_y_min={det0['box_y_min']:.5f}  box_y_max={det0['box_y_max']:.5f}\n"
            f"  grasp_z={det0['grasp_height_z']:.5f}\n"
            f"  viz_right_contact_site={det0['viz_right_contact_world']!s}\n"
            f"  viz_left_contact_site={det0['viz_left_contact_world']!s}\n"
            f"  right_palm_target={right_pre0!s}\n"
            f"  left_palm_target={left_pre0!s}",
            flush=True,
        )
    if dbg_penetration:
        print(
            f"dual penetration tuning: DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M="
            f"{DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M:.5f}",
            flush=True,
        )

    left_target_mode = LEFT_TARGET_MODE_DUAL_OUTSIDE_EDGES

    p_stow_r = _point_world(
        model, data, site_id=right_site_id, fallback_body_id=right_bid
    )
    p_stow_l = _point_world(
        model, data, site_id=left_site_id, fallback_body_id=left_bid
    )

    finger_close_t0: float | None = None
    finger_open_t0: float | None = None
    last_proxy_phase_e = DualPhase.STOW

    squeeze_gate_cleared_t: float | None = None

    initial_touch_r = _contact_count_set(data, box_gid, touch_r)
    initial_touch_l = _contact_count_set(data, box_gid, touch_l)
    initial_contact_any = initial_touch_r + initial_touch_l > 0
    penetration_blocks_assist = False
    if initial_contact_any:
        penetration_blocks_assist = True

    min_touch_r_dist = float("inf")
    min_touch_l_dist = float("inf")
    assist_armed = False
    assist_anchor = np.zeros(3)
    grasp_assist_used = False
    max_pen_any_episode = 0.0
    max_pen_right_palm_episode = 0.0
    max_pen_left_palm_episode = 0.0
    max_pen_right_proxy_episode = 0.0
    max_pen_left_proxy_episode = 0.0
    max_right_c = max_left_c = 0
    lift_success = False
    lift_contact_during_lift = False
    max_assist_force = 0.0
    assist_active_flag = False
    last_print = -print_interval_s
    post_gate_freeze_accum = 0.0
    bilateral_stable_s = 0.0
    max_bilateral_stable_episode = 0.0
    right_contact_stability_s = 0.0
    left_contact_stability_s = 0.0
    max_right_contact_stability_episode = 0.0
    max_left_contact_stability_episode = 0.0
    geom_warned = False
    episode_diag: dict[str, Any] = {
        "right_face_gap": float("nan"),
        "left_face_gap": float("nan"),
        "right_contact_stability_s": 0.0,
        "left_contact_stability_s": 0.0,
        "right_normal_alignment": float("nan"),
        "left_normal_alignment": float("nan"),
        "inside_box_violation": False,
        "cross_center_violation": False,
        "palm_overlap_risk": False,
        "bilateral_stable_s": 0.0,
    }

    bilateral_ok_frame_count = 0
    strict_dual_pose_frame_count = 0

    phase_pen_peak: dict[str, dict[str, Any]] = {}

    prev_pen_rp_frame = 0.0
    prev_pen_lp_frame = 0.0
    prev_pen_rp_palm = 0.0
    prev_pen_lp_palm = 0.0

    max_geom_pair_near_r_episode = False
    max_geom_pair_near_l_episode = False

    palm_dist_final_dr = float("nan")
    palm_dist_final_dl = float("nan")
    mission_complete = False
    final_phase = DualPhase.STOW

    dex3_finger_safety_stop = False
    dex3_real_contact_stable_s = 0.0
    max_dex3_real_contact_stable_episode = 0.0
    dex3_contact_stable_decay_events = 0
    min_cross_hand_fingertip_episode_m = float("inf")
    min_right_fingertip_clearance_episode_m = float("inf")
    min_left_fingertip_clearance_episode_m = float("inf")
    max_dex3_right_fingertip_contacts_episode = 0
    max_dex3_left_fingertip_contacts_episode = 0
    prev_box_com_z = float(box_initial_z)
    prev_bilateral_stable_s = 0.0
    current_dex3_finger_mode_for_step = "open_hand"
    last_dex3_finger_apply: dict[str, float] = {}
    finger_closure_blocked_reason: str | None = None
    dex3_digit_policy_for_step: dict[str, str] = {}
    dex3_index_max_delta_rad_for_step: float | None = None
    dex3_digits_on_box_count = 0
    max_dex3_digits_on_box_count = 0

    wrist_orientation_locked = False
    wrist_orientation_ever_locked = False
    wrist_lock_snap_time: float | None = None
    wrist_frozen_ori_r: dict[str, Any] | None = None
    wrist_frozen_ori_l: dict[str, Any] | None = None
    wrist_lock_axis_r: np.ndarray | None = None
    wrist_lock_axis_l: np.ndarray | None = None
    palm_first_settle_s = 0.0
    wrist_orientation_lock_duration_s = 0.0
    wrist_lock_max_axis_err_r_episode = 0.0
    wrist_lock_max_axis_err_l_episode = 0.0
    wrist_vertical_only_r = False
    wrist_vertical_only_l = False
    wrist_roll_yaw_latch_r: dict[str, float] = {}
    wrist_roll_yaw_latch_l: dict[str, float] = {}
    palm_contact_stable_s_diag = 0.0
    finger_contact_stable_s_diag = 0.0
    max_palm_contact_stable_s_diag_episode = 0.0
    max_finger_contact_stable_s_diag_episode = 0.0
    min_right_index_tip_dist_episode = float("inf")
    min_left_index_tip_dist_episode = float("inf")
    min_right_thumb_tip_dist_episode = float("inf")
    min_left_thumb_tip_dist_episode = float("inf")

    # Warm-start toward neutral then first dual solve pulls both arms inward.
    for adr in ik_dof_adrs:
        data.qvel[adr] = 0.0

    def full_posture_from_dual(q_work: np.ndarray, *, phase: DualPhase) -> dict[str, float]:
        nonlocal last_dex3_finger_apply
        full = dict(neutral_eff)
        for jn, qv in zip(DUAL_IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        if finger_controller is not None:
            freeze = bool(
                (
                    finger_closure_blocked_reason is not None
                    and phase in (DualPhase.CLOSE_PROXY_HANDS, DualPhase.DUAL_GRASP_HOLD)
                )
                or (
                    finger_place_only
                    and max_dex3_digits_on_box_count >= 6
                    and phase in (DualPhase.DUAL_GRASP_HOLD, DualPhase.CLOSE_PROXY_HANDS)
                )
            ) and bool(last_dex3_finger_apply)
            if freeze:
                full.update(last_dex3_finger_apply)
            else:
                if dex3_digit_policy_for_step:
                    last_dex3_finger_apply = finger_controller.step_contact_aware(
                        current_dex3_finger_mode_for_step,
                        digit_policy=dex3_digit_policy_for_step,
                        index_max_delta_rad=dex3_index_max_delta_rad_for_step,
                    )
                else:
                    last_dex3_finger_apply = finger_controller.step(current_dex3_finger_mode_for_step)
                full.update(last_dex3_finger_apply)
        return full

    def step_frame(*, silent: bool = False) -> None:
        nonlocal squeeze_gate_cleared_t
        nonlocal finger_close_t0, finger_open_t0, last_proxy_phase_e
        nonlocal assist_armed, assist_anchor, grasp_assist_used, penetration_blocks_assist
        nonlocal max_pen_any_episode
        nonlocal max_pen_right_palm_episode, max_pen_left_palm_episode
        nonlocal max_pen_right_proxy_episode, max_pen_left_proxy_episode
        nonlocal max_right_c, max_left_c
        nonlocal lift_success, lift_contact_during_lift
        nonlocal box_max_z, assist_active_flag, max_assist_force
        nonlocal mission_complete, last_print
        nonlocal final_phase, min_touch_r_dist, min_touch_l_dist
        nonlocal post_gate_freeze_accum, palm_dist_final_dr, palm_dist_final_dl
        nonlocal bilateral_stable_s, max_bilateral_stable_episode, geom_warned
        nonlocal right_contact_stability_s, left_contact_stability_s
        nonlocal max_right_contact_stability_episode, max_left_contact_stability_episode
        nonlocal waist_frozen_xy, right_stable_for_freeze_frames, sequential_left_arm_frames
        nonlocal joint_jumps_by_phase
        nonlocal max_waist_delta_step_rad_episode
        nonlocal max_palm_convergence_speed_r, max_palm_convergence_speed_l
        nonlocal max_dual_goal_asymmetry_m, prev_dr_track, prev_dl_track
        nonlocal max_wrist_command_delta_episode, lift_phase_reached, lift_assist_applied_in_lift_phase
        nonlocal large_joint_jump_events, max_abs_joint_step_applied_episode
        nonlocal max_joint_vel_mag_episode, max_per_frame_dual_cmd_delta_episode
        nonlocal min_waist_yaw_episode, max_waist_yaw_episode, min_waist_pitch_episode, max_waist_pitch_episode
        nonlocal prev_palm_r_phys, prev_palm_l_phys
        nonlocal right_palm_vel_near_sum, right_palm_vel_near_n, left_palm_vel_near_sum, left_palm_vel_near_n

        nonlocal bilateral_ok_frame_count, strict_dual_pose_frame_count

        nonlocal phase_pen_peak

        nonlocal prev_pen_rp_frame, prev_pen_lp_frame
        nonlocal prev_pen_rp_palm, prev_pen_lp_palm

        nonlocal max_filtered_pr_goal_dist_episode, max_right_joint_command_delta_episode
        nonlocal max_right_bad_ik_streak_episode

        nonlocal max_geom_pair_near_r_episode, max_geom_pair_near_l_episode

        nonlocal dex3_finger_safety_stop
        nonlocal dex3_real_contact_stable_s, max_dex3_real_contact_stable_episode
        nonlocal dex3_contact_stable_decay_events
        nonlocal min_cross_hand_fingertip_episode_m
        nonlocal min_right_fingertip_clearance_episode_m, min_left_fingertip_clearance_episode_m
        nonlocal max_dex3_right_fingertip_contacts_episode, max_dex3_left_fingertip_contacts_episode
        nonlocal prev_box_com_z, prev_bilateral_stable_s
        nonlocal current_dex3_finger_mode_for_step, last_dex3_finger_apply, finger_closure_blocked_reason
        nonlocal dex3_digit_policy_for_step
        nonlocal dex3_index_max_delta_rad_for_step
        nonlocal dex3_digits_on_box_count, max_dex3_digits_on_box_count
        nonlocal wrist_orientation_locked, wrist_orientation_ever_locked, wrist_lock_snap_time
        nonlocal wrist_frozen_ori_r, wrist_frozen_ori_l, wrist_lock_axis_r, wrist_lock_axis_l
        nonlocal wrist_vertical_only_r, wrist_vertical_only_l
        nonlocal wrist_roll_yaw_latch_r, wrist_roll_yaw_latch_l
        nonlocal palm_first_settle_s, wrist_orientation_lock_duration_s
        nonlocal wrist_lock_max_axis_err_r_episode, wrist_lock_max_axis_err_l_episode
        nonlocal palm_contact_stable_s_diag, finger_contact_stable_s_diag
        nonlocal max_palm_contact_stable_s_diag_episode, max_finger_contact_stable_s_diag_episode
        nonlocal min_right_index_tip_dist_episode, min_left_index_tip_dist_episode
        nonlocal min_right_thumb_tip_dist_episode, min_left_thumb_tip_dist_episode

        dt_sim = float(model.opt.timestep)
        tout = float(timeout)
        sim_tm = float(data.time)

        phase = _dual_phase_effective(
            sim_tm,
            tout,
            squeeze_gate_cleared_t,
            post_gate_virtual_lag_s=post_gate_freeze_accum,
            finger_place_only=finger_place_only,
        )
        final_phase = phase

        if phase == DualPhase.DUAL_LIFT_TEST:
            lift_phase_reached = True

        sim_t_ik = (
            sim_tm - post_gate_freeze_accum
            if squeeze_gate_cleared_t is not None
            else sim_tm
        )

        post_spans = (
            _post_gate_abs_spans(
                squeeze_gate_cleared_t, tout, finger_place_only=finger_place_only
            )
            if squeeze_gate_cleared_t is not None
            else None
        )

        gp_proxy = _grasp_phase_for_proxy(phase)
        det_goal = _detect_box_dual(model, data, box_geom_name=box_geom_name)
        lt_touch = np.asarray(det_goal["left_dual_pregrasp_target"], dtype=float)
        tgt_r_pre = np.asarray(det_goal["right_dual_pregrasp_target"], dtype=float)
        palm_rp = _point_world(
            model, data, site_id=right_site_id, fallback_body_id=right_bid
        )
        palm_lp_pre = _point_world(
            model, data, site_id=left_site_id, fallback_body_id=left_bid
        )
        dr_pre = float(np.linalg.norm(palm_rp - tgt_r_pre))
        dl_pre = float(np.linalg.norm(palm_lp_pre - lt_touch))
        rc_pre = _contact_count_set(data, box_gid, touch_r)
        lc_pre = _contact_count_set(data, box_gid, touch_l)

        engage_left = (
            (phase not in PRE_RIGHT_ONLY_PHASES)
            or (dr_pre < _RIGHT_ENGAGE_LEFT_ARM_M)
            or (rc_pre > 0)
        )

        dex3_index_max_delta_rad_for_step = None
        wrist_ik_inner_scale = 1.0
        if phase not in FINGER_REACH_WRIST_PHASES:
            wrist_vertical_only_r = False
            wrist_vertical_only_l = False
            wrist_roll_yaw_latch_r = {}
            wrist_roll_yaw_latch_l = {}
        elif use_dex3_hands:
            if not wrist_vertical_only_r:
                wrist_roll_yaw_latch_r = _latch_wrist_roll_yaw(
                    data, joint_qpos_adr, RIGHT_WRIST_ROLL_YAW_JOINTS
                )
                wrist_vertical_only_r = True
            if not wrist_vertical_only_l:
                wrist_roll_yaw_latch_l = _latch_wrist_roll_yaw(
                    data, joint_qpos_adr, LEFT_WRIST_ROLL_YAW_JOINTS
                )
                wrist_vertical_only_l = True
            wrist_orientation_locked = False
            wrist_frozen_ori_r = None
            wrist_frozen_ori_l = None
            wrist_ik_inner_scale = 1.0
        if phase not in WRIST_ORIENTATION_LOCK_PHASES or (not use_dex3_hands):
            if not wrist_orientation_locked:
                palm_first_settle_s = 0.0
            if phase in (DualPhase.RETREAT, DualPhase.DONE, DualPhase.STOW):
                wrist_orientation_locked = False
                wrist_frozen_ori_r = None
                wrist_frozen_ori_l = None
                wrist_lock_axis_r = None
                wrist_lock_axis_l = None
        elif wrist_orientation_locked:
            wrist_ik_inner_scale = float(WRIST_LOCK_INNER_ITERS_SCALE)
            if WRIST_LOCK_DURATION_TRACK:
                wrist_orientation_lock_duration_s += dt_sim
        elif use_dex3_hands and not (wrist_vertical_only_r or wrist_vertical_only_l):
            dg = det_goal
            bxmi = float(dg["box_x_min"])
            bxma = float(dg["box_x_max"])
            bymi = float(dg["box_y_min"])
            byma = float(dg["box_y_max"])
            bzz = float(dg["bottom_z"])
            tzz = float(dg["top_z"])
            inside_rp_pre = _is_palm_inside_box_volume(
                palm_rp, bxmi, bxma, bymi, byma, bzz, tzz
            )
            inside_lp_pre = _is_palm_inside_box_volume(
                palm_lp_pre, bxmi, bxma, bymi, byma, bzz, tzz
            )
            palm_out_pre = not (inside_rp_pre or inside_lp_pre)
            near_x_e = float(dg["near_face_x"])
            ry_face_y = float(dg["left_edge_y"])
            ly_face_y = float(dg["right_edge_y"])
            cy_clear_e = float(DUAL_SURFACE_CONTACT_CLEARANCE_Y_M)
            palm_x_tol_e = near_x_e + 0.068
            right_strip_min_e = (
                ry_face_y
                - PALM_PLATE_HALF_WIDTH_Y
                - cy_clear_e
                - EDGE_CONTACT_Y_PAD_OUTBOARD_M
            )
            right_strip_max_e = ry_face_y + EDGE_CONTACT_Y_PAD_INBOARD_M
            left_strip_min_e = ly_face_y - EDGE_CONTACT_Y_PAD_INBOARD_M
            left_strip_max_e = (
                ly_face_y
                + PALM_PLATE_HALF_WIDTH_Y
                + cy_clear_e
                + EDGE_CONTACT_Y_PAD_OUTBOARD_M
            )
            strip_ok_pre = (
                right_strip_min_e <= float(palm_rp[1]) <= right_strip_max_e
                and left_strip_min_e <= float(palm_lp_pre[1]) <= left_strip_max_e
                and float(palm_rp[0]) <= palm_x_tol_e
                and float(palm_lp_pre[0]) <= palm_x_tol_e
            )
            r_align_pre = _palm_normal_alignment_score(
                data, right_site_id, RIGHT_PALM_AXIS_TARGET
            )
            l_align_pre = _palm_normal_alignment_score(
                data, left_site_id, LEFT_PALM_AXIS_TARGET
            )
            rpu_pre = float(_palm_up_alignment_score(data, right_site_id))
            lpu_pre = float(_palm_up_alignment_score(data, left_site_id))
            tgt_r_e = np.asarray(dg["right_dual_pregrasp_target"], dtype=float)
            tgt_l_e = np.asarray(dg["left_dual_pregrasp_target"], dtype=float)
            dr_pg = float(np.linalg.norm(palm_rp - tgt_r_e))
            dl_pg = float(np.linalg.norm(palm_lp_pre - tgt_l_e))
            palm_pose_tight = (
                dr_pg < float(RIGHT_PALM_CONTACT_GOAL_M) * 1.08
                and dl_pg < float(LEFT_PALM_CONTACT_GOAL_M) * 1.08
            )
            pen_pre_frame = float(max(prev_pen_rp_frame, prev_pen_lp_frame, 0.0))
            pen_ok_pre = pen_pre_frame <= float(pen_episode_cap_m) * 1.02
            palm_pre_ok = (
                palm_out_pre
                and palm_pose_tight
                and strip_ok_pre
                and r_align_pre >= float(DEX3_PALM_NORMAL_CLOSE_OK)
                and l_align_pre >= float(DEX3_PALM_NORMAL_CLOSE_OK)
                and rpu_pre >= float(DEX3_PALM_UP_OK)
                and lpu_pre >= float(DEX3_PALM_UP_OK)
                and pen_ok_pre
            )
            if palm_pre_ok:
                palm_first_settle_s += dt_sim
            else:
                palm_first_settle_s = max(0.0, palm_first_settle_s - 0.72 * dt_sim)
            if palm_first_settle_s + 1e-9 >= float(WRIST_LOCK_PALM_SETTLE_S):
                wrist_lock_axis_r = _locked_site_axis_snapshot(
                    data, right_site_id, RIGHT_PALM_AXIS_TARGET
                )
                wrist_lock_axis_l = _locked_site_axis_snapshot(
                    data, left_site_id, LEFT_PALM_AXIS_TARGET
                )
                up_r_snap = _locked_up_from_axis(wrist_lock_axis_r)
                up_l_snap = _locked_up_from_axis(wrist_lock_axis_l)
                wrist_frozen_ori_r = _wrist_lock_palm_ori_kw(wrist_lock_axis_r, up_r_snap)
                wrist_frozen_ori_l = _wrist_lock_palm_ori_kw(wrist_lock_axis_l, up_l_snap)
                wrist_orientation_locked = True
                wrist_orientation_ever_locked = True
                wrist_lock_snap_time = float(sim_tm)
                wrist_ik_inner_scale = float(WRIST_LOCK_INNER_ITERS_SCALE)

        if waist_frozen_xy is None and phase not in (DualPhase.STOW, DualPhase.DONE):
            arm_pose_ready_for_waist_freeze = (
                dr_pre < RIGHT_PREGRASP_STABLE_FOR_WAIST_FREEZE_M
                and dl_pre < LEFT_PREGRASP_LOOSE_FOR_WAIST_FREEZE_M
            )
            if arm_pose_ready_for_waist_freeze:
                right_stable_for_freeze_frames += 1
            else:
                right_stable_for_freeze_frames = 0
            if right_stable_for_freeze_frames >= WAIST_FREEZE_DEBOUNCE_STEPS:
                waist_frozen_xy = np.array(
                    [
                        float(data.qpos[int(waist_qpos_tuple[0])]),
                        float(data.qpos[int(waist_qpos_tuple[1])]),
                    ],
                    dtype=float,
                )

        coord_mode = _coordination_mode(phase, waist_frozen_xy=waist_frozen_xy)

        if phase != last_proxy_phase_e:
            if phase == DualPhase.CLOSE_PROXY_HANDS:
                finger_close_t0 = sim_tm
            if phase == DualPhase.OPEN_PROXY_HANDS:
                finger_open_t0 = sim_tm
            last_proxy_phase_e = phase

        if use_dex3_hands:
            slide_tgt = 0.0
        else:
            slide_tgt_raw = _proxy_finger_slide_target(
                gp_proxy,
                sim_tm,
                tout,
                close_t0=finger_close_t0,
                open_t0=finger_open_t0,
            )
            slide_tgt = float(min(slide_tgt_raw, DUAL_PROXY_MAX_SLIDE_M))
            spill_slide = float(
                max(
                    0.0,
                    prev_pen_rp_frame - MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M,
                    prev_pen_lp_frame - MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M,
                )
            )
            if spill_slide >= 0.003:
                slide_tgt = 0.0
            elif spill_slide > 1e-9:
                slide_tgt *= float(max(0.28, 1.0 - min(spill_slide * 30.0, 0.85)))
            if gp_proxy in (GraspPhase.CLOSE_HAND, GraspPhase.GRASP_HOLD, GraspPhase.LIFT_TEST):
                slide_tgt = min(slide_tgt, 0.0020)

        pr_goal = _dual_right_palm_ik_goal(
            gp_proxy,
            p_stow=p_stow_r,
            p_above=np.asarray(det_goal["right_dual_approach_target"], dtype=float),
            p_touch=np.asarray(det_goal["right_dual_pregrasp_target"], dtype=float),
            sim_t=sim_t_ik,
            duration=tout,
            post_spans=post_spans,
        )
        pl_traj = _left_palm_ik_goal(
            phase,
            p_stow=p_stow_l,
            p_above=np.asarray(det_goal["left_dual_approach_target"], dtype=float),
            p_touch=lt_touch,
            sim_t=sim_t_ik,
            duration=tout,
            post_spans=post_spans,
        )
        if (not use_dex3_hands) and gp_proxy == GraspPhase.LIFT_TEST:
            if post_spans is not None:
                ts0, ts1 = post_spans["lift"]
                u_lift = _smoothstep01((sim_t_ik - ts0) / max(ts1 - ts0, 1e-9))
            else:
                b_lb = _dual_phase_boundaries(tout)
                u_lift = _smoothstep01(
                    (sim_t_ik - b_lb["t_grasp_hold_end"])
                    / max(b_lb["t_lift_end"] - b_lb["t_grasp_hold_end"], 1e-9)
                )
            sq_y = DUAL_LIFT_INWARD_SQUEEZE_Y_M * u_lift
            pr_goal = pr_goal.copy()
            pl_traj = pl_traj.copy()
            pr_goal[1] += sq_y
            pl_traj[1] -= sq_y

        if phase in PALM_IK_PULLBACK_PHASES:
            spill_r = float(max(0.0, prev_pen_rp_palm - MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M))
            spill_l = float(max(0.0, prev_pen_lp_palm - MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M))
            if spill_r > 1e-9 or spill_l > 1e-9:
                bc = np.asarray(det_goal["box_center"], dtype=float).reshape(3)
                pr_goal = np.asarray(pr_goal, dtype=float).copy()
                pl_traj = np.asarray(pl_traj, dtype=float).copy()
                if spill_r > 1e-9:
                    vr = pr_goal - bc
                    vr[2] = 0.0
                    nr = float(np.linalg.norm(vr))
                    if nr > 1e-7:
                        pr_goal[:2] += (vr[:2] / nr) * min(spill_r * 2.5, 0.016)
                if spill_l > 1e-9:
                    vl = pl_traj - bc
                    vl[2] = 0.0
                    nl = float(np.linalg.norm(vl))
                    if nl > 1e-7:
                        pl_traj[:2] += (vl[:2] / nl) * min(spill_l * 2.5, 0.016)

        pr_goal_raw = np.asarray(pr_goal, dtype=float).copy()
        if phase in (DualPhase.DUAL_APPROACH, DualPhase.DUAL_DESCEND_TO_PREGRASP):
            a_pf = float(motion.pr_goal_filter_alpha)
            if pr_goal_filt_ref[0] is None:
                pr_goal_filt_ref[0] = pr_goal_raw.copy()
            else:
                pr_goal_filt_ref[0][:] = (1.0 - a_pf) * np.asarray(
                    pr_goal_filt_ref[0], dtype=float
                ) + a_pf * pr_goal_raw
            pr_goal_for_ik = np.asarray(pr_goal_filt_ref[0], dtype=float).copy()
        else:
            pr_goal_filt_ref[0] = None
            pr_goal_for_ik = pr_goal_raw.copy()
        filtered_pr_goal_distance = float(np.linalg.norm(pr_goal_raw - pr_goal_for_ik))
        max_filtered_pr_goal_dist_episode = max(
            max_filtered_pr_goal_dist_episode, filtered_pr_goal_distance
        )

        if engage_left:
            if coord_mode == CoordinationMode.RIGHT_PRIMARY_WAIST_FREE:
                left_target_eff = palm_lp_pre + float(SUPPORT_LEFT_TARGET_BLEND) * (
                    np.asarray(pl_traj, dtype=float) - palm_lp_pre
                )
            else:
                left_target_eff = np.asarray(pl_traj, dtype=float).copy()
        else:
            left_target_eff = p_stow_l.copy()

        dr_right_to_goal = float(
            np.linalg.norm(np.asarray(palm_rp, dtype=float) - np.asarray(pr_goal_for_ik, dtype=float))
        )
        if coord_mode == CoordinationMode.RIGHT_PRIMARY_WAIST_FREE:
            if dr_right_to_goal < SEQUENTIAL_IK_RIGHT_TO_GOAL_M or rc_pre > 0:
                sequential_left_arm_frames += 1
            else:
                sequential_left_arm_frames = 0
        else:
            sequential_left_arm_frames = int(motion.sequential_right_hold_frames)
        sequential_left_ik_ok = (
            sequential_left_arm_frames >= int(motion.sequential_right_hold_frames)
        )
        engage_left_ik = bool(
            engage_left
            and (
                sequential_left_ik_ok
                or rc_pre > 0
                or coord_mode != CoordinationMode.RIGHT_PRIMARY_WAIST_FREE
            )
        )

        waist_lock_xy: np.ndarray | None = (
            waist_frozen_xy if coord_mode == CoordinationMode.WAIST_LOCKED_DUAL_ARM else None
        )

        ori_rec_scale = 1.0

        prev_cmd = cmd_dual_prev.copy()
        i_wy = DUAL_IK_JOINT_NAMES.index("waist_yaw_joint")
        i_wp = DUAL_IK_JOINT_NAMES.index("waist_pitch_joint")
        prev_wy = float(prev_cmd[i_wy])
        prev_wp = float(prev_cmd[i_wp])
        waist_locked_this_frame = waist_lock_xy is not None

        ik_recovery_mode = False
        err_r_fin = err_l_fin = 0.0
        rec_r = rec_l = False

        if coord_mode == CoordinationMode.IDLE:
            alpha_idle = 0.12
            for i, jn in enumerate(DUAL_IK_JOINT_NAMES):
                adr = int(ik_qpos_adrs[i])
                tgt = float(neutral_dual_vec[i])
                blended = (1.0 - alpha_idle) * float(prev_cmd[i]) + alpha_idle * tgt
                data.qpos[adr] = blended
            mujoco.mj_forward(model, data)
            for jn in RIGHT_ARM_CMD_SMOOTH_JOINTS:
                right_arm_cmd_prev[jn] = float(data.qpos[int(joint_qpos_adr[jn])])
            for jn in LEFT_ARM_CMD_SMOOTH_JOINTS:
                left_arm_cmd_prev[jn] = float(data.qpos[int(joint_qpos_adr[jn])])
        else:
            err_r_fin, err_l_fin, rec_r, rec_l = _solve_primary_support_dual_arms(
                model,
                fd,
                data,
                coordination_mode=coord_mode,
                phase=phase,
                bilateral_stable_s=bilateral_stable_s,
                waist_qpos_adrs=waist_qpos_tuple,
                waist_lock_xy=waist_lock_xy,
                engage_left=engage_left_ik,
                r_full_adrs=r_qpos_adrs,
                r_full_low=r_low,
                r_full_high=r_high,
                q_neutral_r_full=q_neutral_r,
                r_arm_adrs=r_arm_adrs,
                r_arm_low=r_arm_low,
                r_arm_high=r_arm_high,
                q_neutral_r_arm=q_neutral_r_arm,
                l_arm_adrs=l_arm_adrs,
                l_arm_low=l_arm_low,
                l_arm_high=l_arm_high,
                q_neutral_l_arm=q_neutral_l_arm,
                q_neutral_l_arm_vals=q_neutral_l_arm_vals,
                target_right=pr_goal_for_ik,
                target_left=left_target_eff,
                right_site_id=right_site_id,
                right_body_id=right_bid,
                left_site_id=left_site_id,
                left_body_id=left_bid,
                posture_gain=posture_gain,
                max_abs_dn=max_joint_from_neutral,
                ori_recovery_scale=ori_rec_scale,
                palm_ori_box_center=np.asarray(det_goal["box_center"], dtype=float).reshape(3),
                palm_right_pre_ik_xyz=palm_rp,
                palm_left_pre_ik_xyz=palm_lp_pre,
                wrist_orientation_lock=bool(
                    wrist_orientation_locked
                    and use_dex3_hands
                    and not (wrist_vertical_only_r or wrist_vertical_only_l)
                ),
                wrist_frozen_kw_r=wrist_frozen_ori_r,
                wrist_frozen_kw_l=wrist_frozen_ori_l,
                ik_inner_iters_scale=float(wrist_ik_inner_scale * motion.ik_iter_scale),
                right_reject_streak=right_reject_streak,
                left_reject_streak=left_reject_streak,
                right_recovery_events=right_recovery_events,
                left_recovery_events=left_recovery_events,
                right_ik_rejects=right_ik_rejects,
                left_ik_rejects=left_ik_rejects,
            )

        if coord_mode != CoordinationMode.IDLE:
            max_right_bad_ik_streak_episode = max(
                max_right_bad_ik_streak_episode, int(right_reject_streak[0])
            )

        if waist_locked_this_frame and waist_lock_xy is not None:
            _enforce_waist_qpos(data, waist_qpos_tuple, waist_lock_xy)

        for i, jn in enumerate(DUAL_IK_JOINT_NAMES):
            adr = int(ik_qpos_adrs[i])
            if waist_locked_this_frame and jn in WAIST_IK_JOINT_NAMES:
                cmd_dual_prev[i] = float(data.qpos[adr])
                continue
            mx = (
                float(motion.max_waist_joint_step_rad)
                if jn in WAIST_IK_JOINT_NAMES
                else float(motion.max_arm_joint_step_rad)
            )
            raw = float(data.qpos[adr])
            dq = float(np.clip(raw - prev_cmd[i], -mx, mx))
            if jn not in RIGHT_ARM_CMD_SMOOTH_JOINTS and jn not in LEFT_ARM_CMD_SMOOTH_JOINTS:
                if abs(raw - prev_cmd[i]) > 3.0 * mx + 1e-9:
                    large_joint_jump_events += 1
                    joint_jumps_by_phase[phase.name] += 1
            applied = prev_cmd[i] + dq
            if coord_mode != CoordinationMode.IDLE and jn in WAIST_IK_JOINT_NAMES and not waist_locked_this_frame:
                dq2 = float(np.clip(applied - prev_cmd[i], -WAIST_RATE_LIMIT_RAD_PER_STEP, WAIST_RATE_LIMIT_RAD_PER_STEP))
                applied = prev_cmd[i] + dq2
            data.qpos[adr] = applied
            max_abs_joint_step_applied_episode = max(max_abs_joint_step_applied_episode, abs(float(applied - prev_cmd[i])))
            cmd_dual_prev[i] = applied

        if not waist_locked_this_frame:
            d_waist = float(
                np.hypot(
                    float(cmd_dual_prev[i_wy]) - prev_wy,
                    float(cmd_dual_prev[i_wp]) - prev_wp,
                )
            )
            max_waist_delta_step_rad_episode = max(max_waist_delta_step_rad_episode, d_waist)

        frame_wrist_max_delta = 0.0
        frame_right_joint_max_delta = 0.0
        right_wrist_rate_limited = False
        left_wrist_rate_limited = False
        wrist_cmd_tight = bool(
            use_dex3_hands
            and wrist_orientation_locked
            and phase in WRIST_ORIENTATION_LOCK_PHASES
            and not (wrist_vertical_only_r or wrist_vertical_only_l)
        )
        if coord_mode != CoordinationMode.IDLE:
            for jn in RIGHT_ARM_CMD_SMOOTH_JOINTS:
                adr = int(joint_qpos_adr[jn])
                wi = DUAL_IK_JOINT_NAMES.index(jn)
                raw_w = float(data.qpos[adr])
                prev_sm = float(right_arm_cmd_prev[jn])
                blended_u = float(motion.right_cmd_lp_alpha) * raw_w + (
                    1.0 - float(motion.right_cmd_lp_alpha)
                ) * prev_sm
                d_sm = blended_u - prev_sm
                lim_r = float(motion.right_cmd_max_delta_rad)
                if wrist_cmd_tight and jn in RIGHT_WRIST_JOINT_NAMES_DUAL:
                    lim_r = min(lim_r, float(WRIST_LOCK_MAX_WRIST_CMD_DELTA_RAD))
                if wrist_vertical_only_r and jn in RIGHT_WRIST_ROLL_YAW_JOINTS:
                    sm = float(wrist_roll_yaw_latch_r.get(jn, prev_sm))
                elif abs(d_sm) > lim_r:
                    sm = prev_sm + float(np.sign(d_sm)) * lim_r
                    if jn in RIGHT_WRIST_JOINT_NAMES_DUAL:
                        right_wrist_rate_limited = True
                else:
                    sm = blended_u
                data.qpos[adr] = sm
                right_arm_cmd_prev[jn] = sm
                cmd_dual_prev[wi] = sm
                d_cmd = abs(sm - prev_sm)
                frame_right_joint_max_delta = max(frame_right_joint_max_delta, d_cmd)
                max_right_joint_command_delta_episode = max(
                    max_right_joint_command_delta_episode, d_cmd
                )
                if jn in RIGHT_WRIST_JOINT_NAMES_DUAL:
                    frame_wrist_max_delta = max(frame_wrist_max_delta, abs(sm - prev_sm))
            for jn in LEFT_ARM_CMD_SMOOTH_JOINTS:
                adr = int(joint_qpos_adr[jn])
                wi = DUAL_IK_JOINT_NAMES.index(jn)
                raw_w = float(data.qpos[adr])
                prev_sm = float(left_arm_cmd_prev[jn])
                blended_u = float(motion.left_cmd_lp_alpha) * raw_w + (
                    1.0 - float(motion.left_cmd_lp_alpha)
                ) * prev_sm
                d_sm = blended_u - prev_sm
                lim_l = float(motion.left_cmd_max_delta_rad)
                if wrist_cmd_tight and jn in LEFT_WRIST_JOINT_NAMES_DUAL:
                    lim_l = min(lim_l, float(WRIST_LOCK_MAX_WRIST_CMD_DELTA_RAD))
                if wrist_vertical_only_l and jn in LEFT_WRIST_ROLL_YAW_JOINTS:
                    sm = float(wrist_roll_yaw_latch_l.get(jn, prev_sm))
                elif abs(d_sm) > lim_l:
                    sm = prev_sm + float(np.sign(d_sm)) * lim_l
                    if jn in LEFT_WRIST_JOINT_NAMES_DUAL:
                        left_wrist_rate_limited = True
                else:
                    sm = blended_u
                data.qpos[adr] = sm
                left_arm_cmd_prev[jn] = sm
                cmd_dual_prev[wi] = sm
                if jn in LEFT_WRIST_JOINT_NAMES_DUAL:
                    frame_wrist_max_delta = max(frame_wrist_max_delta, abs(sm - prev_sm))

        max_wrist_command_delta_episode = max(
            max_wrist_command_delta_episode, frame_wrist_max_delta
        )
        max_per_frame_dual_cmd_delta_episode = max(
            max_per_frame_dual_cmd_delta_episode,
            float(np.max(np.abs(cmd_dual_prev - prev_cmd))),
        )

        for adr in ik_dof_adrs:
            data.qvel[adr] = 0.0

        mujoco.mj_forward(model, data)

        q_work = np.array([float(data.qpos[adr]) for adr in ik_qpos_adrs], dtype=float)

        waist_frozen_episode = bool(waist_frozen_xy is not None)
        right_stage_active = bool(coord_mode != CoordinationMode.IDLE)
        left_stage_active = bool(coord_mode != CoordinationMode.IDLE)

        episode_diag["sequential_left_arm_frames"] = int(sequential_left_arm_frames)
        episode_diag["engage_left_ik"] = bool(engage_left_ik)
        episode_diag["right_stage_active"] = bool(right_stage_active)
        episode_diag["left_stage_active"] = bool(left_stage_active)
        episode_diag["waist_frozen"] = bool(waist_frozen_episode)
        episode_diag["right_ik_err"] = float(err_r_fin)
        episode_diag["left_ik_err"] = float(err_l_fin)
        episode_diag["max_wrist_command_delta"] = float(frame_wrist_max_delta)
        episode_diag["ik_recovery_mode"] = bool(ik_recovery_mode)
        episode_diag["right_wrist_rate_limited"] = bool(right_wrist_rate_limited)
        episode_diag["left_wrist_rate_limited"] = bool(left_wrist_rate_limited)
        episode_diag["right_arm_smoothing_active"] = bool(
            coord_mode != CoordinationMode.IDLE
        )
        episode_diag["right_bad_ik_frame_count"] = int(right_reject_streak[0])
        episode_diag["right_recovery_events"] = int(right_recovery_events[0])
        episode_diag["max_right_joint_command_delta"] = float(frame_right_joint_max_delta)
        episode_diag["filtered_pr_goal_distance"] = float(filtered_pr_goal_distance)

        if dbg_markers:
            det_vis = _detect_box_dual(model, data, box_geom_name=box_geom_name)
            G1BoxPerception.sync_debug_marker_sites(
                model, data, det_vis, box_body_name=BOX_BODY_NAME
            )
            mujoco.mj_forward(model, data)

        det_met = _detect_box_dual(model, data, box_geom_name=box_geom_name)

        palm_r = _point_world(
            model, data, site_id=right_site_id, fallback_body_id=right_bid
        )
        palm_l = _point_world(
            model, data, site_id=left_site_id, fallback_body_id=left_bid
        )
        tgt_r = np.asarray(det_met["right_dual_pregrasp_target"], dtype=float)
        tgt_left_ref = np.asarray(det_met["left_dual_pregrasp_target"], dtype=float)

        dr = float(np.linalg.norm(palm_r - tgt_r))
        dl_tgt = float(np.linalg.norm(palm_l - tgt_left_ref))
        min_touch_r_dist = min(min_touch_r_dist, dr)
        min_touch_l_dist = min(min_touch_l_dist, dl_tgt)
        palm_dist_final_dr = dr
        palm_dist_final_dl = dl_tgt

        dt_s = float(model.opt.timestep)
        if np.isfinite(prev_dr_track):
            max_palm_convergence_speed_r = max(
                max_palm_convergence_speed_r,
                abs(dr - prev_dr_track) / max(dt_s, 1e-9),
            )
        if np.isfinite(prev_dl_track):
            max_palm_convergence_speed_l = max(
                max_palm_convergence_speed_l,
                abs(dl_tgt - prev_dl_track) / max(dt_s, 1e-9),
            )
        max_dual_goal_asymmetry_m = max(max_dual_goal_asymmetry_m, abs(dr - dl_tgt))
        prev_dr_track = dr
        prev_dl_track = dl_tgt

        near_x = float(det_met["near_face_x"])
        ly = float(det_met["left_edge_y"])
        ry_edge = float(det_met["right_edge_y"])
        bz = float(det_met["bottom_z"])
        tz = float(det_met["top_z"])

        rc = _contact_count_set(data, box_gid, touch_r)
        lc = _contact_count_set(data, box_gid, touch_l)
        max_right_c = max(max_right_c, rc)
        max_left_c = max(max_left_c, lc)

        dex3_rsum: dict[str, Any] | None = None
        dex3_lsum: dict[str, Any] | None = None
        dex3_assist_contact_ok = False
        dex3_dual_relaxed_ready = False
        dex3_r_digit_dist = {"thumb": float("inf"), "index": float("inf"), "middle": float("inf")}
        dex3_l_digit_dist = {"thumb": float("inf"), "index": float("inf"), "middle": float("inf")}
        dex3_r_digit_contacts = {"thumb": 0, "index": 0, "middle": 0}
        dex3_l_digit_contacts = {"thumb": 0, "index": 0, "middle": 0}
        if use_dex3_hands:
            dex3_rsum = hand_box_contact_summary(
                model, data, box_gid=box_gid, side="right", palm_geom_id=palm_r_gid
            )
            dex3_lsum = hand_box_contact_summary(
                model, data, box_gid=box_gid, side="left", palm_geom_id=palm_l_gid
            )
            dex3_r_digit_dist = dex3_digit_box_distances(model, data, box_gid=box_gid, side="right")
            dex3_l_digit_dist = dex3_digit_box_distances(model, data, box_gid=box_gid, side="left")
            dex3_r_digit_contacts = digit_contact_counts_by_role(
                model, data, box_gid=box_gid, side="right"
            )
            dex3_l_digit_contacts = digit_contact_counts_by_role(
                model, data, box_gid=box_gid, side="left"
            )
            if int(dex3_rsum["palm_contacts"]) >= 1 and int(dex3_lsum["palm_contacts"]) >= 1:
                palm_contact_stable_s_diag += dt_sim
            else:
                palm_contact_stable_s_diag = max(0.0, palm_contact_stable_s_diag - 1.15 * dt_sim)
            if int(dex3_rsum["fingertip_contacts"]) >= 1 and int(dex3_lsum["fingertip_contacts"]) >= 1:
                finger_contact_stable_s_diag += dt_sim
            else:
                finger_contact_stable_s_diag = max(0.0, finger_contact_stable_s_diag - 0.95 * dt_sim)
            max_palm_contact_stable_s_diag_episode = max(
                max_palm_contact_stable_s_diag_episode, palm_contact_stable_s_diag
            )
            max_finger_contact_stable_s_diag_episode = max(
                max_finger_contact_stable_s_diag_episode, finger_contact_stable_s_diag
            )
            min_right_index_tip_dist_episode = min(
                min_right_index_tip_dist_episode, float(dex3_r_digit_dist["index"])
            )
            min_left_index_tip_dist_episode = min(
                min_left_index_tip_dist_episode, float(dex3_l_digit_dist["index"])
            )
            min_right_thumb_tip_dist_episode = min(
                min_right_thumb_tip_dist_episode, float(dex3_r_digit_dist["thumb"])
            )
            min_left_thumb_tip_dist_episode = min(
                min_left_thumb_tip_dist_episode, float(dex3_l_digit_dist["thumb"])
            )

        xmin = float(det_met["box_x_min"])
        xmax = float(det_met["box_x_max"])
        ymin = float(det_met["box_y_min"])
        ymax = float(det_met["box_y_max"])
        box_cy = float(det_met["box_center"][1])

        inside_rp = _is_palm_inside_box_volume(palm_r, xmin, xmax, ymin, ymax, bz, tz)
        inside_lp = _is_palm_inside_box_volume(palm_l, xmin, xmax, ymin, ymax, bz, tz)
        cross_rp = _is_proxy_crossing_box_center(palm_r, box_cy, side="right")
        cross_lp = _is_proxy_crossing_box_center(palm_l, box_cy, side="left")
        overlap_risk = _is_cross_arm_intersection_risk(
            model, data, left_gids=left_proxy_gids, right_gids=right_proxy_gids
        )
        geom_viol = bool(inside_rp or inside_lp or cross_rp or cross_lp)

        if (geom_viol or overlap_risk) and not geom_warned and verbose and not silent:
            geom_warned = True
            print(
                "[dual_arm] geometry guard tripped: "
                f"inside_box_violation={bool(inside_rp or inside_lp)} "
                f"cross_center_violation={bool(cross_rp or cross_lp)} "
                f"palm_overlap_risk={bool(overlap_risk)}",
                flush=True,
            )

        if geom_viol:
            penetration_blocks_assist = True
            assist_armed = False

        def _pen_gid(gid: int) -> float:
            if gid < 0:
                return 0.0
            return float(
                geom_max_penetration_into_world_axis_aligned_box(
                    model,
                    data,
                    gid,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    bz,
                    tz,
                )
            )

        pen_rp = _pen_gid(palm_r_gid)
        pen_lp = _pen_gid(palm_l_gid)

        prx_ids: list[int]
        plx_ids: list[int]
        if use_dex3_hands:
            prx_ids = dex3_pen_right_gid_list
            plx_ids = dex3_pen_left_gid_list
        else:
            prx_ids = []
            for nm in RIGHT_PROXY_GEOMS:
                g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
                if g >= 0:
                    prx_ids.append(int(g))
            plx_ids = []
            for nm in LEFT_PROXY_GEOMS:
                g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
                if g >= 0:
                    plx_ids.append(int(g))

        pen_rpx = max((_pen_gid(g) for g in prx_ids), default=0.0)
        pen_lpx = max((_pen_gid(g) for g in plx_ids), default=0.0)

        prior_pen_rp_frame = float(prev_pen_rp_frame)
        prior_pen_lp_frame = float(prev_pen_lp_frame)
        prev_pen_rp_frame = float(max(pen_rp, pen_rpx))
        prev_pen_lp_frame = float(max(pen_lp, pen_lpx))
        prev_pen_rp_palm = float(pen_rp)
        prev_pen_lp_palm = float(pen_lp)

        max_pen_any_frame = float(max(pen_rp, pen_lp, pen_rpx, pen_lpx))
        max_pen_any_episode = max(max_pen_any_episode, max_pen_any_frame)
        max_pen_right_palm_episode = max(max_pen_right_palm_episode, pen_rp)
        max_pen_left_palm_episode = max(max_pen_left_palm_episode, pen_lp)
        max_pen_right_proxy_episode = max(max_pen_right_proxy_episode, pen_rpx)
        max_pen_left_proxy_episode = max(max_pen_left_proxy_episode, pen_lpx)

        axis_r_diag = _site_axis_world(data, right_site_id, axis_col=0)
        align_r_diag = float(_normal_alignment_score(axis_r_diag, RIGHT_PALM_AXIS_TARGET))

        pen_candidates: list[tuple[str, float, int]] = [
            (RIGHT_PALM_GEOM, pen_rp, palm_r_gid),
            (LEFT_PALM_GEOM, pen_lp, palm_l_gid),
        ]
        for gid in prx_ids:
            gnm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            pen_candidates.append((gnm, _pen_gid(gid), gid))
        for gid in plx_ids:
            gnm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            pen_candidates.append((gnm, _pen_gid(gid), gid))
        leader_name, leader_pen, leader_gid = max(pen_candidates, key=lambda t: t[1])

        if phase in PENETRATION_DIAG_PHASES:
            pk = phase.name
            prev_best = phase_pen_peak.get(pk)
            if prev_best is None or max_pen_any_frame > float(prev_best["max_penetration_m"]) + 1e-15:
                Rg = np.asarray(data.geom_xmat[leader_gid]).reshape(3, 3)
                local_x_w = Rg @ np.array([1.0, 0.0, 0.0], dtype=float)
                nx = float(np.linalg.norm(local_x_w))
                if nx > 1e-12:
                    local_x_w = local_x_w / nx
                geom_c = np.asarray(data.geom_xpos[leader_gid, :3], dtype=float)
                leader_is_left = bool(
                    leader_name == LEFT_PALM_GEOM
                    or str(leader_name).startswith("left_")
                    or int(leader_gid) == int(palm_l_gid)
                )
                site_xyz = (
                    np.asarray(palm_l, dtype=float)
                    if leader_is_left
                    else np.asarray(palm_r, dtype=float)
                )
                r_plate_max_x = float(palm_plate_max_world_x_extent(model, data, palm_r_gid))
                phase_pen_peak[pk] = {
                    "max_penetration_m": float(max_pen_any_frame),
                    "leader_geom": leader_name,
                    "leader_penetration_m": float(leader_pen),
                    "sim_time": float(sim_tm),
                    "palm_site_xyz": site_xyz.tolist(),
                    "leader_geom_center_xyz": geom_c.tolist(),
                    "near_face_x": float(det_met["near_face_x"]),
                    "geom_local_x_world_unit": local_x_w.tolist(),
                    "right_palm_ik_target_xyz": np.asarray(tgt_r, dtype=float).tolist(),
                    "right_site_to_target_m": float(np.linalg.norm(palm_r - tgt_r)),
                    "left_site_to_target_m": float(np.linalg.norm(palm_l - tgt_left_ref)),
                    "right_normal_alignment": align_r_diag,
                    "right_plate_max_world_x": r_plate_max_x,
                    "box_x_min": float(xmin),
                    "site_minus_geom_center": (site_xyz - geom_c).tolist(),
                }

        if max_pen_any_frame > pen_episode_cap_m and phase in PRE_GATE:
            penetration_blocks_assist = True

        sep_r = _min_geom_distance_to_box(
            model, data, box_gid=box_gid, touch_gids=touch_r
        )
        sep_l = _min_geom_distance_to_box(
            model, data, box_gid=box_gid, touch_gids=touch_l
        )
        _geom_touch_eps = (
            max(float(GEOM_PAIR_TOUCH_DIST_OK_M), 0.0068) if use_dex3_hands else float(GEOM_PAIR_TOUCH_DIST_OK_M)
        )
        geom_touch_r = sep_r <= _geom_touch_eps
        geom_touch_l = sep_l <= _geom_touch_eps

        # Outside-y strip: ymin face (robot-right palm) vs ymax face (robot-left palm).
        ry_face_y = ly
        ly_face_y = ry_edge
        palm_x_tol = near_x + 0.068
        cy_clear = float(DUAL_SURFACE_CONTACT_CLEARANCE_Y_M)
        right_strip_min = (
            ry_face_y
            - PALM_PLATE_HALF_WIDTH_Y
            - cy_clear
            - EDGE_CONTACT_Y_PAD_OUTBOARD_M
        )
        right_strip_max = ry_face_y + EDGE_CONTACT_Y_PAD_INBOARD_M
        left_strip_min = ly_face_y - EDGE_CONTACT_Y_PAD_INBOARD_M
        left_strip_max = (
            ly_face_y
            + PALM_PLATE_HALF_WIDTH_Y
            + cy_clear
            + EDGE_CONTACT_Y_PAD_OUTBOARD_M
        )

        axis_r_w = _site_axis_world(data, right_site_id, axis_col=0)
        right_normal_alignment = _palm_normal_alignment_score(
            data, right_site_id, RIGHT_PALM_AXIS_TARGET
        )
        left_normal_alignment = _palm_normal_alignment_score(
            data, left_site_id, LEFT_PALM_AXIS_TARGET
        )

        rw_orient_err = 0.0
        lw_orient_err = 0.0
        if wrist_orientation_locked and wrist_lock_axis_r is not None:
            rw_orient_err = float(
                _palm_axis_tracking_error(data, right_site_id, wrist_lock_axis_r)
            )
        if wrist_orientation_locked and wrist_lock_axis_l is not None:
            lw_orient_err = float(
                _palm_axis_tracking_error(data, left_site_id, wrist_lock_axis_l)
            )
        if wrist_orientation_locked:
            wrist_lock_max_axis_err_r_episode = max(
                wrist_lock_max_axis_err_r_episode, rw_orient_err
            )
            wrist_lock_max_axis_err_l_episode = max(
                wrist_lock_max_axis_err_l_episode, lw_orient_err
            )

        right_touch_quality_gate = rc >= 1 or geom_touch_r
        left_touch_quality_gate = lc >= 1 or geom_touch_l

        valid_right_contact = bool(
            right_touch_quality_gate
            and right_strip_min <= float(palm_r[1]) <= right_strip_max
            and float(palm_r[0]) <= palm_x_tol
        )
        valid_left_contact = bool(
            left_touch_quality_gate
            and left_strip_min <= float(palm_l[1]) <= left_strip_max
            and float(palm_l[0]) <= palm_x_tol
        )
        right_side_geometry_ok = bool(
            right_strip_min <= float(palm_r[1]) <= right_strip_max
            and float(palm_r[0]) <= palm_x_tol
        )
        left_side_geometry_ok = bool(
            left_strip_min <= float(palm_l[1]) <= left_strip_max
            and float(palm_l[0]) <= palm_x_tol
        )

        min_rd = min(float(dex3_r_digit_dist[k]) for k in dex3_r_digit_dist)
        min_ld = min(float(dex3_l_digit_dist[k]) for k in dex3_l_digit_dist)
        if use_dex3_hands and dex3_rsum is not None and dex3_lsum is not None:
            dex3_dual_relaxed_ready = bool(
                dex3_per_hand_lift_ready_relaxed(
                    side_face_ok=valid_right_contact,
                    palm_contacts=int(dex3_rsum["palm_contacts"]),
                    fingertip_contacts=int(dex3_rsum["fingertip_contacts"]),
                    min_digit_to_box_m=min_rd,
                    near_threshold_m=float(DEX3_FINGERTIP_NEAR_BOX_M),
                )
                and dex3_per_hand_lift_ready_relaxed(
                    side_face_ok=valid_left_contact,
                    palm_contacts=int(dex3_lsum["palm_contacts"]),
                    fingertip_contacts=int(dex3_lsum["fingertip_contacts"]),
                    min_digit_to_box_m=min_ld,
                    near_threshold_m=float(DEX3_FINGERTIP_NEAR_BOX_M),
                )
            )
            dex3_assist_contact_ok = bool(dex3_dual_relaxed_ready)

        right_palm_up_alignment = float(_palm_up_alignment_score(data, right_site_id))
        left_palm_up_alignment = float(_palm_up_alignment_score(data, left_site_id))
        wp_r = float(data.qpos[int(hinge_addrs["right_wrist_pitch_joint"]["qpos_adr"])])
        wp_l = float(data.qpos[int(hinge_addrs["left_wrist_pitch_joint"]["qpos_adr"])])
        wrist_pitch_down_warning = bool(
            wp_r < -0.48
            or wp_l < -0.48
            or right_palm_up_alignment < DEX3_PALM_UP_OK
            or left_palm_up_alignment < DEX3_PALM_UP_OK
        )

        strict_dual_pose = dr < RIGHT_PALM_CONTACT_GOAL_M and dl_tgt < LEFT_PALM_CONTACT_GOAL_M

        palm_outside_vol = bool((not inside_rp) and (not inside_lp))
        pen_ok = bool(max_pen_any_frame <= pen_episode_cap_m)
        pen_slack = 0.0025 if use_dex3_hands else 0.0075
        pen_ok_acc = bool(max_pen_any_frame <= pen_episode_cap_m + pen_slack)

        surface_ok_r = bool(
            (rc >= 1 or geom_touch_r)
            and valid_right_contact
            and palm_outside_vol
            and pen_ok
        )
        surface_ok_l = bool(
            (lc >= 1 or geom_touch_l)
            and valid_left_contact
            and palm_outside_vol
            and pen_ok
        )

        bilateral_ok_frame = bool(
            surface_ok_r
            and surface_ok_l
            and (not cross_rp)
            and (not cross_lp)
            and palm_outside_vol
            and pen_ok
        )

        assist_bilateral_for_prep = bool(bilateral_ok_frame)
        if use_dex3_hands:
            assist_bilateral_for_prep = bool(
                bilateral_ok_frame
                or (
                    dex3_assist_contact_ok
                    and palm_outside_vol
                    and pen_ok
                )
            )

        if use_dex3_hands:
            _clm = dex3_finger_clearance_metrics(model, data)
            _mcr = float(_clm["cross_hand_fingertip_min_m"])
            min_cross_hand_fingertip_episode_m = min(min_cross_hand_fingertip_episode_m, _mcr)
            min_right_fingertip_clearance_episode_m = min(
                min_right_fingertip_clearance_episode_m,
                float(_clm["right_hand_fingertip_min_m"]),
            )
            min_left_fingertip_clearance_episode_m = min(
                min_left_fingertip_clearance_episode_m,
                float(_clm["left_hand_fingertip_min_m"]),
            )
            if _mcr < float(DEX3_CROSS_HAND_FINGER_STOP_M):
                dex3_finger_safety_stop = True
            elif _mcr > float(DEX3_CROSS_HAND_FINGER_STOP_M) * 1.28:
                dex3_finger_safety_stop = False
            if dex3_rsum is not None:
                max_dex3_right_fingertip_contacts_episode = max(
                    max_dex3_right_fingertip_contacts_episode,
                    int(dex3_rsum["fingertip_contacts"]),
                )
            if dex3_lsum is not None:
                max_dex3_left_fingertip_contacts_episode = max(
                    max_dex3_left_fingertip_contacts_episode,
                    int(dex3_lsum["fingertip_contacts"]),
                )

        geom_dual_soft_ok = bool(
            geom_touch_r
            and geom_touch_l
            and valid_right_contact
            and valid_left_contact
            and palm_outside_vol
            and pen_ok_acc
        )

        if geom_touch_r and valid_right_contact and palm_outside_vol and pen_ok_acc:
            max_geom_pair_near_r_episode = True
        if geom_touch_l and valid_left_contact and palm_outside_vol and pen_ok_acc:
            max_geom_pair_near_l_episode = True

        if strict_dual_pose:
            strict_dual_pose_frame_count += 1
        if bilateral_ok_frame:
            bilateral_ok_frame_count += 1

        if bilateral_ok_frame:
            bilateral_stable_s += dt_sim
        elif strict_dual_pose and geom_dual_soft_ok:
            bilateral_stable_s += 0.78 * dt_sim
        elif (
            strict_dual_pose
            and palm_outside_vol
            and pen_ok_acc
            and valid_right_contact
            and valid_left_contact
        ):
            bilateral_stable_s += 0.34 * dt_sim
        elif (
            strict_dual_pose
            and palm_outside_vol
            and pen_ok_acc
            and (not cross_rp)
            and (not cross_lp)
            and (not geom_viol)
        ):
            bilateral_stable_s += 0.29 * dt_sim
        else:
            decay = 2.8 * dt_sim
            if strict_dual_pose and palm_outside_vol and pen_ok_acc:
                decay = 0.48 * dt_sim
            bilateral_stable_s = max(0.0, bilateral_stable_s - decay)
        max_bilateral_stable_episode = max(max_bilateral_stable_episode, bilateral_stable_s)

        box_c_pre = np.asarray(det_met["box_center"], dtype=float).reshape(3)
        closure_block_this: str | None = None
        if (
            use_dex3_hands
            and dex3_rsum is not None
            and dex3_lsum is not None
            and phase in (DualPhase.CLOSE_PROXY_HANDS, DualPhase.DUAL_GRASP_HOLD)
        ):
            if (
                phase == DualPhase.CLOSE_PROXY_HANDS
                and not wrist_orientation_locked
                and not (wrist_vertical_only_r and wrist_vertical_only_l)
            ):
                closure_block_this = "await_wrist_orientation_lock"
            elif (
                right_normal_alignment + 1e-9 < float(DEX3_PALM_NORMAL_CLOSE_OK)
                or left_normal_alignment + 1e-9 < float(DEX3_PALM_NORMAL_CLOSE_OK)
            ):
                closure_block_this = "palm_not_aligned"
            elif (
                right_palm_up_alignment + 1e-9 < float(DEX3_PALM_UP_OK)
                or left_palm_up_alignment + 1e-9 < float(DEX3_PALM_UP_OK)
            ):
                closure_block_this = "palm_pitch_down"
            elif max_pen_any_frame > float(DEX3_CLEAN_PENETRATION_CAP_M):
                closure_block_this = "penetration_high"
            elif float(box_c_pre[2]) + 1e-4 < float(prev_box_com_z):
                closure_block_this = "box_being_pushed"
            elif float(bilateral_stable_s) + 1e-6 < float(prev_bilateral_stable_s) - 0.038:
                closure_block_this = "contact_stability_decreasing"
            elif dex3_finger_safety_stop:
                closure_block_this = "cross_hand_clearance_low"

        palms_aligned_for_close = bool(
            dr < RIGHT_PALM_CONTACT_GOAL_M * 0.86 and dl_tgt < LEFT_PALM_CONTACT_GOAL_M * 0.86
        )

        dex3_align_ok = bool(
            right_normal_alignment + 1e-9 >= float(DEX3_PALM_NORMAL_OK)
            and left_normal_alignment + 1e-9 >= float(DEX3_PALM_NORMAL_OK)
            and right_palm_up_alignment + 1e-9 >= float(DEX3_PALM_UP_OK)
            and left_palm_up_alignment + 1e-9 >= float(DEX3_PALM_UP_OK)
        )
        dex3_sticky_gate = bool(
            use_dex3_hands
            and dex3_dual_relaxed_ready
            and palm_outside_vol
            and pen_ok
            and dex3_align_ok
        )
        if dex3_sticky_gate:
            dex3_real_contact_stable_s += dt_sim
        elif use_dex3_hands:
            before_s = float(dex3_real_contact_stable_s)
            dex3_real_contact_stable_s = max(
                0.0, dex3_real_contact_stable_s - dt_sim * float(DEX3_STICKY_CONTACT_LEAK)
            )
            if before_s > 1e-6 and dex3_real_contact_stable_s < before_s - 1e-9:
                dex3_contact_stable_decay_events += 1
        max_dex3_real_contact_stable_episode = max(
            max_dex3_real_contact_stable_episode, dex3_real_contact_stable_s
        )

        if use_dex3_hands:
            finger_closure_blocked_reason = closure_block_this
            current_dex3_finger_mode_for_step = _dex3_effective_finger_mode(
                phase,
                safety_stop=dex3_finger_safety_stop,
                closure_block=closure_block_this,
                palms_aligned_for_close=palms_aligned_for_close,
                finger_place_only=finger_place_only,
            )
            dex3_digit_policy_for_step = {}
            palm_has_r = bool(dex3_rsum is not None and int(dex3_rsum["palm_contacts"]) >= 1)
            palm_has_l = bool(dex3_lsum is not None and int(dex3_lsum["palm_contacts"]) >= 1)
            if phase in (
                DualPhase.DUAL_CONTACT_HOLD,
                DualPhase.CLOSE_PROXY_HANDS,
                DualPhase.DUAL_GRASP_HOLD,
                DualPhase.DUAL_LIFT_TEST,
                DualPhase.DUAL_LOWER_BACK,
            ):
                dex3_digit_policy_for_step.update(
                    _dex3_digit_policy(
                        phase=phase,
                        side="right",
                        side_geometry_ok=right_side_geometry_ok,
                        palm_aligned=right_normal_alignment >= DEX3_PALM_NORMAL_CLOSE_OK,
                        palm_up_ok=right_palm_up_alignment >= DEX3_PALM_UP_OK,
                        palm_outside=palm_outside_vol,
                        palm_has_contact=palm_has_r,
                        wrist_orientation_locked=wrist_orientation_locked,
                        penetration_m=float(prev_pen_rp_frame),
                        penetration_prev_m=float(prior_pen_rp_frame),
                        digit_distances=dex3_r_digit_dist,
                        digit_contacts=dex3_r_digit_contacts,
                        safety_stop=dex3_finger_safety_stop,
                        finger_place_only=finger_place_only,
                    )
                )
                dex3_digit_policy_for_step.update(
                    _dex3_digit_policy(
                        phase=phase,
                        side="left",
                        side_geometry_ok=left_side_geometry_ok,
                        palm_aligned=left_normal_alignment >= DEX3_PALM_NORMAL_CLOSE_OK,
                        palm_up_ok=left_palm_up_alignment >= DEX3_PALM_UP_OK,
                        palm_outside=palm_outside_vol,
                        palm_has_contact=palm_has_l,
                        wrist_orientation_locked=wrist_orientation_locked,
                        penetration_m=float(prev_pen_lp_frame),
                        penetration_prev_m=float(prior_pen_lp_frame),
                        digit_distances=dex3_l_digit_dist,
                        digit_contacts=dex3_l_digit_contacts,
                        safety_stop=dex3_finger_safety_stop,
                        finger_place_only=finger_place_only,
                    )
                )
            if finger_place_only and dex3_digit_policy_for_step and finger_controller is not None:
                dex3_index_max_delta_rad_for_step = float(motion.digit_approach_max_delta_rad)
                dex3_digits_on_box_count = 0
                for side, dcont, ddist in (
                    ("right", dex3_r_digit_contacts, dex3_r_digit_dist),
                    ("left", dex3_l_digit_contacts, dex3_l_digit_dist),
                ):
                    for role in ("thumb", "index", "middle"):
                        key = f"{side}_{role}"
                        touching = int(dcont.get(role, 0)) > 0 or float(
                            ddist.get(role, float("inf"))
                        ) <= float(DEX3_DIGIT_CONTACT_FREEZE_DIST_M)
                        if touching:
                            dex3_digits_on_box_count += 1
                        if dex3_digit_policy_for_step.get(key) == "hold" and touching:
                            finger_controller.freeze_digit_joints(
                                dex3_digit_joint_names(side, role)
                            )
                max_dex3_digits_on_box_count = max(
                    max_dex3_digits_on_box_count, dex3_digits_on_box_count
                )
            elif (
                wrist_orientation_locked
                and dex3_rsum is not None
                and dex3_lsum is not None
                and palm_has_r
                and palm_has_l
                and phase
                in (
                    DualPhase.CLOSE_PROXY_HANDS,
                    DualPhase.DUAL_GRASP_HOLD,
                    DualPhase.DUAL_LIFT_TEST,
                    DualPhase.DUAL_LOWER_BACK,
                )
            ):
                dex3_index_max_delta_rad_for_step = 0.036
        else:
            finger_closure_blocked_reason = None
            current_dex3_finger_mode_for_step = "open_hand"
            dex3_digit_policy_for_step = {}

        prev_bilateral_stable_s = float(bilateral_stable_s)
        prev_box_com_z = float(box_c_pre[2])

        episode_diag["dex3_timeline_phase_display"] = _dex3_timeline_phase_display(
            phase, use_dex3=use_dex3_hands
        )
        episode_diag["dex3_real_contact_stable_s"] = float(dex3_real_contact_stable_s)
        episode_diag["dex3_contact_stable_decay_events"] = int(dex3_contact_stable_decay_events)
        episode_diag["finger_closure_blocked_reason"] = finger_closure_blocked_reason
        episode_diag["dex3_digit_policy"] = dict(dex3_digit_policy_for_step)
        episode_diag["right_palm_normal_alignment"] = float(right_normal_alignment)
        episode_diag["left_palm_normal_alignment"] = float(left_normal_alignment)
        episode_diag["right_normal_alignment"] = float(right_normal_alignment)
        episode_diag["left_normal_alignment"] = float(left_normal_alignment)
        episode_diag["right_palm_up_alignment"] = float(right_palm_up_alignment)
        episode_diag["left_palm_up_alignment"] = float(left_palm_up_alignment)
        episode_diag["wrist_pitch_down_warning"] = bool(wrist_pitch_down_warning)
        if use_dex3_hands and dex3_rsum is not None and dex3_lsum is not None:
            episode_diag["right_real_finger_contact_count"] = int(dex3_rsum["fingertip_contacts"])
            episode_diag["left_real_finger_contact_count"] = int(dex3_lsum["fingertip_contacts"])
            episode_diag["right_thumb_box_distance"] = float(dex3_r_digit_dist["thumb"])
            episode_diag["right_index_box_distance"] = float(dex3_r_digit_dist["index"])
            episode_diag["right_middle_box_distance"] = float(dex3_r_digit_dist["middle"])
            episode_diag["left_thumb_box_distance"] = float(dex3_l_digit_dist["thumb"])
            episode_diag["left_index_box_distance"] = float(dex3_l_digit_dist["index"])
            episode_diag["left_middle_box_distance"] = float(dex3_l_digit_dist["middle"])
            episode_diag["wrist_orientation_locked"] = bool(wrist_orientation_locked)
            episode_diag["right_wrist_orientation_error"] = float(rw_orient_err)
            episode_diag["left_wrist_orientation_error"] = float(lw_orient_err)
            episode_diag["wrist_orientation_lock_time"] = (
                float(wrist_lock_snap_time) if wrist_lock_snap_time is not None else float("nan")
            )
            episode_diag["palm_contact_stable_time"] = float(palm_contact_stable_s_diag)
            episode_diag["finger_contact_stable_time"] = float(finger_contact_stable_s_diag)
            episode_diag["right_index_tip_distance"] = float(dex3_r_digit_dist["index"])
            episode_diag["left_index_tip_distance"] = float(dex3_l_digit_dist["index"])
            episode_diag["right_thumb_tip_distance"] = float(dex3_r_digit_dist["thumb"])
            episode_diag["left_thumb_tip_distance"] = float(dex3_l_digit_dist["thumb"])
            episode_diag["palm_first_settle_s"] = float(palm_first_settle_s)
        else:
            episode_diag["wrist_orientation_locked"] = bool(
                wrist_orientation_locked and use_dex3_hands
            )
            episode_diag["right_wrist_orientation_error"] = float(rw_orient_err)
            episode_diag["left_wrist_orientation_error"] = float(lw_orient_err)
            episode_diag["wrist_orientation_lock_time"] = (
                float(wrist_lock_snap_time) if wrist_lock_snap_time is not None else float("nan")
            )
            episode_diag["palm_contact_stable_time"] = float(palm_contact_stable_s_diag)
            episode_diag["finger_contact_stable_time"] = float(finger_contact_stable_s_diag)
            episode_diag["right_index_tip_distance"] = float("nan")
            episode_diag["left_index_tip_distance"] = float("nan")
            episode_diag["right_thumb_tip_distance"] = float("nan")
            episode_diag["left_thumb_tip_distance"] = float("nan")
            episode_diag["palm_first_settle_s"] = float(palm_first_settle_s)

        if pen_ok_acc and (
            (rc >= 1 or geom_touch_r)
            or (dr < RIGHT_PALM_CONTACT_GOAL_M * 1.08)
        ):
            right_contact_stability_s += dt_sim
        else:
            right_contact_stability_s = max(0.0, right_contact_stability_s - 3.2 * dt_sim)
        if pen_ok_acc and (
            (lc >= 1 or geom_touch_l)
            or (dl_tgt < LEFT_PALM_CONTACT_GOAL_M * 1.08)
        ):
            left_contact_stability_s += dt_sim
        else:
            left_contact_stability_s = max(0.0, left_contact_stability_s - 3.2 * dt_sim)
        max_right_contact_stability_episode = max(
            max_right_contact_stability_episode, right_contact_stability_s
        )
        max_left_contact_stability_episode = max(
            max_left_contact_stability_episode, left_contact_stability_s
        )

        squeeze_ready_geometric = (
            (not initial_contact_any)
            and strict_dual_pose
            and (not geom_viol)
            and pen_ok
            and palm_outside_vol
            and bilateral_stable_s >= DUAL_CONTACT_SETTLE_TIME_S
        )

        if squeeze_gate_cleared_t is None and phase == DualPhase.DUAL_CONTACT_HOLD:
            if squeeze_ready_geometric:
                squeeze_gate_cleared_t = sim_tm
                if FREEZE_WAIST_AFTER_CONTACT and waist_frozen_xy is None:
                    waist_frozen_xy = np.array(
                        [
                            float(data.qpos[int(waist_qpos_tuple[0])]),
                            float(data.qpos[int(waist_qpos_tuple[1])]),
                        ],
                        dtype=float,
                    )

        if squeeze_gate_cleared_t is not None and phase == DualPhase.CLOSE_PROXY_HANDS:
            if dr >= RIGHT_PALM_CONTACT_GOAL_M or dl_tgt >= LEFT_PALM_CONTACT_GOAL_M:
                post_gate_freeze_accum += dt_sim

        if squeeze_gate_cleared_t is not None and phase == DualPhase.DUAL_GRASP_HOLD:
            if bilateral_stable_s < DUAL_CONTACT_SETTLE_TIME_S:
                post_gate_freeze_accum += dt_sim

        if phase == DualPhase.DUAL_LIFT_TEST and not bilateral_ok_frame:
            post_gate_freeze_accum += dt_sim

        settle_gate = bilateral_stable_s >= DUAL_CONTACT_SETTLE_TIME_S

        assist_prep = bool(
            (not penetration_blocks_assist)
            and (not geom_viol)
            and pen_ok
            and assist_bilateral_for_prep
            and settle_gate
        )

        assist_should_arm = bool(
            (not finger_place_only)
            and phase in ARM_ASSIST_PHASES
            and (not assist_finger_actuators_unsupported)
            and (not penetration_blocks_assist)
            and assist_prep
        )

        if geom_viol:
            assist_should_arm = False

        if assist_should_arm:
            if not assist_armed:
                assist_anchor = np.asarray(
                    det_met["box_center"],
                    dtype=float,
                ).reshape(3).copy()
            assist_armed = True
            grasp_assist_used = True

        _command_grasp_actuators(
            model,
            data,
            full_posture_from_dual(q_work, phase=phase),
            actuator_ids,
            proxy_slide_target=float(slide_tgt),
            use_legacy_proxy_finger_motors=not use_dex3_hands,
        )

        box_c = np.asarray(det_met["box_center"], dtype=float).reshape(3)
        box_max_z = max(box_max_z, float(box_c[2]))

        dex3_ftip_lift_ok = True
        if use_dex3_hands:
            dex3_ftip_lift_ok = bool(dex3_assist_contact_ok)

        dex3_wrist_lift_ok = (not use_dex3_hands) or bool(wrist_orientation_locked)

        dex3_lift_settle_ok = (not use_dex3_hands) or (
            dex3_real_contact_stable_s + 1e-9 >= float(DEX3_LIFT_CONTACT_SETTLE_TIME_S)
        )
        lift_contact_ok_for_assist = bool(
            bilateral_ok_frame
            or (
                use_dex3_hands
                and dex3_assist_contact_ok
                and palm_outside_vol
                and pen_ok
            )
        )

        assist_lift_gate = bool(
            (not finger_place_only)
            and assist_prep
            and squeeze_gate_cleared_t is not None
            and max_pen_any_episode <= pen_episode_cap_m + 1e-12
            and dex3_ftip_lift_ok
            and dex3_wrist_lift_ok
            and dex3_lift_settle_ok
            and lift_contact_ok_for_assist
        )

        data.xfrc_applied[:] = 0.0
        assist_active_flag = False
        assist_f_norm = 0.0
        gp_assist = gp_proxy
        if (
            assist_armed
            and (not no_assist)
            and (not assist_finger_actuators_unsupported)
            and phase in ACTIVE_ASSIST_PHASES
            and assist_lift_gate
        ):
            des = _assist_desired_xyz_dual(
                phase,
                anchor=assist_anchor,
                sim_t=sim_tm,
                duration=tout,
                post_spans=post_spans,
            )
            err = des - box_c
            if slide_dofadr >= 0:
                vz = float(data.qvel[slide_dofadr])
            else:
                vz = 0.0
            if gp_assist == GraspPhase.LIFT_TEST:
                ez = float(err[2])
                fz = ASSIST_LIFT_Z_SCALE * ASSIST_KP_POS * ez - ASSIST_KD_Z * vz
                f_lin = np.array([0.0, 0.0, fz], dtype=float)
            else:
                f_lin = ASSIST_KP_POS * err
                f_lin[2] -= ASSIST_KD_Z * vz
            f_lin[2] -= float(model.opt.gravity[2]) * float(model.body_subtreemass[box_bid])
            fn = float(np.linalg.norm(f_lin))
            if fn > MAX_ASSIST_FORCE_NEWTONS and fn > 1e-9:
                f_lin *= MAX_ASSIST_FORCE_NEWTONS / fn
                fn = MAX_ASSIST_FORCE_NEWTONS
            max_assist_force = max(max_assist_force, fn)
            data.xfrc_applied[box_bid, :3] = f_lin
            data.xfrc_applied[box_bid, 3:6] = 0.0
            assist_active_flag = True
            assist_f_norm = fn

        if assist_active_flag and gp_assist == GraspPhase.LIFT_TEST:
            lift_assist_applied_in_lift_phase = True

        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos)
        mujoco.mj_forward(model, data)

        jv_max = max(abs(float(data.qvel[i])) for i in range(model.nv))
        max_joint_vel_mag_episode = max(max_joint_vel_mag_episode, float(jv_max))

        wya = int(waist_qpos_tuple[0])
        wpa = int(waist_qpos_tuple[1])
        wqy = float(data.qpos[wya])
        wqp = float(data.qpos[wpa])
        min_waist_yaw_episode = min(min_waist_yaw_episode, wqy)
        max_waist_yaw_episode = max(max_waist_yaw_episode, wqy)
        min_waist_pitch_episode = min(min_waist_pitch_episode, wqp)
        max_waist_pitch_episode = max(max_waist_pitch_episode, wqp)

        dt_phys = float(model.opt.timestep)
        pr_phys = _point_world(
            model, data, site_id=right_site_id, fallback_body_id=right_bid
        )
        pl_phys = _point_world(
            model, data, site_id=left_site_id, fallback_body_id=left_bid
        )
        if prev_palm_r_phys is not None and prev_palm_l_phys is not None and dt_phys > 1e-9:
            vr = float(np.linalg.norm(pr_phys - prev_palm_r_phys) / dt_phys)
            vl = float(np.linalg.norm(pl_phys - prev_palm_l_phys) / dt_phys)
            if dr < NEAR_CONTACT_PALM_DIST_M:
                right_palm_vel_near_sum += vr
                right_palm_vel_near_n += 1
            if dl_tgt < NEAR_CONTACT_PALM_DIST_M:
                left_palm_vel_near_sum += vl
                left_palm_vel_near_n += 1
        prev_palm_r_phys = np.asarray(pr_phys, dtype=float).copy()
        prev_palm_l_phys = np.asarray(pl_phys, dtype=float).copy()

        box_max_z = max(
            box_max_z,
            float(
                _detect_box_dual(model, data, box_geom_name=box_geom_name)["box_center"][2]
            ),
        )

        bh = box_max_z - box_initial_z
        if gp_assist == GraspPhase.LIFT_TEST and assist_active_flag:
            lift_contact_during_lift = True
            if bh >= LIFT_SUCCESS_DELTA_Z_M:
                lift_success = True

        if verbose and not silent and (data.time - last_print >= print_interval_s):
            episode_diag["right_face_gap"] = float(sep_r)
            episode_diag["left_face_gap"] = float(sep_l)
            episode_diag["right_contact_stability_s"] = float(right_contact_stability_s)
            episode_diag["left_contact_stability_s"] = float(left_contact_stability_s)
            episode_diag["right_normal_alignment"] = float(right_normal_alignment)
            episode_diag["left_normal_alignment"] = float(left_normal_alignment)
            episode_diag["inside_box_violation"] = bool(inside_rp or inside_lp)
            episode_diag["cross_center_violation"] = bool(cross_rp or cross_lp)
            episode_diag["palm_overlap_risk"] = bool(overlap_risk)
            episode_diag["bilateral_stable_s"] = float(bilateral_stable_s)
            _aux_cmd = (
                f"dex3_phase={_dex3_timeline_phase_display(phase, use_dex3=use_dex3_hands)!r}  "
                f"dex3_finger={current_dex3_finger_mode_for_step!r}"
                if use_dex3_hands
                else f"proxy_slide_tgt={slide_tgt:.4f}"
            )
            phase_disp = _dex3_timeline_phase_display(phase, use_dex3=use_dex3_hands)
            print(
                f"phase={phase_disp}  palm_r={palm_r.tolist()}  palm_l={palm_l.tolist()}\n"
                f"          coordination={coord_mode.name}  right_act={right_stage_active}  left_act={left_stage_active}  "
                f"waist_frozen={waist_frozen_episode}\n"
                f"          dr={dr:.5f}  dl={dl_tgt:.5f}  rc={rc}  lc={lc}  "
                f"v_right={valid_right_contact}  v_left={valid_left_contact}\n"
                f"          sep_r={sep_r:.5f}  sep_l={sep_l:.5f}  gt_r={geom_touch_r}  gt_l={geom_touch_l}\n"
                f"          right_face_gap={sep_r:.5f}  left_face_gap={sep_l:.5f}  "
                f"right_stab={right_contact_stability_s:.3f}  left_stab={left_contact_stability_s:.3f}\n"
                f"          right_n_align={right_normal_alignment:.3f}  left_n_align={left_normal_alignment:.3f}  "
                f"bilateral_stable={bilateral_stable_s:.3f}\n"
                f"          pen_any={max_pen_any_frame:.5f}  {_aux_cmd}  "
                f"assist_active={assist_active_flag}  squeeze_t={squeeze_gate_cleared_t}\n"
                f"          box_dz={bh:.5f}  lift_ok={lift_success}  ik_err_r={err_r_fin:.5f}  ik_err_l={err_l_fin:.5f}\n"
                f"          ik_recovery_mode={ik_recovery_mode}  max_wrist_d={frame_wrist_max_delta:.5f}  "
                f"wrist_rl_r={right_wrist_rate_limited}  wrist_rl_l={left_wrist_rate_limited}",
                flush=True,
            )
            last_print = data.time


    def run_headless() -> None:
        nonlocal mission_complete
        step_counter = 0
        while data.time < float(timeout) and (
            max_steps_limit is None or step_counter < int(max_steps_limit)
        ):
            step_frame(silent=silent)
            step_counter += 1
            if (
                _dual_phase_effective(
                    float(data.time),
                    float(timeout),
                    squeeze_gate_cleared_t,
                    post_gate_virtual_lag_s=post_gate_freeze_accum,
                    finger_place_only=finger_place_only,
                )
                == DualPhase.DONE
            ):
                mission_complete = True
                return

    def run_viewer() -> None:
        nonlocal mission_complete
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            step_counter = 0
            while viewer.is_running() and (
                max_steps_limit is None or step_counter < int(max_steps_limit)
            ):
                loop_t0 = time.time()
                step_frame(silent=False)
                step_counter += 1
                if (
                    _dual_phase_effective(
                        float(data.time),
                        float(timeout),
                        squeeze_gate_cleared_t,
                        post_gate_virtual_lag_s=post_gate_freeze_accum,
                        finger_place_only=finger_place_only,
                    )
                    == DualPhase.DONE
                ):
                    mission_complete = True
                viewer.sync()
                sleep_s = max(
                    0.0,
                    float(model.opt.timestep) - (time.time() - loop_t0),
                ) / max(float(motion.viewer_playback_mul), 1.0)
                time.sleep(sleep_s)

    if headless:
        run_headless()
    else:
        run_viewer()

    mujoco.mj_forward(model, data)

    det_fin = _detect_box_dual(model, data, box_geom_name=box_geom_name)
    box_center_y = float(det_fin["box_center"][1])

    bh_final = box_max_z - box_initial_z
    squeeze_was_cleared = bool(squeeze_gate_cleared_t is not None)
    true_dual_contact_ready_out = bool(
        squeeze_was_cleared
        and np.isfinite(min_touch_r_dist)
        and np.isfinite(min_touch_l_dist)
        and min_touch_r_dist < RIGHT_PALM_CONTACT_GOAL_M
        and min_touch_l_dist <= LEFT_PALM_CONTACT_GOAL_M
        and max_pen_any_episode <= pen_episode_cap_m
        and max_bilateral_stable_episode + 1e-9 >= DUAL_CONTACT_SETTLE_TIME_S
        and (
            (max_right_c >= 1 and max_left_c >= 1)
            or (max_geom_pair_near_r_episode and max_geom_pair_near_l_episode)
            or (
                min_touch_r_dist < 0.012
                and min_touch_l_dist < 0.085
            )
        )
    )
    lift_raw = lift_success or (
        grasp_assist_used and lift_contact_during_lift and bh_final >= LIFT_SUCCESS_DELTA_Z_M
    )
    lift_final = bool(
        lift_raw
        and true_dual_contact_ready_out
        and bh_final >= LIFT_SUCCESS_DELTA_Z_M
        and max_pen_any_episode <= pen_episode_cap_m
    )
    reach_dual_ok = bool(
        np.isfinite(min_touch_r_dist)
        and np.isfinite(min_touch_l_dist)
        and (
            true_dual_contact_ready_out
            or (
                min_touch_r_dist < RIGHT_PALM_CONTACT_GOAL_M * 1.6
                and min_touch_l_dist < LEFT_PALM_CONTACT_GOAL_M * 1.6
            )
        )
    )
    penetration_ok_dual = bool(max_pen_any_episode <= pen_episode_cap_m)

    lift_blocked_out: str | None = None
    if lift_phase_reached and (not lift_assist_applied_in_lift_phase) and (not lift_final):
        if use_dex3_hands and max_dex3_real_contact_stable_episode + 1e-9 < float(
            DEX3_LIFT_CONTACT_SETTLE_TIME_S
        ):
            lift_blocked_out = "dex3_contact_settle_insufficient"
        else:
            lift_blocked_out = "bilateral_contact_not_stable"

    pelvis_bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    torso_payload_diag: dict[str, Any] = {
        "payload_mass_estimate_kg": float(model.body_subtreemass[box_bid]),
        "box_com_world_xyz": np.asarray(det_fin["box_center"], dtype=float).reshape(3).tolist(),
        "approx_horiz_moment_arm_pelvis_to_box_m": 0.0,
        "approx_torso_pitch_load_moment_nm": 0.0,
        "suggested_torso_pitch_compensation_rad": 0.0,
    }
    if pelvis_bid >= 0:
        px = np.asarray(data.xpos[pelvis_bid, :3], dtype=float)
        com_xyz = np.asarray(det_fin["box_center"], dtype=float).reshape(3)
        r = com_xyz - px
        arm_xy = float(np.hypot(float(r[0]), float(r[1])))
        m = float(model.body_subtreemass[box_bid])
        gmag = float(np.linalg.norm(model.opt.gravity))
        torso_payload_diag["approx_horiz_moment_arm_pelvis_to_box_m"] = arm_xy
        torso_payload_diag["approx_torso_pitch_load_moment_nm"] = float(m * gmag * arm_xy)
        torso_payload_diag["suggested_torso_pitch_compensation_rad"] = float(
            np.clip(-np.arctan2(float(com_xyz[0] - px[0]), 0.72), -0.14, 0.14)
        )

    ik_debounce_rec = int(right_recovery_events[0] + left_recovery_events[0])
    ik_rej_total = int(right_ik_rejects[0] + left_ik_rejects[0])
    wy_rng = (
        float(max_waist_yaw_episode - min_waist_yaw_episode)
        if max_waist_yaw_episode > min_waist_yaw_episode
        else 0.0
    )
    wp_rng = (
        float(max_waist_pitch_episode - min_waist_pitch_episode)
        if max_waist_pitch_episode > min_waist_pitch_episode
        else 0.0
    )
    avg_r_vel = float(right_palm_vel_near_sum / max(int(right_palm_vel_near_n), 1))
    avg_l_vel = float(left_palm_vel_near_sum / max(int(left_palm_vel_near_n), 1))

    episode_diag["ik_recovery_count"] = int(ik_debounce_rec)
    episode_diag["max_wrist_command_delta_episode"] = float(max_wrist_command_delta_episode)
    episode_diag["large_joint_jump_events"] = int(large_joint_jump_events)
    episode_diag["max_abs_joint_step_applied_episode"] = float(max_abs_joint_step_applied_episode)
    episode_diag["max_waist_delta_step_rad_episode"] = float(max_waist_delta_step_rad_episode)
    episode_diag["max_palm_convergence_speed_r_m_per_s"] = float(max_palm_convergence_speed_r)
    episode_diag["max_palm_convergence_speed_l_m_per_s"] = float(max_palm_convergence_speed_l)
    episode_diag["max_dual_goal_asymmetry_m"] = float(max_dual_goal_asymmetry_m)
    episode_diag["joint_jumps_by_phase"] = {k: int(v) for k, v in sorted(joint_jumps_by_phase.items())}
    episode_diag["max_right_joint_command_delta_episode"] = float(
        max_right_joint_command_delta_episode
    )
    episode_diag["max_filtered_pr_goal_distance_episode"] = float(
        max_filtered_pr_goal_dist_episode
    )
    episode_diag["max_right_bad_ik_streak_episode"] = int(max_right_bad_ik_streak_episode)
    episode_diag["max_joint_vel_mag_episode"] = float(max_joint_vel_mag_episode)
    episode_diag["max_per_frame_dual_cmd_delta_episode"] = float(
        max_per_frame_dual_cmd_delta_episode
    )
    episode_diag["ik_rejected_updates_total"] = int(ik_rej_total)
    episode_diag["ik_debounce_recovery_events"] = int(ik_debounce_rec)
    episode_diag["left_recovery_events"] = int(left_recovery_events[0])
    episode_diag["waist_yaw_range_episode_rad"] = float(wy_rng)
    episode_diag["waist_pitch_range_episode_rad"] = float(wp_rng)
    episode_diag["avg_right_palm_vel_near_contact_m_per_s"] = float(avg_r_vel)
    episode_diag["avg_left_palm_vel_near_contact_m_per_s"] = float(avg_l_vel)
    episode_diag["max_palm_contact_stable_time_episode"] = float(max_palm_contact_stable_s_diag_episode)
    episode_diag["max_finger_contact_stable_time_episode"] = float(
        max_finger_contact_stable_s_diag_episode
    )
    episode_diag["wrist_orientation_lock_duration_s"] = float(wrist_orientation_lock_duration_s)
    episode_diag["wrist_orientation_ever_locked"] = bool(wrist_orientation_ever_locked)
    episode_diag["max_right_wrist_orientation_error_episode"] = float(
        wrist_lock_max_axis_err_r_episode
    )
    episode_diag["max_left_wrist_orientation_error_episode"] = float(
        wrist_lock_max_axis_err_l_episode
    )
    episode_diag["min_right_index_tip_distance_episode"] = (
        float(min_right_index_tip_dist_episode)
        if np.isfinite(min_right_index_tip_dist_episode)
        else float("nan")
    )
    episode_diag["min_left_index_tip_distance_episode"] = (
        float(min_left_index_tip_dist_episode)
        if np.isfinite(min_left_index_tip_dist_episode)
        else float("nan")
    )
    episode_diag["min_right_thumb_tip_distance_episode"] = (
        float(min_right_thumb_tip_dist_episode)
        if np.isfinite(min_right_thumb_tip_dist_episode)
        else float("nan")
    )
    episode_diag["min_left_thumb_tip_distance_episode"] = (
        float(min_left_thumb_tip_dist_episode)
        if np.isfinite(min_left_thumb_tip_dist_episode)
        else float("nan")
    )

    out: dict[str, Any] = {
        "right_palm_distance_to_target": float(min_touch_r_dist),
        "left_palm_distance_to_target": float(min_touch_l_dist),
        "final_right_palm_distance_to_target": float(palm_dist_final_dr),
        "final_left_palm_distance_to_target": float(palm_dist_final_dl),
        "box_center_y": box_center_y,
        "right_contact_count_max": int(max_right_c),
        "left_contact_count_max": int(max_left_c),
        "dual_contact_ready": true_dual_contact_ready_out,
        "true_dual_contact_ready": true_dual_contact_ready_out,
        "squeeze_gate_was_cleared": bool(squeeze_was_cleared),
        "max_right_contact_count_episode": int(max_right_c),
        "max_left_contact_count_episode": int(max_left_c),
        "proximity_fallback_used": False,
        "dual_contact_via_proximity_fallback": False,
        "proximity_fallback_used_for_success": False,
        "max_box_penetration_any_geom": float(max_pen_any_episode),
        "right_palm_penetration": float(max_pen_right_palm_episode),
        "left_palm_penetration": float(max_pen_left_palm_episode),
        "right_proxy_penetration": float(max_pen_right_proxy_episode),
        "left_proxy_penetration": float(max_pen_left_proxy_episode),
        "left_target_mode": left_target_mode,
        "grasp_assist_used": bool(grasp_assist_used),
        "no_assist_mode": bool(no_assist),
        "box_height_delta": float(bh_final),
        "lift_success": bool(lift_final),
        "max_penetration_depth_right": float(max_pen_right_palm_episode),
        "max_penetration_depth_left": float(max_pen_left_palm_episode),
        "assist_mode": (
            "xfrc_bounded_stabilizer_dex3_fingers"
            if use_dex3_hands
            else (
                "xfrc_bounded_stabilizer"
                if not assist_finger_actuators_unsupported
                else "finger_actuators_present_but_unimplemented"
            )
        ),
        "use_dex3_pipeline": bool(use_dex3_hands),
        "sim_physics_timestep_s": float(applied_dt),
        "sim_timeline_speed": float(motion.timeline_speed),
        "sim_fast_profile": bool(fast),
        "finger_place_only": bool(finger_place_only),
        "max_dex3_digits_on_box_count": int(max_dex3_digits_on_box_count),
        "dex3_digits_frozen_joint_count": int(
            len(finger_controller.frozen_joint_snapshot()) if finger_controller else 0
        ),
        "all_dex3_digits_on_box": bool(max_dex3_digits_on_box_count >= 6),
        "dual_episode_penetration_cap_m": float(pen_episode_cap_m),
        "dual_pregrasp_reach_metric_ok": reach_dual_ok,
        "dual_penetration_ok": penetration_ok_dual,
        "motion_complete": bool(mission_complete),
        "phase": final_phase.name,
        "initial_contact_any": bool(initial_contact_any),
        "initial_contact_before_approach": bool(initial_contact_any),
        "max_assist_force": float(max_assist_force),
        "sim_time": float(data.time),
        "dual_contact_settle_time_required_s": float(DUAL_CONTACT_SETTLE_TIME_S),
        "max_bilateral_stable_s": float(max_bilateral_stable_episode),
        "max_right_contact_stability_s": float(max_right_contact_stability_episode),
        "max_left_contact_stability_s": float(max_left_contact_stability_episode),
        "dual_surface_clearance_y_m": float(DUAL_SURFACE_CONTACT_CLEARANCE_Y_M),
        "lift_attempted": bool(lift_assist_applied_in_lift_phase),
        "lift_blocked_reason": lift_blocked_out,
        "box_half_size": np.asarray(det_fin["box_half_size"], dtype=float).reshape(3).tolist(),
        "torso_payload_diag": torso_payload_diag,
        "dex3_min_cross_hand_fingertip_m": float(min_cross_hand_fingertip_episode_m)
        if np.isfinite(min_cross_hand_fingertip_episode_m)
        else float("nan"),
        "dex3_min_right_fingertip_clearance_m": float(min_right_fingertip_clearance_episode_m)
        if np.isfinite(min_right_fingertip_clearance_episode_m)
        else float("nan"),
        "dex3_min_left_fingertip_clearance_m": float(min_left_fingertip_clearance_episode_m)
        if np.isfinite(min_left_fingertip_clearance_episode_m)
        else float("nan"),
        "dex3_max_right_fingertip_contacts": int(max_dex3_right_fingertip_contacts_episode),
        "dex3_max_left_fingertip_contacts": int(max_dex3_left_fingertip_contacts_episode),
        "dex3_max_real_contact_stable_s": float(max_dex3_real_contact_stable_episode),
        "dex3_contact_stable_decay_events": int(dex3_contact_stable_decay_events),
        "wrist_orientation_locked": bool(wrist_orientation_ever_locked),
        "wrist_orientation_lock_time": (
            float(wrist_lock_snap_time) if wrist_lock_snap_time is not None else float("nan")
        ),
        "right_wrist_orientation_error": float(wrist_lock_max_axis_err_r_episode),
        "left_wrist_orientation_error": float(wrist_lock_max_axis_err_l_episode),
        "palm_contact_stable_time": float(max_palm_contact_stable_s_diag_episode),
        "finger_contact_stable_time": float(max_finger_contact_stable_s_diag_episode),
        "right_index_tip_distance": episode_diag.get("right_index_tip_distance", float("nan")),
        "left_index_tip_distance": episode_diag.get("left_index_tip_distance", float("nan")),
        "right_thumb_tip_distance": episode_diag.get("right_thumb_tip_distance", float("nan")),
        "left_thumb_tip_distance": episode_diag.get("left_thumb_tip_distance", float("nan")),
        "dex3_phase_name": str(episode_diag.get("dex3_timeline_phase_display", final_phase.name)),
        "finger_closure_blocked_reason": episode_diag.get("finger_closure_blocked_reason"),
        "ik_recovery_count": int(ik_debounce_rec),
        "max_wrist_command_delta_episode": float(max_wrist_command_delta_episode),
        "large_joint_jump_events": int(large_joint_jump_events),
        "max_abs_joint_step_applied_episode": float(max_abs_joint_step_applied_episode),
        "bilateral_ok_frame_count": int(bilateral_ok_frame_count),
        "strict_dual_pose_frame_count": int(strict_dual_pose_frame_count),
        "dual_palm_extra_near_face_clearance_x_m": float(DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M),
        "penetration_phase_peak": {k: dict(v) for k, v in phase_pen_peak.items()},
        **episode_diag,
    }

    if dbg_penetration:
        _print_penetration_phase_report(
            phase_pen_peak,
            dual_extra_clearance_x_m=DUAL_PALM_EXTRA_NEAR_FACE_CLEARANCE_X_M,
        )

    if verbose and (not silent):
        print(
            "----- dual_arm_box summary -----\n"
            f"true_dual_contact_ready: {out['true_dual_contact_ready']}\n"
            f"max_box_penetration_any_geom: {out['max_box_penetration_any_geom']:.5f}\n"
            f"proximity_fallback_used_for_success: {out['proximity_fallback_used_for_success']}\n"
            f"left_target_mode: {out['left_target_mode']}\n"
            f"right_palm_distance_to_target: {out['right_palm_distance_to_target']:.5f}\n"
            f"left_palm_distance_to_target: {out['left_palm_distance_to_target']:.5f}\n"
            f"lift_success: {out['lift_success']}\n"
            f"box_height_delta: {out['box_height_delta']:.5f}\n"
            f"right_palm_penetration: {out['right_palm_penetration']:.5f}\n"
            f"left_palm_penetration: {out['left_palm_penetration']:.5f}"
        )

    if verbose and (not silent):
        _print_episode_stability_summary(out)

    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="G1 locked-base dual-arm box coordination demo.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=DEFAULT_DUAL_ARM_TIMEOUT_S)
    ap.add_argument("--pelvis-z", type=float, default=DEFAULT_PELVIS_Z)
    ap.add_argument("--posture-gain", type=float, default=IK_POSTURE_GAIN)
    ap.add_argument(
        "--max-joint-from-neutral",
        type=float,
        default=IK_MAX_ABS_JOINT_FROM_NEUTRAL + 0.40,
    )
    ap.add_argument("--dbg-markers", action="store_true")
    ap.add_argument(
        "--dbg-penetration",
        action="store_true",
        help="Print per-phase penetration peaks and palm/plate diagnostics.",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress startup geometry print and per-step logs.")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="Cap simulated horizon for quicker smoke runs (see FAST_DUAL_ARM_TIMEOUT_CAP_S).",
    )
    ap.add_argument(
        "--allow-lift",
        action="store_true",
        help="Enable lift/lower timeline and bounded box assist (default: finger placement only).",
    )
    ap.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable bounded xfrc_applied lift assist; keep IK/finger behavior unchanged.",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="Stop after N physics steps (headless/viewer).")
    ap.add_argument("--print-every", type=float, default=None, help="Seconds between progress prints when verbose.")
    ns = ap.parse_args(argv)

    try:
        run_g1_dual_arm_box(
            headless=ns.headless,
            timeout=ns.timeout,
            initial_pelvis_z=ns.pelvis_z,
            verbose=not ns.quiet,
            quiet=ns.quiet,
            fast=ns.fast,
            max_steps=ns.max_steps,
            print_every_s=ns.print_every,
            posture_gain=ns.posture_gain,
            max_joint_from_neutral=ns.max_joint_from_neutral,
            dbg_markers=ns.dbg_markers,
            dbg_penetration=ns.dbg_penetration,
            finger_place_only=not bool(ns.allow_lift),
            no_assist=ns.no_assist,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
