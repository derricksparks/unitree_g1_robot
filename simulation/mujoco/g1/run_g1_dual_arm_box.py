#!/usr/bin/env python3
"""
Locked-base **dual-arm** box coordination (IK + dual-side contact + bounded ``xfrc_applied`` assist).

Uses ``g1_reach_box_scene.xml`` (dual-milestone centered box), :class:`~perception.g1_box_perception.G1BoxPerception`, and the
same vertical box slide injection as ``run_g1_grasp_box.py``. IK is **prioritized**: the right arm
settles the near-face goal first, then the waist/left chain is interleaved so the left palm can
seek **symmetric near-face lateral** support targets (same −x face, ``box_center_y ± offset``)—not far-face grasps.

**Assist / lift gating** requires geometric dual readiness **at squeeze clearance**: palms stay in allowed outside ±y strips,
MuJoCo ``mj_geomDistance`` separation (or native contacts) shows palm/proxy geoms flush with the box within a mm-scale budget,
penetration into the box AABB stays ≤ ``MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M``, and there was no contact before approach.
Lift-phase ``xfrc`` stays enabled while penetration budgets remain satisfied (squeeze clearance already certified dual-sided touch).
Palm–IK norms (**right < 7 cm**, **left < 10 cm**) define ``strict_dual_pose`` for timeline squeeze clearance.

The timeline is **staged**: nominal approach/descend, then **DUAL_CONTACT_HOLD** until **both**
arms satisfy palm–target thresholds; only **afterwards** does the scripted ``CLOSE_PROXY_HANDS``
segment begin.

Translucent fixed ``left_proxy_*`` geoms and ``right_proxy_*`` sliders are **simulated** palms /
fingers—not Unitree articulated hands. Assisted lift remains a bounded wrench on ``reach_target_box``
after bilateral contact cues.

Run::

    python simulation/mujoco/g1/run_g1_dual_arm_box.py
    python simulation/mujoco/g1/run_g1_dual_arm_box.py --headless --timeout 18 --dbg-markers

"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
import tempfile
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
    APPROACH_ABOVE_OFFSET,
    MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M,
    OUTSIDE_Y_CLEARANCE,
    PALM_PLATE_HALF_THICKNESS_X,
    PALM_PLATE_HALF_WIDTH_Y,
    PALM_SURFACE_CLEARANCE,
    geom_max_penetration_into_world_axis_aligned_box,
)
from run_g1_grasp_box import (  # noqa: E402 — shared injector + actuator merge + proxy ramps
    ASSIST_KP_POS,
    ASSIST_KD_Z,
    ASSIST_LIFT_Z_SCALE,
    BOX_SLIDE_JOINT_NAME,
    GraspPhase,
    LIFT_PALM_DELTA_Z_M,
    LIFT_SUCCESS_DELTA_Z_M,
    LIFT_ASSIST_TARGET_DZ_M,
    MAX_ASSIST_FORCE_NEWTONS,
    BOX_BODY_NAME,
    BOX_GEOM_NAME,
    LEGACY_BOX_GEOM_NAME,
    BASE_G1_XML,
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
    floating_base_address_map,
    stabilize_floating_base,
)
from run_g1_right_arm_ik_demo import (  # noqa: E402
    IK_POSTURE_GAIN,
    IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    IK_JOINT_NAMES as RIGHT_IK_JOINT_NAMES_ONLY,
    _apply_kp_scale_for_joint_subset,
    _build_ik_metadata as _build_right_ik_metadata,
    solve_ik_q,
)

RIGHT_SITE = "right_palm_site"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_SITE = "left_palm_site"
LEFT_WRIST_BODY = "left_wrist_yaw_link"

RIGHT_PALM_GEOM = "right_palm_contact_geom"
LEFT_PALM_GEOM = "left_palm_contact_geom"
LEFT_PROXY_GEOMS = ("left_proxy_left_finger_geom", "left_proxy_right_finger_geom")
DUAL_IK_KP_SCALE = 0.74

# Alternating IK is Jacobian-heavy; fewer inner NR steps keep headless timelines practical.
DUAL_ALT_IK_INNER_ITERS = 7
DUAL_HOLD_IK_INNER_ITERS = 9

# Palm–IK targets (instantaneous norms after mj_forward during the physics step).
RIGHT_PALM_CONTACT_GOAL_M = 0.070
LEFT_PALM_CONTACT_GOAL_M = 0.100

# Dual milestone: outside ±y palm strips + capped proxy slide travel (see MJCF slide ranges).
DUAL_PROXY_MAX_SLIDE_M = 0.011
# Allowed ±y deviation from nominal outside-edge palm strips (faces at ymin / ymax).
EDGE_CONTACT_Y_PAD_INBOARD_M = 0.025
EDGE_CONTACT_Y_PAD_OUTBOARD_M = 0.035
# Separation threshold for mj_geomDistance (negative slight overlap allowed; capped by penetration checks).
GEOM_PAIR_TOUCH_DIST_OK_M = 0.004
# Dual motion: defer left IK until right is nearly on the scripted near-face palm target.
_RIGHT_ENGAGE_LEFT_ARM_M = 0.110

# Post-contact-hold fractions of remaining sim budget (after squeeze gate clears).
_POST_GATE_FRACS: tuple[tuple[str, float], ...] = (
    ("close", 0.045),
    ("grasp_hold", 0.028),
    ("lift", 0.465),
    ("lower", 0.148),
    ("open", 0.058),
    ("retreat", 0.256),
)

DUAL_IK_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_pitch_joint",
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
ARM_ASSIST_PHASES = frozenset({DualPhase.DUAL_CONTACT_HOLD, DualPhase.CLOSE_PROXY_HANDS})
ACTIVE_ASSIST_PHASES = frozenset(
    {
        DualPhase.DUAL_GRASP_HOLD,
        DualPhase.DUAL_LIFT_TEST,
        DualPhase.DUAL_LOWER_BACK,
    }
)


def _dual_phase_boundaries(duration: float) -> dict[str, float]:
    d = max(float(duration), 1e-6)
    return {
        "t_stow_end": 0.09 * d,
        "t_approach_end": 0.28 * d,
        "t_descend_end": 0.44 * d,
        "t_contact_hold_end": 0.58 * d,
        # Legacy timings (informational — post-squeeze pacing uses ``_POST_GATE_FRACS``).
        "t_close_proxy_end": 0.62 * d,
        "t_grasp_hold_end": 0.64 * d,
        "t_lift_end": 0.84 * d,
        "t_lower_end": 0.90 * d,
        "t_open_end": 0.92 * d,
        "t_retreat_end": 0.973 * d,
    }


def _dual_phase_effective(
    sim_t: float,
    duration: float,
    squeeze_cleared_t: float | None,
    *,
    post_gate_virtual_lag_s: float = 0.0,
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
    return _post_gate_budget_and_phase(dt, pb)


def _post_gate_abs_spans(
    squeeze_t: float, duration: float
) -> dict[str, tuple[float, float]]:
    dur = max(float(duration), 1e-9)
    t0 = float(squeeze_t)
    pb = max(dur - t0, 1e-9)
    wt = sum(w for _, w in _POST_GATE_FRACS)
    t = t0
    out: dict[str, tuple[float, float]] = {}
    for key, frac in _POST_GATE_FRACS:
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


def _post_gate_budget_and_phase(dt_after_gate: float, post_budget: float) -> DualPhase:
    """Map elapsed time since squeeze gate clears into post-contact timeline."""
    dt = float(max(0.0, dt_after_gate))
    pb = float(max(post_budget, 1e-6))
    wt = sum(w for _, w in _POST_GATE_FRACS)
    acc = 0.0
    for key, frac in _POST_GATE_FRACS:
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


def _dual_ik_dof_adrs(model: mujoco.MjModel) -> list[int]:
    out: list[int] = []
    for jn in DUAL_IK_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        out.append(int(model.jnt_dofadr[jid]))
    return out


def _build_left_ik_metadata(model: mujoco.MjModel) -> tuple[list[int], np.ndarray, np.ndarray]:
    qadrs: list[int] = []
    lows: list[float] = []
    highs: list[float] = []
    for jn in LEFT_IK_JOINT_NAMES_ONLY:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"Missing joint {jn}")
        qadrs.append(int(model.jnt_qposadr[jid]))
        lows.append(float(model.jnt_range[jid, 0]))
        highs.append(float(model.jnt_range[jid, 1]))
    return qadrs, np.asarray(lows), np.asarray(highs)


def _apply_chain_positions(
    model: mujoco.MjModel, data: mujoco.MjData, adrs: list[int], vals: np.ndarray
) -> None:
    for i, adr in enumerate(adrs):
        data.qpos[adr] = float(vals[i])


def _solve_prioritized_dual_arms(
    model: mujoco.MjModel,
    fd: mujoco.MjData,
    data: mujoco.MjData,
    *,
    phase: DualPhase,
    r_qpos_adrs: list[int],
    r_low: np.ndarray,
    r_high: np.ndarray,
    q_neutral_r: np.ndarray,
    l_qpos_adrs: list[int],
    l_low: np.ndarray,
    l_high: np.ndarray,
    q_neutral_l: np.ndarray,
    target_right: np.ndarray,
    target_left: np.ndarray,
    right_site_id: int,
    right_body_id: int,
    left_site_id: int,
    left_body_id: int,
    posture_gain: float,
    max_abs_dn: float,
    engage_left: bool = True,
) -> tuple[float, float]:
    """Emphasize the near-face (right) reach early, then interleave waist+left refinement."""
    err_r = err_l = 0.0

    emphasize_right_early = phase in frozenset(
        {
            DualPhase.STOW,
            DualPhase.DUAL_APPROACH,
            DualPhase.DUAL_DESCEND_TO_PREGRASP,
        }
    )

    if not engage_left:
        for _ in range(10):
            q_wr, err_r = solve_ik_q(
                model,
                fd,
                data,
                r_qpos_adrs,
                r_low,
                r_high,
                right_body_id,
                target_right,
                q_neutral_r,
                posture_gain=posture_gain,
                max_abs_joint_from_neutral=max_abs_dn,
                target_site_id=right_site_id if right_site_id >= 0 else -1,
                inner_iters=DUAL_ALT_IK_INNER_ITERS,
            )
            _apply_chain_positions(model, data, r_qpos_adrs, q_wr)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        return float(err_r), float(err_l)

    if emphasize_right_early:
        for _ in range(3):
            q_wr, err_r = solve_ik_q(
                model,
                fd,
                data,
                r_qpos_adrs,
                r_low,
                r_high,
                right_body_id,
                target_right,
                q_neutral_r,
                posture_gain=posture_gain,
                max_abs_joint_from_neutral=max_abs_dn,
                target_site_id=right_site_id if right_site_id >= 0 else -1,
                inner_iters=DUAL_ALT_IK_INNER_ITERS,
            )
            _apply_chain_positions(model, data, r_qpos_adrs, q_wr)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        for _ in range(3):
            q_wl, err_l = solve_ik_q(
                model,
                fd,
                data,
                l_qpos_adrs,
                l_low,
                l_high,
                left_body_id,
                target_left,
                q_neutral_l,
                posture_gain=posture_gain,
                max_abs_joint_from_neutral=max_abs_dn,
                target_site_id=left_site_id if left_site_id >= 0 else -1,
                inner_iters=DUAL_ALT_IK_INNER_ITERS,
            )
            _apply_chain_positions(model, data, l_qpos_adrs, q_wl)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            q_wr, err_r = solve_ik_q(
                model,
                fd,
                data,
                r_qpos_adrs,
                r_low,
                r_high,
                right_body_id,
                target_right,
                q_neutral_r,
                posture_gain=posture_gain,
                max_abs_joint_from_neutral=max_abs_dn,
                target_site_id=right_site_id if right_site_id >= 0 else -1,
                inner_iters=DUAL_ALT_IK_INNER_ITERS,
            )
            _apply_chain_positions(model, data, r_qpos_adrs, q_wr)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

        return float(err_r), float(err_l)

    ik_dual = (
        DUAL_HOLD_IK_INNER_ITERS
        if phase == DualPhase.DUAL_CONTACT_HOLD
        else DUAL_ALT_IK_INNER_ITERS
    )

    for _ in range(3):
        q_wl, err_l = solve_ik_q(
            model,
            fd,
            data,
            l_qpos_adrs,
            l_low,
            l_high,
            left_body_id,
            target_left,
            q_neutral_l,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_abs_dn,
            target_site_id=left_site_id if left_site_id >= 0 else -1,
            inner_iters=ik_dual,
        )
        _apply_chain_positions(model, data, l_qpos_adrs, q_wl)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        q_wr, err_r = solve_ik_q(
            model,
            fd,
            data,
            r_qpos_adrs,
            r_low,
            r_high,
            right_body_id,
            target_right,
            q_neutral_r,
            posture_gain=posture_gain,
            max_abs_joint_from_neutral=max_abs_dn,
            target_site_id=right_site_id if right_site_id >= 0 else -1,
            inner_iters=ik_dual,
        )
        _apply_chain_positions(model, data, r_qpos_adrs, q_wr)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    return float(err_r), float(err_l)


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


def _left_touch_geom_ids(model: mujoco.MjModel) -> set[int]:
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


def _right_touch_geom_ids_box(model: mujoco.MjModel) -> tuple[set[int], str]:
    gids, mode = _right_touch_contact_gids_grasp(model)
    return gids, mode


def run_g1_dual_arm_box(
    *,
    headless: bool = False,
    timeout: float = 18.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL + 0.40,
    dbg_markers: bool = False,
) -> dict[str, Any]:
    if not BASE_G1_XML.is_file():
        raise FileNotFoundError(f"Missing packaged model: {BASE_G1_XML}")

    model = _load_dual_scene_model()
    if model.nu != 31:
        raise RuntimeError(f"Expected nu=31 (29 hinge + 2 right proxy sliders), got {model.nu}")

    hinge_names = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if nm:
            hinge_names.add(nm)
    if set(NEUTRAL_POSTURE.keys()) != hinge_names:
        raise ValueError("NEUTRAL_POSTURE must list all hinges")

    finger_joints = _has_actuated_finger_joints(model)
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
    r_qpos_adrs, r_low, r_high = _build_right_ik_metadata(model)
    l_qpos_adrs, l_low, l_high = _build_left_ik_metadata(model)
    q_neutral_r = np.array([float(NEUTRAL_POSTURE[jn]) for jn in RIGHT_IK_JOINT_NAMES_ONLY])
    q_neutral_l = np.array([float(NEUTRAL_POSTURE[jn]) for jn in LEFT_IK_JOINT_NAMES_ONLY])

    nominal_base_qpos = apply_neutral_pose(
        model,
        data,
        initial_pelvis_z=initial_pelvis_z,
        neutral=NEUTRAL_POSTURE,
        base_map=base_map,
    )
    mujoco.mj_forward(model, data)

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

    slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_SLIDE_JOINT_NAME)
    slide_dofadr = int(model.jnt_dofadr[slide_jid]) if slide_jid >= 0 else -1

    box_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, box_gid) or BOX_GEOM_NAME

    if verbose:
        print(
            "\nNOTE: Dual-arm coordinated assist uses **bounded** ``xfrc_applied`` (no real grippers).\n"
            "Translucent proxies are simulated visualization / contact cues only—not Unitree fingers.\n"
            f"right_touch_detection={detect_mode_r!r}  slide_joint="
            f"{BOX_SLIDE_JOINT_NAME if slide_jid >= 0 else 'MISSING'}\n"
            f"touches: right_geoms={_geom_names(model, touch_r)}  left_geoms={_geom_names(model, touch_l)}\n"
        )

    box0 = np.asarray(G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)["box_center"])
    det0 = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
    box_initial_z = float(box0[2])
    box_max_z = box_initial_z

    right_pre0 = np.asarray(det0["right_dual_pregrasp_target"], dtype=float)
    left_pre0 = np.asarray(det0["left_dual_pregrasp_target"], dtype=float)
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
    dual_geom_ok_for_lift = False
    max_pen_right_palm_episode = 0.0
    max_pen_left_palm_episode = 0.0
    max_pen_right_proxy_episode = 0.0
    max_pen_left_proxy_episode = 0.0
    max_right_c = max_left_c = 0
    lift_success = False
    lift_contact_during_lift = False
    max_assist_force = 0.0
    assist_active_flag = False
    last_print = -PRINT_INTERVAL
    post_gate_freeze_accum = 0.0

    palm_dist_final_dr = float("nan")
    palm_dist_final_dl = float("nan")
    mission_complete = False
    final_phase = DualPhase.STOW

    # Warm-start toward neutral then first dual solve pulls both arms inward.
    for adr in ik_dof_adrs:
        data.qvel[adr] = 0.0

    def full_posture_from_dual(q_work: np.ndarray) -> dict[str, float]:
        full = dict(NEUTRAL_POSTURE)
        for jn, qv in zip(DUAL_IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        return full

    def step_frame(*, silent: bool = False) -> None:
        nonlocal squeeze_gate_cleared_t
        nonlocal finger_close_t0, finger_open_t0, last_proxy_phase_e
        nonlocal assist_armed, assist_anchor, grasp_assist_used, penetration_blocks_assist
        nonlocal max_pen_any_episode, dual_geom_ok_for_lift
        nonlocal max_pen_right_palm_episode, max_pen_left_palm_episode
        nonlocal max_pen_right_proxy_episode, max_pen_left_proxy_episode
        nonlocal max_right_c, max_left_c
        nonlocal lift_success, lift_contact_during_lift
        nonlocal box_max_z, assist_active_flag, max_assist_force
        nonlocal mission_complete, last_print
        nonlocal final_phase, min_touch_r_dist, min_touch_l_dist
        nonlocal post_gate_freeze_accum, palm_dist_final_dr, palm_dist_final_dl

        tout = float(timeout)
        sim_tm = float(data.time)

        phase = _dual_phase_effective(
            sim_tm,
            tout,
            squeeze_gate_cleared_t,
            post_gate_virtual_lag_s=post_gate_freeze_accum,
        )
        final_phase = phase

        sim_t_ik = (
            sim_tm - post_gate_freeze_accum
            if squeeze_gate_cleared_t is not None
            else sim_tm
        )

        post_spans = (
            _post_gate_abs_spans(squeeze_gate_cleared_t, tout)
            if squeeze_gate_cleared_t is not None
            else None
        )

        gp_proxy = _grasp_phase_for_proxy(phase)
        det_goal = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
        lt_touch = np.asarray(det_goal["left_dual_pregrasp_target"], dtype=float)
        tgt_r_pre = np.asarray(det_goal["right_dual_pregrasp_target"], dtype=float)
        palm_rp = _point_world(
            model, data, site_id=right_site_id, fallback_body_id=right_bid
        )
        dr_pre = float(np.linalg.norm(palm_rp - tgt_r_pre))
        rc_pre = _contact_count_set(data, box_gid, touch_r)

        engage_left = (
            (phase not in PRE_RIGHT_ONLY_PHASES)
            or (dr_pre < _RIGHT_ENGAGE_LEFT_ARM_M)
            or (rc_pre > 0)
        )

        if phase != last_proxy_phase_e:
            if phase == DualPhase.CLOSE_PROXY_HANDS:
                finger_close_t0 = sim_tm
            if phase == DualPhase.OPEN_PROXY_HANDS:
                finger_open_t0 = sim_tm
            last_proxy_phase_e = phase

        slide_tgt_raw = _proxy_finger_slide_target(
            gp_proxy,
            sim_tm,
            tout,
            close_t0=finger_close_t0,
            open_t0=finger_open_t0,
        )
        slide_tgt = float(min(slide_tgt_raw, DUAL_PROXY_MAX_SLIDE_M))

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
        left_target_eff = pl_traj if engage_left else p_stow_l.copy()

        err_r_fin, err_l_fin = _solve_prioritized_dual_arms(
            model,
            fd,
            data,
            phase=phase,
            r_qpos_adrs=r_qpos_adrs,
            r_low=r_low,
            r_high=r_high,
            q_neutral_r=q_neutral_r,
            l_qpos_adrs=l_qpos_adrs,
            l_low=l_low,
            l_high=l_high,
            q_neutral_l=q_neutral_l,
            target_right=pr_goal,
            target_left=left_target_eff,
            right_site_id=right_site_id,
            right_body_id=right_bid,
            left_site_id=left_site_id,
            left_body_id=left_bid,
            posture_gain=posture_gain,
            max_abs_dn=max_joint_from_neutral,
            engage_left=engage_left,
        )
        half_err_l = 0.5 * (err_r_fin + err_l_fin)

        for adr in ik_dof_adrs:
            data.qvel[adr] = 0.0

        q_work = np.array([float(data.qpos[adr]) for adr in ik_qpos_adrs], dtype=float)

        if dbg_markers:
            det_vis = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
            G1BoxPerception.sync_debug_marker_sites(
                model, data, det_vis, box_body_name=BOX_BODY_NAME
            )
            mujoco.mj_forward(model, data)

        det_met = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)

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

        near_x = float(det_met["near_face_x"])
        ly = float(det_met["left_edge_y"])
        ry_edge = float(det_met["right_edge_y"])
        bz = float(det_met["bottom_z"])
        tz = float(det_met["top_z"])
        fx = float(det_met["far_face_x"])

        rc = _contact_count_set(data, box_gid, touch_r)
        lc = _contact_count_set(data, box_gid, touch_l)
        max_right_c = max(max_right_c, rc)
        max_left_c = max(max_left_c, lc)

        xmin = float(det_met["box_x_min"])
        xmax = float(det_met["box_x_max"])
        ymin = float(det_met["box_y_min"])
        ymax = float(det_met["box_y_max"])

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

        prx_ids: list[int] = []
        for nm in ("right_proxy_left_finger_geom", "right_proxy_right_finger_geom"):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if g >= 0:
                prx_ids.append(int(g))
        plx_ids: list[int] = []
        for nm in LEFT_PROXY_GEOMS:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if g >= 0:
                plx_ids.append(int(g))

        pen_rpx = max((_pen_gid(g) for g in prx_ids), default=0.0)
        pen_lpx = max((_pen_gid(g) for g in plx_ids), default=0.0)

        max_pen_any_frame = float(max(pen_rp, pen_lp, pen_rpx, pen_lpx))
        max_pen_any_episode = max(max_pen_any_episode, max_pen_any_frame)
        max_pen_right_palm_episode = max(max_pen_right_palm_episode, pen_rp)
        max_pen_left_palm_episode = max(max_pen_left_palm_episode, pen_lp)
        max_pen_right_proxy_episode = max(max_pen_right_proxy_episode, pen_rpx)
        max_pen_left_proxy_episode = max(max_pen_left_proxy_episode, pen_lpx)

        if max_pen_any_frame > MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M and phase in PRE_GATE:
            penetration_blocks_assist = True

        sep_r = _min_geom_distance_to_box(
            model, data, box_gid=box_gid, touch_gids=touch_r
        )
        sep_l = _min_geom_distance_to_box(
            model, data, box_gid=box_gid, touch_gids=touch_l
        )
        geom_touch_r = sep_r <= GEOM_PAIR_TOUCH_DIST_OK_M
        geom_touch_l = sep_l <= GEOM_PAIR_TOUCH_DIST_OK_M

        # Outside-y strip: ymin face (robot right strip) vs ymax face (robot left strip).
        ry_face_y = ly
        ly_face_y = ry_edge
        palm_x_tol = near_x + 0.045
        right_strip_min = (
            ry_face_y
            - PALM_PLATE_HALF_WIDTH_Y
            - OUTSIDE_Y_CLEARANCE
            - EDGE_CONTACT_Y_PAD_OUTBOARD_M
        )
        right_strip_max = ry_face_y + EDGE_CONTACT_Y_PAD_INBOARD_M
        left_strip_min = ly_face_y - EDGE_CONTACT_Y_PAD_INBOARD_M
        left_strip_max = (
            ly_face_y
            + PALM_PLATE_HALF_WIDTH_Y
            + OUTSIDE_Y_CLEARANCE
            + EDGE_CONTACT_Y_PAD_OUTBOARD_M
        )

        valid_right_contact = bool(
            (rc > 0 or geom_touch_r)
            and right_strip_min <= float(palm_r[1]) <= right_strip_max
            and float(palm_r[0]) <= palm_x_tol
        )
        valid_left_contact = bool(
            (lc > 0 or geom_touch_l)
            and left_strip_min <= float(palm_l[1]) <= left_strip_max
            and float(palm_l[0]) <= palm_x_tol
        )

        strict_dual_pose = dr < RIGHT_PALM_CONTACT_GOAL_M and dl_tgt < LEFT_PALM_CONTACT_GOAL_M

        squeeze_ready_geometric = (
            (not initial_contact_any)
            and strict_dual_pose
            and valid_right_contact
            and valid_left_contact
            and max_pen_any_frame <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
        )

        if squeeze_gate_cleared_t is None and phase == DualPhase.DUAL_CONTACT_HOLD:
            if squeeze_ready_geometric:
                squeeze_gate_cleared_t = sim_tm
                dual_geom_ok_for_lift = True

        if squeeze_gate_cleared_t is not None and phase == DualPhase.CLOSE_PROXY_HANDS:
            if dr >= RIGHT_PALM_CONTACT_GOAL_M or dl_tgt >= LEFT_PALM_CONTACT_GOAL_M:
                post_gate_freeze_accum += float(model.opt.timestep)

        assist_prep = bool(
            (not penetration_blocks_assist)
            and max_pen_any_frame <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
            and (
                strict_dual_pose
                or (
                    phase == DualPhase.DUAL_CONTACT_HOLD
                    and rc >= 1
                    and dr < RIGHT_PALM_CONTACT_GOAL_M * 1.15
                )
            )
        )

        assist_should_arm = bool(
            phase in ARM_ASSIST_PHASES
            and (not finger_joints)
            and (not penetration_blocks_assist)
            and assist_prep
        )

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
            full_posture_from_dual(q_work),
            actuator_ids,
            proxy_slide_target=slide_tgt,
        )

        box_c = np.asarray(det_met["box_center"], dtype=float).reshape(3)
        box_max_z = max(box_max_z, float(box_c[2]))

        assist_lift_gate = bool(
            dual_geom_ok_for_lift
            and max_pen_any_frame <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
            and max_pen_any_episode <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
        )

        data.xfrc_applied[:] = 0.0
        assist_active_flag = False
        assist_f_norm = 0.0
        gp_assist = gp_proxy
        if (
            assist_armed
            and (not finger_joints)
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

        mujoco.mj_step(model, data)
        if fix_base_in_world:
            stabilize_floating_base(model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos)
        mujoco.mj_forward(model, data)

        box_max_z = max(
            box_max_z,
            float(
                G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)[
                    "box_center"
                ][2]
            ),
        )

        bh = box_max_z - box_initial_z
        if gp_assist == GraspPhase.LIFT_TEST and assist_active_flag:
            lift_contact_during_lift = True
            if bh >= LIFT_SUCCESS_DELTA_Z_M:
                lift_success = True

        if verbose and not silent and (data.time - last_print >= PRINT_INTERVAL):
            print(
                f"phase={phase.name}  palm_r={palm_r.tolist()}  palm_l={palm_l.tolist()}\n"
                f"          dr={dr:.5f}  dl={dl_tgt:.5f}  rc={rc}  lc={lc}  "
                f"v_right={valid_right_contact}  v_left={valid_left_contact}\n"
                f"          sep_r={sep_r:.5f}  sep_l={sep_l:.5f}  gt_r={geom_touch_r}  gt_l={geom_touch_l}\n"
                f"          pen_any={max_pen_any_frame:.5f}  proxy_tgt={slide_tgt:.4f}  "
                f"assist_active={assist_active_flag}  squeeze_t={squeeze_gate_cleared_t}\n"
                f"          box_dz={bh:.5f}  lift_ok={lift_success}  ik_err~={half_err_l:.5f}",
                flush=True,
            )
            last_print = data.time


    def run_headless() -> None:
        nonlocal mission_complete
        while data.time < float(timeout):
            step_frame()
            if (
                _dual_phase_effective(
                    float(data.time),
                    float(timeout),
                    squeeze_gate_cleared_t,
                    post_gate_virtual_lag_s=post_gate_freeze_accum,
                )
                == DualPhase.DONE
            ):
                mission_complete = True
                return

    def run_viewer() -> None:
        nonlocal mission_complete
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_t0 = time.time()
                step_frame(silent=False)
                if (
                    _dual_phase_effective(
                        float(data.time),
                        float(timeout),
                        squeeze_gate_cleared_t,
                        post_gate_virtual_lag_s=post_gate_freeze_accum,
                    )
                    == DualPhase.DONE
                ):
                    mission_complete = True
                viewer.sync()
                time.sleep(max(0.0, float(model.opt.timestep) - (time.time() - loop_t0)))

    if headless:
        run_headless()
    else:
        run_viewer()

    mujoco.mj_forward(model, data)

    det_fin = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
    box_center_y = float(det_fin["box_center"][1])

    bh_final = box_max_z - box_initial_z
    true_dual_contact_ready_out = bool(
        dual_geom_ok_for_lift
        and np.isfinite(min_touch_r_dist)
        and np.isfinite(min_touch_l_dist)
        and min_touch_r_dist < RIGHT_PALM_CONTACT_GOAL_M
        and min_touch_l_dist <= LEFT_PALM_CONTACT_GOAL_M
        and max_pen_any_episode <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
    )
    lift_raw = lift_success or (
        grasp_assist_used and lift_contact_during_lift and bh_final >= LIFT_SUCCESS_DELTA_Z_M
    )
    lift_final = bool(
        lift_raw
        and true_dual_contact_ready_out
        and bh_final >= LIFT_SUCCESS_DELTA_Z_M
        and max_pen_any_episode <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M
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
    penetration_ok_dual = bool(max_pen_any_episode <= MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M)

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
        "box_height_delta": float(bh_final),
        "lift_success": bool(lift_final),
        "max_penetration_depth_right": float(max_pen_right_palm_episode),
        "max_penetration_depth_left": float(max_pen_left_palm_episode),
        "assist_mode": (
            "xfrc_bounded_stabilizer"
            if not finger_joints
            else "finger_actuators_present_but_unimplemented"
        ),
        "dual_pregrasp_reach_metric_ok": reach_dual_ok,
        "dual_penetration_ok": penetration_ok_dual,
        "motion_complete": bool(mission_complete),
        "phase": final_phase.name,
        "initial_contact_any": bool(initial_contact_any),
        "initial_contact_before_approach": bool(initial_contact_any),
        "max_assist_force": float(max_assist_force),
        "sim_time": float(data.time),
    }

    if verbose:
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

    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="G1 locked-base dual-arm box coordination demo.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--timeout", type=float, default=18.0)
    ap.add_argument("--pelvis-z", type=float, default=DEFAULT_PELVIS_Z)
    ap.add_argument("--posture-gain", type=float, default=IK_POSTURE_GAIN)
    ap.add_argument(
        "--max-joint-from-neutral",
        type=float,
        default=IK_MAX_ABS_JOINT_FROM_NEUTRAL + 0.40,
    )
    ap.add_argument("--dbg-markers", action="store_true")
    ns = ap.parse_args(argv)

    try:
        run_g1_dual_arm_box(
            headless=ns.headless,
            timeout=ns.timeout,
            initial_pelvis_z=ns.pelvis_z,
            verbose=True,
            posture_gain=ns.posture_gain,
            max_joint_from_neutral=ns.max_joint_from_neutral,
            dbg_markers=ns.dbg_markers,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
