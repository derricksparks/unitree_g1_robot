#!/usr/bin/env python3
"""
Locked-base G1 **grasp-box prototype** (IK + contact + bounded assist + small lift test).

Pre-grasp and approach targets come from :class:`~perception.g1_box_perception.G1BoxPerception`
(simulated perception from ``box_geom``).

**Vision/perception selects the grasp point, but there is still no Unitree articulated hand.**

The MJCF adds **animated translucent slide “finger” proxies** (``*_proxy_*_finger`` with
``right_proxy_*_finger_slide`` joints and ``*_finger_motor`` actuators). These are purely a **simulated
interpretation aid**, not hardware fingers. Closing/opening follows the grasp script phases while
lift remains **bounded** :obj:`xfrc_applied` stabilization after contact—not a kinematic teleport.

Assist prefers **palm plus at least one finger-proxy** box contact when those actuators exist; if the
fingers never touch the box, a **palm-only fallback** arms after ``PRE_GRASP_HOLD`` finishes.

**Scene / dynamics:** Default box pose for **single‑arm** scripts is lateral offset MJCF:
``g1_reach_box_scene_single_arm_offset.xml`` (neutral STOW/APPROACH clear the torso—see comment there).
Dual‑arm demos load the **torso‑centered** ``g1_reach_box_scene.xml`` instead.

At load time a
**vertical slide joint with a stiff reference spring** is injected on ``reach_target_box`` so the box
can translate in :math:`+z` for a lift metric without editing the shared MJCF on disk. (A fully
free floating box would collapse under gravity during the long approach; the slide keeps the task
focused on locked-base vertical co-lift.)

Run::

    ./.venv/bin/python simulation/mujoco/g1/run_g1_grasp_box.py
    ./.venv/bin/python simulation/mujoco/g1/run_g1_grasp_box.py --headless --timeout 10

"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import re
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
    MAX_PENETRATION_DEPTH_OK_M,
    PENETRATION_FACE_TOLERANCE,
    TOUCH_DISTANCE_M,
    palm_in_side_contact_corridor,
    palm_plate_outer_face_x_nominal,
    penetration_depth_m,
)
from run_g1_posture_hold import (  # noqa: E402
    NEUTRAL_POSTURE,
    DEFAULT_PELVIS_Z,
    PRINT_INTERVAL,
    SUFFIX_POS,
    apply_neutral_pose,
    build_actuator_id_map,
    floating_base_address_map,
    joint_name_from_actuator,
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

# Centered box (dual-arm IK milestone layout) — **Dex3-1 hands** pipeline scene.
REACH_BOX_DUAL_MILESTONE_SCENE_PATH = _G1 / "g1_reach_box_scene_dex3.xml"
# Authoritative Dex3 MJCF (forked from ``assets/unitree_g1/``; do not edit upstream).
PIPELINE_G1_DEX3_HANDS_XML = _G1 / "assets" / "g1_dex3_hands_actuated.xml"
G1_DUAL_ARM_NU_DEX3 = 43
# Lateral offset clears neutral-pose self overlap for grasp/touch/reach demos.
REACH_BOX_SCENE_SINGLE_ARM_OFFSET_PATH = _G1 / "g1_reach_box_scene_single_arm_offset.xml"
SCENE_PATH = REACH_BOX_SCENE_SINGLE_ARM_OFFSET_PATH
BASE_G1_XML = _G1 / "assets" / "g1_position_actuated.xml"

PALM_SITE_NAME = "right_palm_site"
FALLBACK_TARGET_BODY_NAME = "right_wrist_yaw_link"
BOX_BODY_NAME = "reach_target_box"
BOX_GEOM_NAME = "box_geom"
LEGACY_BOX_GEOM_NAME = "reach_box_geom"

PALM_CONTACT_GEOM_NAME = "right_palm_contact_geom"
PROXY_LEFT_FINGER_GEOM_NAME = "right_proxy_left_finger_geom"
PROXY_RIGHT_FINGER_GEOM_NAME = "right_proxy_right_finger_geom"

GRASP_IK_KP_SCALE = 0.86

PREGRASP_METRIC_PHASES = frozenset({"DESCEND_TO_PREGRASP", "PRE_GRASP_HOLD"})
_EARLY_PHASES = frozenset({"STOW", "APPROACH_ABOVE_BOX"})
PRE_HOLD_ASSIST_GATE_PHASES = frozenset(
    {"STOW", "APPROACH_ABOVE_BOX", "DESCEND_TO_PREGRASP"}
)

ASSIST_ARM_PHASES = frozenset({"PRE_GRASP_HOLD", "CLOSE_HAND"})
ASSIST_ACTIVE_PHASES = frozenset({"GRASP_HOLD", "LIFT_TEST", "LOWER_BACK"})

RIGHT_TOUCH_GEOM_NAMES = (
    "right_hand_collision",
    "right_wrist_collision",
)

BOX_SLIDE_JOINT_NAME = "box_lift_slide"
LIFT_ASSIST_TARGET_DZ_M = 0.175

LIFT_PALM_DELTA_Z_M = 0.08
LIFT_SUCCESS_DELTA_Z_M = 0.04

# Slightly looser than palm-only touch: dynamic box + slide joint can deepen contact briefly.
_GRASP_PREGRASP_PENETRATION_OK_M = MAX_PENETRATION_DEPTH_OK_M + 0.007

ASSIST_KP_POS = 420.0
ASSIST_KD_Z = 85.0
ASSIST_LIFT_Z_SCALE = 8.0
MAX_ASSIST_FORCE_NEWTONS = 720.0

PROXY_FINGER_OPEN = 0.0
PROXY_FINGER_CLOSED = 0.012
PROXY_CLOSE_TIME = 0.8
PROXY_OPEN_TIME = 0.35
PROXY_LEFT_MOTOR = "right_proxy_left_finger_motor"
PROXY_RIGHT_MOTOR = "right_proxy_right_finger_motor"
PROXY_LEFT_SLIDE_JOINT = "right_proxy_left_finger_slide"
PROXY_RIGHT_SLIDE_JOINT = "right_proxy_right_finger_slide"
G1_GRASP_SCENE_NU_EXPECTED = 31

_SLIDE_JOINT_FRAGMENT = (
    '\n      <joint name="box_lift_slide" type="slide" axis="0 0 1" '
    'limited="true" range="-0.14 0.38" stiffness="2100" damping="135" armature="0.02"/>\n'
)


class GraspPhase(Enum):
    STOW = auto()
    APPROACH_ABOVE_BOX = auto()
    DESCEND_TO_PREGRASP = auto()
    PRE_GRASP_HOLD = auto()
    CLOSE_HAND = auto()
    GRASP_HOLD = auto()
    LIFT_TEST = auto()
    LOWER_BACK = auto()
    OPEN_HAND = auto()
    RETREAT = auto()
    DONE = auto()


def _proxy_finger_slide_target(
    phase: GraspPhase,
    sim_t: float,
    duration: float,
    *,
    close_t0: float | None,
    open_t0: float | None,
) -> float:
    if phase in (
        GraspPhase.STOW,
        GraspPhase.APPROACH_ABOVE_BOX,
        GraspPhase.DESCEND_TO_PREGRASP,
        GraspPhase.PRE_GRASP_HOLD,
    ):
        return float(PROXY_FINGER_OPEN)
    if phase == GraspPhase.CLOSE_HAND:
        if close_t0 is None:
            return float(PROXY_FINGER_OPEN)
        u = (float(sim_t) - close_t0) / max(float(PROXY_CLOSE_TIME), 1e-9)
        return float(PROXY_FINGER_CLOSED) * float(min(1.0, max(0.0, u)))
    if phase in (GraspPhase.GRASP_HOLD, GraspPhase.LIFT_TEST, GraspPhase.LOWER_BACK):
        return float(PROXY_FINGER_CLOSED)
    # OPEN_HAND, RETREAT, DONE
    if open_t0 is None:
        return float(PROXY_FINGER_CLOSED)
    u_open = (float(sim_t) - open_t0) / max(float(PROXY_OPEN_TIME), 1e-9)
    return float(PROXY_FINGER_CLOSED) * float(max(0.0, 1.0 - min(1.0, u_open)))


def _command_grasp_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    full_posture: dict[str, float],
    actuator_ids: dict[str, int],
    *,
    proxy_slide_target: float,
    use_legacy_proxy_finger_motors: bool = True,
) -> None:
    for aname, aid in actuator_ids.items():
        an = str(aname)
        if use_legacy_proxy_finger_motors and an in (PROXY_LEFT_MOTOR, PROXY_RIGHT_MOTOR):
            data.ctrl[aid] = float(proxy_slide_target)
            continue
        if an.endswith(SUFFIX_POS):
            jn = joint_name_from_actuator(an)
            data.ctrl[aid] = float(full_posture[jn])
            continue
        # Unitree Dex3 MJCF: ``<position name=\"joint_name\" joint=\"joint_name\"/>`` (no ``_pos_actuator`` suffix).
        if an in full_posture:
            data.ctrl[aid] = float(full_posture[an])
            continue
        data.ctrl[aid] = 0.0

def _has_actuated_finger_joints(model: mujoco.MjModel) -> bool:
    """True if any actuator drives a joint whose name looks like an articulated finger / gripper."""
    fingerish = re.compile(
        r"(finger|thumb|index|middle|ring|pinky|gripper|hand_(open|close)|_hand_)",
        re.IGNORECASE,
    )
    for aid in range(model.nu):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if jn is None:
            continue
        trnid = model.actuator_trnid[aid, 0]
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if trnid < 0:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(trnid))
        if nm and nm.startswith("right_proxy_"):
            continue
        if nm and fingerish.search(nm):
            return True
    return False


def _inject_box_vertical_slide_for_grasp(scene_xml_text: str) -> str:
    key = '<body name="reach_target_box"'
    if key not in scene_xml_text:
        raise ValueError(f"Missing {key!r} in scene XML text")
    if BOX_SLIDE_JOINT_NAME in scene_xml_text:
        return scene_xml_text
    i = scene_xml_text.index(key)
    j = scene_xml_text.index(">", i) + 1
    return scene_xml_text[:j] + _SLIDE_JOINT_FRAGMENT + scene_xml_text[j:]


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_grasp_scene_model() -> mujoco.MjModel:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing scene file: {SCENE_PATH}")
    xml_text = _inject_box_vertical_slide_for_grasp(SCENE_PATH.read_text())
    scene_root = SCENE_PATH.resolve().parent
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


def _geom_names(model: mujoco.MjModel, gids: set[int]) -> list[str]:
    names: list[str] = []
    for gid in sorted(gids):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        names.append(nm if nm else f"geom[{gid}]")
    return names


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


def _box_contact_split_counts(
    data: mujoco.MjData,
    box_gid: int,
    *,
    palm_geom_id: int,
    left_finger_geom_id: int,
    right_finger_geom_id: int,
) -> tuple[int, int, int]:
    """Separate box↔geom contact counts for palm plate vs assisted finger proxies."""
    palm_c = left_c = right_c = 0
    for cid in range(data.ncon):
        contact = data.contact[cid]
        if box_gid not in (contact.geom1, contact.geom2):
            continue
        other = int(contact.geom2 if contact.geom1 == box_gid else contact.geom1)
        if palm_geom_id >= 0 and other == palm_geom_id:
            palm_c += 1
            continue
        if left_finger_geom_id >= 0 and other == left_finger_geom_id:
            left_c += 1
            continue
        if right_finger_geom_id >= 0 and other == right_finger_geom_id:
            right_c += 1
            continue
    return palm_c, left_c, right_c


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
        gids: set[int] = {int(palm_gid)}
        lg = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, PROXY_LEFT_FINGER_GEOM_NAME
        )
        rg = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, PROXY_RIGHT_FINGER_GEOM_NAME
        )
        if lg >= 0:
            gids.add(int(lg))
        if rg >= 0:
            gids.add(int(rg))
        if len(gids) >= 3:
            mode = "palm_contact_and_proxy_fingers"
        elif len(gids) >= 2:
            mode = "palm_contact_partial_proxy"
        else:
            mode = "palm_contact_geom"
        return gids, mode
    return _hand_fallback_touch_gids(model), "hand_geoms_fallback"


def _phase_boundaries(duration: float) -> dict[str, float]:
    d = max(float(duration), 1e-6)
    return {
        "t_stow_end": 0.10 * d,
        "t_approach_end": 0.29 * d,
        "t_descend_end": 0.46 * d,
        "t_prehold_end": 0.60 * d,
        "t_close_hand_end": 0.628 * d,
        "t_grasp_hold_end": 0.63 * d,
        "t_lift_end": 0.86 * d,
        "t_lower_end": 0.92 * d,
        "t_open_end": 0.94 * d,
        "t_retreat_end": 0.979 * d,
    }


def _phase_at(sim_t: float, duration: float) -> GraspPhase:
    b = _phase_boundaries(duration)
    if sim_t < b["t_stow_end"]:
        return GraspPhase.STOW
    if sim_t < b["t_approach_end"]:
        return GraspPhase.APPROACH_ABOVE_BOX
    if sim_t < b["t_descend_end"]:
        return GraspPhase.DESCEND_TO_PREGRASP
    if sim_t < b["t_prehold_end"]:
        return GraspPhase.PRE_GRASP_HOLD
    if sim_t < b["t_close_hand_end"]:
        return GraspPhase.CLOSE_HAND
    if sim_t < b["t_grasp_hold_end"]:
        return GraspPhase.GRASP_HOLD
    if sim_t < b["t_lift_end"]:
        return GraspPhase.LIFT_TEST
    if sim_t < b["t_lower_end"]:
        return GraspPhase.LOWER_BACK
    if sim_t < b["t_open_end"]:
        return GraspPhase.OPEN_HAND
    if sim_t < b["t_retreat_end"]:
        return GraspPhase.RETREAT
    return GraspPhase.DONE


def _palm_ik_goal(
    phase: GraspPhase,
    *,
    p_stow: np.ndarray,
    p_above: np.ndarray,
    p_touch: np.ndarray,
    sim_t: float,
    duration: float,
) -> np.ndarray:
    b = _phase_boundaries(duration)

    if phase == GraspPhase.STOW:
        return p_stow.copy()
    if phase == GraspPhase.APPROACH_ABOVE_BOX:
        u = _smoothstep01(
            (sim_t - b["t_stow_end"])
            / max(b["t_approach_end"] - b["t_stow_end"], 1e-9)
        )
        return (1.0 - u) * p_stow + u * p_above
    if phase == GraspPhase.DESCEND_TO_PREGRASP:
        u = _smoothstep01(
            (sim_t - b["t_approach_end"])
            / max(b["t_descend_end"] - b["t_approach_end"], 1e-9)
        )
        return (1.0 - u) * p_above + u * p_touch
    if phase in (
        GraspPhase.PRE_GRASP_HOLD,
        GraspPhase.CLOSE_HAND,
        GraspPhase.GRASP_HOLD,
    ):
        return p_touch.copy()
    if phase == GraspPhase.LIFT_TEST:
        u = _smoothstep01(
            (sim_t - b["t_grasp_hold_end"])
            / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
        )
        return p_touch + np.array([0.0, 0.0, LIFT_PALM_DELTA_Z_M * u], dtype=float)
    if phase == GraspPhase.LOWER_BACK:
        u_lo = _smoothstep01(
            (sim_t - b["t_lift_end"])
            / max(b["t_lower_end"] - b["t_lift_end"], 1e-9)
        )
        u_hi = _smoothstep01(
            (sim_t - b["t_grasp_hold_end"])
            / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
        )
        dz_end = LIFT_PALM_DELTA_Z_M * u_hi
        dz = dz_end * (1.0 - u_lo)
        return p_touch + np.array([0.0, 0.0, dz], dtype=float)
    if phase == GraspPhase.OPEN_HAND:
        return p_touch.copy()
    if phase == GraspPhase.RETREAT:
        u = _smoothstep01(
            (sim_t - b["t_open_end"])
            / max(b["t_retreat_end"] - b["t_open_end"], 1e-9)
        )
        return (1.0 - u) * p_touch + u * p_above
    return p_stow.copy()


def _assist_desired_xyz(
    phase: GraspPhase,
    *,
    anchor: np.ndarray,
    sim_t: float,
    duration: float,
) -> np.ndarray:
    b = _phase_boundaries(duration)
    xy = anchor[:2]
    az = float(anchor[2])
    dz_max = float(LIFT_ASSIST_TARGET_DZ_M)
    if phase == GraspPhase.GRASP_HOLD:
        return np.array([xy[0], xy[1], az], dtype=float)
    if phase == GraspPhase.LIFT_TEST:
        u = _smoothstep01(
            (sim_t - b["t_grasp_hold_end"])
            / max(b["t_lift_end"] - b["t_grasp_hold_end"], 1e-9)
        )
        return np.array([xy[0], xy[1], az + dz_max * u], dtype=float)
    if phase == GraspPhase.LOWER_BACK:
        u_lo = _smoothstep01(
            (sim_t - b["t_lift_end"])
            / max(b["t_lower_end"] - b["t_lift_end"], 1e-9)
        )
        z_top = az + dz_max
        return np.array([xy[0], xy[1], (1.0 - u_lo) * z_top + u_lo * az], dtype=float)
    return anchor.copy()


def run_g1_grasp_box(
    *,
    headless: bool = False,
    timeout: float = 10.0,
    initial_pelvis_z: float = DEFAULT_PELVIS_Z,
    verbose: bool = True,
    fix_base_in_world: bool = True,
    posture_gain: float = IK_POSTURE_GAIN,
    max_joint_from_neutral: float = IK_MAX_ABS_JOINT_FROM_NEUTRAL,
    dbg_markers: bool = False,
) -> dict[str, Any]:
    if not BASE_G1_XML.is_file():
        raise FileNotFoundError(f"Missing packaged model: {BASE_G1_XML}")

    model = _load_grasp_scene_model()
    if model.nu != G1_GRASP_SCENE_NU_EXPECTED:
        raise RuntimeError(
            f"Expected nu={G1_GRASP_SCENE_NU_EXPECTED}, got {model.nu} "
            "(29 hinge _pos_actuators + 2 proxy finger motors)"
        )

    hinge_names = set()
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            hinge_names.add(name)
    if set(NEUTRAL_POSTURE.keys()) != hinge_names:
        raise ValueError("NEUTRAL_POSTURE must list all hinges")

    finger_joints = _has_actuated_finger_joints(model)
    _apply_kp_scale_for_joint_subset(model, set(IK_JOINT_NAMES), GRASP_IK_KP_SCALE)
    _apply_kp_scale_for_joint_subset(
        model, {"right_wrist_roll_joint", "right_wrist_yaw_joint"}, GRASP_IK_KP_SCALE
    )

    data = mujoco.MjData(model)
    fd = mujoco.MjData(model)
    actuator_ids = build_actuator_id_map(model)
    proxy_finger_actuators_present = bool(
        PROXY_LEFT_MOTOR in actuator_ids and PROXY_RIGHT_MOTOR in actuator_ids
    )
    left_slide_jid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROXY_LEFT_SLIDE_JOINT)
    )
    right_slide_jid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PROXY_RIGHT_SLIDE_JOINT)
    )
    left_proxy_qpos_adr = (
        int(model.jnt_qposadr[left_slide_jid]) if left_slide_jid >= 0 else -1
    )
    right_proxy_qpos_adr = (
        int(model.jnt_qposadr[right_slide_jid]) if right_slide_jid >= 0 else -1
    )

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
    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME)
    if box_bid < 0:
        raise RuntimeError(f"Body {BOX_BODY_NAME} not found")

    palm_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PALM_SITE_NAME))
    box_gid = _box_geom_id(model)
    touch_geom_set, contact_detection_mode = _touch_contact_gids(model)
    palm_contact_gid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PALM_CONTACT_GEOM_NAME)
    )
    left_finger_gid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROXY_LEFT_FINGER_GEOM_NAME)
    )
    right_finger_gid = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PROXY_RIGHT_FINGER_GEOM_NAME)
    )
    used_contact_detection = bool(touch_geom_set) and box_gid >= 0

    box_geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, box_gid) or BOX_GEOM_NAME
    touch_geom_names = _geom_names(model, touch_geom_set)

    slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BOX_SLIDE_JOINT_NAME)
    slide_dofadr = (
        int(model.jnt_dofadr[slide_jid]) if slide_jid >= 0 else -1
    )

    if verbose:
        print(
            "\nNOTE: Assisted locked-base grasp prototype.\n"
            "Blue ``*_proxy_*_finger`` sliders are **simulated** gripper visuals only—not Unitree "
            "hand hardware. They close during ``CLOSE_HAND`` and open during ``OPEN_HAND``.\n"
            "Lift still uses bounded :obj:`xfrc_applied` after contact; assist prefers palm + "
            "finger-proxy box contact (palm-only fallback may apply after ``PRE_GRASP_HOLD``).\n"
            f"proxy_finger_actuators_present={proxy_finger_actuators_present}  "
            f"finger_actuators_present={finger_joints}  slide_joint="
            f"{BOX_SLIDE_JOINT_NAME if slide_jid >= 0 else 'MISSING'}\n"
            f"Contact detection geoms: box={box_geom_name!r}; touch={touch_geom_names}; "
            f"mode={contact_detection_mode!r}\n"
        )

    box0 = _box_center(model, data)
    det0 = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
    box_initial_z = float(box0[2])
    box_max_z = box_initial_z
    p_stow = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)

    if verbose:
        print(
            "G1BoxPerception (startup):\n"
            f"  box_center={det0['box_center'].tolist()}\n"
            f"  box_half_size={det0['box_half_size'].tolist()}\n"
            f"  near_face_x={det0['near_face_x']:.5f}\n"
            f"  bottom_z={det0['bottom_z']:.5f}\n"
            f"  grasp_height_z={det0['grasp_height_z']:.5f}\n"
            f"  palm_pregrasp_target={det0['palm_pregrasp_target'].tolist()}\n"
            f"  palm_approach_target={det0['palm_approach_target'].tolist()}"
        )

    initial_contact_n = _contact_count(data, box_gid, touch_geom_set)
    initial_contact_before_approach = initial_contact_n > 0
    stow_contact_warned = initial_contact_before_approach

    min_palm_touch_target_distance = float("inf")
    max_contact_count_overall = 0
    palm_contact_count_max = 0
    left_finger_contact_count_max = 0
    right_finger_contact_count_max = 0
    max_penetration_depth_pre = 0.0
    early_phase_mission_violation = False
    penetration_during_pregrasp_hold = False
    last_print = -PRINT_INTERVAL
    mission_complete = False
    final_phase = GraspPhase.STOW

    assist_armed = False
    assist_anchor = np.zeros(3, dtype=float)
    max_assist_force = 0.0
    grasp_assist_used = False
    lift_contact_during_lift = False
    assist_active_flag = False
    lift_success = False
    penetration_blocks_assist = False

    finger_close_t0: float | None = None
    finger_open_t0: float | None = None
    last_grasp_phase_e = GraspPhase.STOW
    proxy_finger_contact_ready_dual = False
    fallback_palm_only_assist_used_flag = False
    proxy_finger_closed_metric_flag = False
    proxy_finger_target_max = 0.0

    def full_posture_from_ik(q_work: np.ndarray) -> dict[str, float]:
        full = dict(NEUTRAL_POSTURE)
        for jn, qv in zip(IK_JOINT_NAMES, q_work, strict=True):
            full[jn] = float(qv)
        return full

    def step_frame(*, force_report: bool = False, silent: bool = False) -> None:
        nonlocal min_palm_touch_target_distance, max_contact_count_overall
        nonlocal palm_contact_count_max, left_finger_contact_count_max
        nonlocal right_finger_contact_count_max
        nonlocal max_penetration_depth_pre, early_phase_mission_violation
        nonlocal penetration_during_pregrasp_hold, last_print, final_phase
        nonlocal stow_contact_warned, mission_complete
        nonlocal assist_armed, assist_anchor, max_assist_force, grasp_assist_used
        nonlocal lift_contact_during_lift, assist_active_flag, lift_success
        nonlocal box_max_z, penetration_blocks_assist
        nonlocal finger_close_t0, finger_open_t0, last_grasp_phase_e
        nonlocal proxy_finger_contact_ready_dual, fallback_palm_only_assist_used_flag
        nonlocal proxy_finger_closed_metric_flag, proxy_finger_target_max

        phase = _phase_at(data.time, timeout)
        final_phase = phase

        if phase != last_grasp_phase_e:
            if phase == GraspPhase.CLOSE_HAND:
                finger_close_t0 = float(data.time)
            if phase == GraspPhase.OPEN_HAND:
                finger_open_t0 = float(data.time)
            last_grasp_phase_e = phase

        proxy_finger_target = _proxy_finger_slide_target(
            phase,
            float(data.time),
            float(timeout),
            close_t0=finger_close_t0,
            open_t0=finger_open_t0,
        )
        proxy_finger_target_max = max(proxy_finger_target_max, proxy_finger_target)
        box_c = _box_center(model, data)
        box_z = float(box_c[2])
        box_max_z = max(box_max_z, box_z)

        det_goal = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
        p_touch_now = np.asarray(det_goal["palm_pregrasp_target"], dtype=float)
        p_above_now = np.asarray(det_goal["palm_approach_target"], dtype=float)
        palm_goal = _palm_ik_goal(
            phase,
            p_stow=p_stow,
            p_above=p_above_now,
            p_touch=p_touch_now,
            sim_t=float(data.time),
            duration=float(timeout),
        )

        q_work, _ = solve_ik_q(
            model,
            fd,
            data,
            ik_qpos_adrs,
            q_low,
            q_high,
            wrist_bid,
            palm_goal,
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

        det_met = G1BoxPerception.detect_box(model, data, box_geom_name=box_geom_name)
        palm = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)
        touch_tgt = np.asarray(det_met["palm_pregrasp_target"], dtype=float)
        touch_target_dist = float(np.linalg.norm(palm - touch_tgt))
        phase_name = phase.name
        near_face_x = float(det_met["near_face_x"])
        ly = float(det_met["left_edge_y"])
        ry = float(det_met["right_edge_y"])
        bz = float(det_met["bottom_z"])
        tz = float(det_met["top_z"])
        if palm_contact_gid >= 0:
            depth_pre = penetration_depth_m(
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
            od = palm_plate_outer_face_x_nominal(float(palm[0]))
            palm_in_corr = palm_in_side_contact_corridor(
                palm,
                left_edge_y=ly,
                right_edge_y=ry,
                bottom_z=bz,
                top_z=tz,
            )
            depth_pre = (
                max(0.0, float(od) - float(near_face_x)) if palm_in_corr else 0.0
            )
        palm_c, lf_c, rf_c = _box_contact_split_counts(
            data,
            box_gid,
            palm_geom_id=palm_contact_gid,
            left_finger_geom_id=left_finger_gid,
            right_finger_geom_id=right_finger_gid,
        )
        grasp_contact_ready = bool(palm_c > 0 or lf_c > 0 or rf_c > 0)
        contact_count = palm_c + lf_c + rf_c
        palm_contact_count_max = max(palm_contact_count_max, palm_c)
        left_finger_contact_count_max = max(left_finger_contact_count_max, lf_c)
        right_finger_contact_count_max = max(right_finger_contact_count_max, rf_c)

        if (
            phase_name in PRE_HOLD_ASSIST_GATE_PHASES
            and palm_contact_gid >= 0
            and depth_pre > MAX_PENETRATION_DEPTH_OK_M
        ):
            penetration_blocks_assist = True

        if phase_name in _EARLY_PHASES and contact_count > 0:
            early_phase_mission_violation = True

        if phase_name == "STOW" and contact_count > 0 and verbose and not stow_contact_warned:
            print(
                "WARNING: contact during STOW "
                f"(contacts={contact_count}, mode={contact_detection_mode})"
            )
            stow_contact_warned = True

        if phase_name in PREGRASP_METRIC_PHASES:
            min_palm_touch_target_distance = min(
                min_palm_touch_target_distance, touch_target_dist
            )
            max_contact_count_overall = max(max_contact_count_overall, contact_count)
            max_penetration_depth_pre = max(max_penetration_depth_pre, depth_pre)
            if phase_name == "PRE_GRASP_HOLD" and depth_pre > _GRASP_PREGRASP_PENETRATION_OK_M:
                penetration_during_pregrasp_hold = True

        palm_ok_contact = palm_c > 0
        finger_side_ok_contact = lf_c > 0 or rf_c > 0
        if palm_ok_contact and finger_side_ok_contact:
            proxy_finger_contact_ready_dual = True

        b_gate_pre = _phase_boundaries(float(timeout))
        after_prehold = float(data.time) >= float(b_gate_pre["t_prehold_end"]) - 1e-9

        should_arm_assist = False
        if (
            phase_name in ASSIST_ARM_PHASES
            and not finger_joints
            and (not penetration_blocks_assist)
        ):
            if proxy_finger_actuators_present:
                if palm_ok_contact and finger_side_ok_contact:
                    should_arm_assist = True
                elif after_prehold and palm_ok_contact:
                    should_arm_assist = True
                    fallback_palm_only_assist_used_flag = True
            else:
                should_arm_assist = bool(grasp_contact_ready)

        if should_arm_assist:
            if not assist_armed:
                assist_anchor = np.asarray(box_c, dtype=float).copy()
            assist_armed = True
            grasp_assist_used = True

        _command_grasp_actuators(
            model,
            data,
            full_posture_from_ik(q_work),
            actuator_ids,
            proxy_slide_target=proxy_finger_target,
            use_legacy_proxy_finger_motors=True,
        )

        data.xfrc_applied[:] = 0.0
        assist_active_flag = False
        assist_f_norm = 0.0
        if (
            assist_armed
            and (not finger_joints)
            and phase_name in ASSIST_ACTIVE_PHASES
        ):
            des = _assist_desired_xyz(
                phase,
                anchor=assist_anchor,
                sim_t=float(data.time),
                duration=float(timeout),
            )
            err = des - np.asarray(det_met["box_center"], dtype=float)
            if slide_dofadr >= 0:
                vz = float(data.qvel[slide_dofadr])
            else:
                vz = 0.0
            if phase == GraspPhase.LIFT_TEST:
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
            stabilize_floating_base(
                model, data, base_map=base_map, nominal_qpos7=nominal_base_qpos
            )
        mujoco.mj_forward(model, data)

        box_c2 = _box_center(model, data)
        box_z2 = float(box_c2[2])
        box_max_z = max(box_max_z, box_z2)
        palm2 = _point_world(model, data, site_id=palm_site_id, fallback_body_id=wrist_bid)
        palm_c2, lf_c2, rf_c2 = _box_contact_split_counts(
            data,
            box_gid,
            palm_geom_id=palm_contact_gid,
            left_finger_geom_id=left_finger_gid,
            right_finger_geom_id=right_finger_gid,
        )
        palm_contact_count_max = max(palm_contact_count_max, palm_c2)
        left_finger_contact_count_max = max(left_finger_contact_count_max, lf_c2)
        right_finger_contact_count_max = max(right_finger_contact_count_max, rf_c2)
        contact_count2 = palm_c2 + lf_c2 + rf_c2
        max_contact_count_overall = max(max_contact_count_overall, contact_count2)
        grasp_contact_ready_post = bool(palm_c2 > 0 or lf_c2 > 0 or rf_c2 > 0)

        proxy_lq = 0.0
        proxy_rq = 0.0
        if left_proxy_qpos_adr >= 0:
            proxy_lq = float(data.qpos[left_proxy_qpos_adr])
        if right_proxy_qpos_adr >= 0:
            proxy_rq = float(data.qpos[right_proxy_qpos_adr])
        q_close_thr = PROXY_FINGER_CLOSED * 0.92
        if palm_c2 > 0 and (lf_c2 > 0 or rf_c2 > 0):
            proxy_finger_contact_ready_dual = True
        if proxy_lq >= q_close_thr and proxy_rq >= q_close_thr:
            proxy_finger_closed_metric_flag = True

        box_height_delta = box_max_z - box_initial_z
        if phase_name == "LIFT_TEST" and assist_active_flag:
            lift_contact_during_lift = True
        if (
            phase_name == "LIFT_TEST"
            and assist_active_flag
            and box_height_delta >= LIFT_SUCCESS_DELTA_Z_M
        ):
            lift_success = True

        if verbose and not silent and (
            force_report or data.time - last_print >= PRINT_INTERVAL
        ):
            print(
                f"phase={phase_name}  palm_xyz={palm2.tolist()}  box_xyz={box_c2.tolist()}  "
                f"proxy_finger_target={proxy_finger_target:.5f}  proxy_left_qpos={proxy_lq:.5f}  "
                f"proxy_right_qpos={proxy_rq:.5f}  "
                f"palm_contact_count={palm_c2}  left_finger_contact_count={lf_c2}  "
                f"right_finger_contact_count={rf_c2}  "
                f"fallback_palm_only_assist_used={fallback_palm_only_assist_used_flag}  "
                f"grasp_contact_ready={grasp_contact_ready_post}  "
                f"grasp_assist_active={assist_active_flag}  "
                f"assist_force_norm={assist_f_norm:.4f}  box_height_delta={box_height_delta:.5f}  "
                f"lift_success={lift_success}"
            )
            last_print = data.time

    def run_headless_loop() -> None:
        nonlocal mission_complete
        while data.time < timeout:
            step_frame()
            if _phase_at(data.time, timeout) == GraspPhase.DONE:
                mission_complete = True
                return

    def run_viewer_loop() -> None:
        nonlocal mission_complete
        viewer_mod = importlib.import_module("mujoco.viewer")
        with viewer_mod.launch_passive(model, data) as viewer:
            while viewer.is_running():
                loop_t0 = time.time()
                step_frame(silent=mission_complete)
                if _phase_at(data.time, timeout) == GraspPhase.DONE:
                    mission_complete = True
                viewer.sync()
                dt = model.opt.timestep
                time.sleep(max(0.0, dt - (time.time() - loop_t0)))

    if headless:
        run_headless_loop()
    else:
        run_viewer_loop()

    mujoco.mj_forward(model, data)
    box_final = _box_center(model, data)

    reach_ok_pre = np.isfinite(min_palm_touch_target_distance) and (
        min_palm_touch_target_distance < TOUCH_DISTANCE_M or max_contact_count_overall > 0
    )
    penetration_ok = max_penetration_depth_pre <= _GRASP_PREGRASP_PENETRATION_OK_M
    pregrasp_success = bool(
        (not initial_contact_before_approach)
        and reach_ok_pre
        and penetration_ok
        and (not early_phase_mission_violation)
        and (not penetration_during_pregrasp_hold)
        and assist_armed
    )

    disp = np.asarray(box_final - box0, dtype=float).reshape(3,)
    box_height_delta = box_max_z - box_initial_z
    lift_success_final = (
        lift_success
        or (
            grasp_assist_used
            and lift_contact_during_lift
            and box_height_delta >= LIFT_SUCCESS_DELTA_Z_M
        )
    )

    grasp_contact_ready_final = bool(
        palm_contact_count_max > 0
        or left_finger_contact_count_max > 0
        or right_finger_contact_count_max > 0
    )

    proxy_finger_contact_ready_dual_final = bool(proxy_finger_contact_ready_dual)

    proxy_finger_closed_final = bool(
        proxy_finger_actuators_present
        and (
            proxy_finger_closed_metric_flag
            or proxy_finger_target_max >= PROXY_FINGER_CLOSED * 0.98
        )
    )
    fallback_palm_only_out = bool(
        proxy_finger_actuators_present and fallback_palm_only_assist_used_flag
    )

    out: dict[str, Any] = {
        "success": bool(pregrasp_success and lift_success_final and grasp_assist_used),
        "initial_contact_before_approach": initial_contact_before_approach,
        "pregrasp_success": pregrasp_success,
        "early_phase_mission_violation": early_phase_mission_violation,
        "penetration_during_pregrasp_hold": penetration_during_pregrasp_hold,
        "min_palm_touch_target_distance": float(min_palm_touch_target_distance),
        "pregrasp_assist_armed": bool(assist_armed),
        "grasp_assist_used": bool(grasp_assist_used),
        "finger_actuators_present": finger_joints,
        "assist_mode": (
            "xfrc_bounded_stabilizer"
            if not finger_joints
            else "finger_actuators_present_but_unimplemented"
        ),
        "max_contact_count": int(max_contact_count_overall),
        "palm_contact_count_max": int(palm_contact_count_max),
        "left_finger_contact_count_max": int(left_finger_contact_count_max),
        "right_finger_contact_count_max": int(right_finger_contact_count_max),
        "grasp_contact_ready": bool(grasp_contact_ready_final),
        "proxy_finger_actuators_present": bool(proxy_finger_actuators_present),
        "proxy_finger_closed": bool(proxy_finger_closed_final),
        "proxy_finger_contact_ready": bool(proxy_finger_contact_ready_dual_final),
        "fallback_palm_only_assist_used": bool(fallback_palm_only_out),
        "max_penetration_depth": float(max_penetration_depth_pre),
        "max_penetration_depth_ok_m": _GRASP_PREGRASP_PENETRATION_OK_M,
        "touch_baseline_penetration_ok_m": MAX_PENETRATION_DEPTH_OK_M,
        "touch_distance_threshold_m": TOUCH_DISTANCE_M,
        "box_initial_z": float(box_initial_z),
        "box_max_z": float(box_max_z),
        "box_height_delta": float(box_height_delta),
        "lift_success": bool(lift_success_final),
        "max_assist_force": float(max_assist_force),
        "final_box_displacement": float(np.linalg.norm(disp)),
        "contact_detection_mode": contact_detection_mode,
        "penetration_face_tolerance": PENETRATION_FACE_TOLERANCE,
        "model_path": str(SCENE_PATH.resolve()),
        "motion_complete": mission_complete,
        "phase": final_phase.name,
        "sim_time": float(data.time),
        "used_contact_detection": used_contact_detection,
    }

    if verbose:
        print(
            "----- grasp_box summary -----\n"
            f"initial_contact_before_approach: {out['initial_contact_before_approach']}\n"
            f"pregrasp_success: {out['pregrasp_success']}\n"
            f"grasp_assist_used: {out['grasp_assist_used']}\n"
            f"assist_mode: {out['assist_mode']}\n"
            f"finger_actuators_present: {out['finger_actuators_present']}\n"
            f"proxy_finger_actuators_present: {out['proxy_finger_actuators_present']}\n"
            f"proxy_finger_closed: {out['proxy_finger_closed']}\n"
            f"proxy_finger_contact_ready: {out['proxy_finger_contact_ready']}\n"
            f"fallback_palm_only_assist_used: {out['fallback_palm_only_assist_used']}\n"
            f"max_contact_count: {out['max_contact_count']}\n"
            f"palm_contact_count_max: {out['palm_contact_count_max']}  "
            f"left_finger_contact_count_max: {out['left_finger_contact_count_max']}  "
            f"right_finger_contact_count_max: {out['right_finger_contact_count_max']}\n"
            f"grasp_contact_ready: {out['grasp_contact_ready']}\n"
            f"max_penetration_depth: {out['max_penetration_depth']:.5f}\n"
            f"box_initial_z: {out['box_initial_z']:.5f}  "
            f"box_max_z: {out['box_max_z']:.5f}\n"
            f"box_height_delta: {out['box_height_delta']:.5f}\n"
            f"lift_success: {out['lift_success']}\n"
            f"max_assist_force: {out['max_assist_force']:.4f} N\n"
            f"final_box_displacement: {out['final_box_displacement']:.5f} m\n"
            f"motion_complete: {mission_complete}  sim_time: {data.time:.5f}s"
        )

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="G1 locked-base grasp prototype (assist + lift smoke)."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
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
        run_g1_grasp_box(
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
