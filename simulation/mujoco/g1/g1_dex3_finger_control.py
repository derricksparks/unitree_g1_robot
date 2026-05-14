"""Gradual Dex3 finger position targets (rate-limited, low-passed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import mujoco
import numpy as np

DEX3_FINGER_JOINTS_LEFT: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
DEX3_FINGER_JOINTS_RIGHT: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
)

# --- Named postures (radians): asymmetric Dex3 grasp (index/middle side support, thumb opposes). ---

_FINGER_OPEN_HAND_L: dict[str, float] = {jn: 0.0 for jn in DEX3_FINGER_JOINTS_LEFT}
_FINGER_OPEN_HAND_R: dict[str, float] = {jn: 0.0 for jn in DEX3_FINGER_JOINTS_RIGHT}

_FINGER_PREGRASP_SPREAD_L: dict[str, float] = {
    "left_hand_thumb_0_joint": 0.12,
    "left_hand_thumb_1_joint": -0.18,
    "left_hand_thumb_2_joint": 0.28,
    "left_hand_middle_0_joint": -0.28,
    "left_hand_middle_1_joint": -0.32,
    "left_hand_index_0_joint": -0.28,
    "left_hand_index_1_joint": -0.32,
}
_FINGER_PREGRASP_SPREAD_R: dict[str, float] = {
    "right_hand_thumb_0_joint": 0.12,
    "right_hand_thumb_1_joint": 0.18,
    "right_hand_thumb_2_joint": -0.28,
    "right_hand_middle_0_joint": 0.28,
    "right_hand_middle_1_joint": 0.32,
    "right_hand_index_0_joint": 0.28,
    "right_hand_index_1_joint": 0.32,
}

# Index/middle wrap ±y faces; thumb rolls inward-up; distal joints moderate (not identical closure).
_FINGER_SIDE_SUPPORT_L: dict[str, float] = {
    "left_hand_thumb_0_joint": 0.42,
    "left_hand_thumb_1_joint": 0.22,
    "left_hand_thumb_2_joint": 0.85,
    "left_hand_middle_0_joint": -0.72,
    "left_hand_middle_1_joint": -0.78,
    "left_hand_index_0_joint": -0.72,
    "left_hand_index_1_joint": -0.78,
}
_FINGER_SIDE_SUPPORT_R: dict[str, float] = {
    "right_hand_thumb_0_joint": 0.42,
    "right_hand_thumb_1_joint": -0.22,
    "right_hand_thumb_2_joint": -0.85,
    "right_hand_middle_0_joint": 0.72,
    "right_hand_middle_1_joint": 0.78,
    "right_hand_index_0_joint": 0.72,
    "right_hand_index_1_joint": 0.78,
}

_FINGER_LIFT_HOLD_L: dict[str, float] = {
    "left_hand_thumb_0_joint": 0.52,
    "left_hand_thumb_1_joint": 0.30,
    "left_hand_thumb_2_joint": 1.05,
    "left_hand_middle_0_joint": -0.95,
    "left_hand_middle_1_joint": -1.05,
    "left_hand_index_0_joint": -0.95,
    "left_hand_index_1_joint": -1.05,
}
_FINGER_LIFT_HOLD_R: dict[str, float] = {
    "right_hand_thumb_0_joint": 0.52,
    "right_hand_thumb_1_joint": -0.30,
    "right_hand_thumb_2_joint": -1.05,
    "right_hand_middle_0_joint": 0.95,
    "right_hand_middle_1_joint": 1.05,
    "right_hand_index_0_joint": 0.95,
    "right_hand_index_1_joint": 1.05,
}

_FINGER_RELEASE_L: dict[str, float] = dict(_FINGER_PREGRASP_SPREAD_L)
_FINGER_RELEASE_R: dict[str, float] = dict(_FINGER_PREGRASP_SPREAD_R)

# Legacy aliases (single posture for both hands)
_FINGER_OPEN_L = _FINGER_OPEN_HAND_L
_FINGER_OPEN_R = _FINGER_OPEN_HAND_R
_FINGER_PRE_L = _FINGER_PREGRASP_SPREAD_L
_FINGER_PRE_R = _FINGER_PREGRASP_SPREAD_R
_FINGER_CLOSED_L = _FINGER_SIDE_SUPPORT_L
_FINGER_CLOSED_R = _FINGER_SIDE_SUPPORT_R


def merge_dex3_neutral_into(
    base: dict[str, float],
    *,
    hinge_joint_names: set[str],
) -> dict[str, float]:
    """Fill missing finger hinge entries with open posture for ``apply_neutral_pose`` checks."""
    out = dict(base)
    for jn in hinge_joint_names:
        if jn in out:
            continue
        if jn in DEX3_FINGER_JOINTS_LEFT:
            out[jn] = float(_FINGER_OPEN_L[jn])
        elif jn in DEX3_FINGER_JOINTS_RIGHT:
            out[jn] = float(_FINGER_OPEN_R[jn])
    return out


def dex3_semantic_finger_actuator_groups(
    model: mujoco.MjModel,
) -> dict[str, tuple[str, ...]]:
    """Discover position actuators on Dex3 finger joints, grouped for diagnostics / tuning."""
    groups: dict[str, list[str]] = {
        "right_thumb": [],
        "right_index": [],
        "right_middle": [],
        "left_thumb": [],
        "left_index": [],
        "left_middle": [],
    }
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not aname:
            continue
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if "_hand_thumb_" in aname:
            key = "right_thumb" if aname.startswith("right_") else "left_thumb"
        elif "_hand_index_" in aname:
            key = "right_index" if aname.startswith("right_") else "left_index"
        elif "_hand_middle_" in aname:
            key = "right_middle" if aname.startswith("right_") else "left_middle"
        else:
            continue
        groups[key].append(aname)
    return {k: tuple(sorted(v)) for k, v in groups.items()}


@dataclass
class Dex3FingerController:
    """Semantic finger groups with LP + per-step rate limit on joint targets."""

    lp_alpha: float = 0.22
    max_delta_rad: float = 0.018
    _state: dict[str, float] = field(default_factory=dict)

    def reset(self, qpos_snapshot: Mapping[str, float]) -> None:
        self._state = {
            **{jn: float(qpos_snapshot[jn]) for jn in DEX3_FINGER_JOINTS_LEFT},
            **{jn: float(qpos_snapshot[jn]) for jn in DEX3_FINGER_JOINTS_RIGHT},
        }

    def _clip_step(self, cur: float, tgt: float) -> float:
        d = float(np.clip(tgt - cur, -self.max_delta_rad, self.max_delta_rad))
        return cur + d

    def targets_for_mode(self, mode: str) -> dict[str, float]:
        if mode in ("open_hand", "open"):
            return {**_FINGER_OPEN_HAND_L, **_FINGER_OPEN_HAND_R}
        if mode in ("pregrasp_spread", "pregrasp"):
            return {**_FINGER_PREGRASP_SPREAD_L, **_FINGER_PREGRASP_SPREAD_R}
        if mode in ("side_support_grasp", "close", "closed", "grasp"):
            return {**_FINGER_SIDE_SUPPORT_L, **_FINGER_SIDE_SUPPORT_R}
        if mode == "lift_hold_grasp":
            return {**_FINGER_LIFT_HOLD_L, **_FINGER_LIFT_HOLD_R}
        if mode == "release":
            return {**_FINGER_RELEASE_L, **_FINGER_RELEASE_R}
        raise ValueError(f"unknown finger mode {mode!r}")

    def step(self, mode: str) -> dict[str, float]:
        raw = self.targets_for_mode(mode)
        out: dict[str, float] = {}
        for jn, tgt in raw.items():
            prev = float(self._state.get(jn, tgt))
            blended = (1.0 - self.lp_alpha) * prev + self.lp_alpha * float(tgt)
            out[jn] = self._clip_step(prev, blended)
            self._state[jn] = out[jn]
        return out
