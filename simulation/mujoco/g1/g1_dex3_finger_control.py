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
# Distal joints driven near MuJoCo limits so fingertips can reach the box side face.
_FINGER_SIDE_REACH_MAX_R: dict[str, float] = {
    "right_hand_thumb_0_joint": 0.52,
    "right_hand_thumb_1_joint": -0.30,
    "right_hand_thumb_2_joint": -1.05,
    "right_hand_middle_0_joint": 1.72,
    "right_hand_middle_1_joint": 1.73,
    "right_hand_index_0_joint": 1.52,
    "right_hand_index_1_joint": 1.52,
}
_FINGER_SIDE_REACH_MAX_L: dict[str, float] = {
    "left_hand_thumb_0_joint": 0.52,
    "left_hand_thumb_1_joint": 0.30,
    "left_hand_thumb_2_joint": 1.05,
    "left_hand_middle_0_joint": -1.48,
    "left_hand_middle_1_joint": -1.68,
    "left_hand_index_0_joint": -1.48,
    "left_hand_index_1_joint": -1.48,
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

# Carry posture for a box: index hooks under the lower edge, thumb opposes the
# front face, and middle finger stays mostly extended to avoid box penetration.
_FINGER_BOX_STABLE_SUPPORT_L: dict[str, float] = {
    "left_hand_thumb_0_joint": 0.62,
    "left_hand_thumb_1_joint": 0.42,
    "left_hand_thumb_2_joint": 1.18,
    "left_hand_middle_0_joint": -0.10,
    "left_hand_middle_1_joint": -0.12,
    "left_hand_index_0_joint": -0.68,
    "left_hand_index_1_joint": -0.72,
}
_FINGER_BOX_STABLE_SUPPORT_R: dict[str, float] = {
    "right_hand_thumb_0_joint": 0.62,
    "right_hand_thumb_1_joint": -0.42,
    "right_hand_thumb_2_joint": -1.18,
    "right_hand_middle_0_joint": 0.10,
    "right_hand_middle_1_joint": 0.12,
    "right_hand_index_0_joint": 0.68,
    "right_hand_index_1_joint": 0.72,
}

_FINGER_RELEASE_L: dict[str, float] = dict(_FINGER_PREGRASP_SPREAD_L)
_FINGER_RELEASE_R: dict[str, float] = dict(_FINGER_PREGRASP_SPREAD_R)


def _blend_postures(
    a: dict[str, float], b: dict[str, float], u: float
) -> dict[str, float]:
    uu = float(np.clip(u, 0.0, 1.0))
    return {k: float((1.0 - uu) * float(a[k]) + uu * float(b[k])) for k in a.keys()}


_LIGHT_SIDE_PREPARE_L = _blend_postures(_FINGER_PREGRASP_SPREAD_L, _FINGER_SIDE_SUPPORT_L, 0.32)
_LIGHT_SIDE_PREPARE_R = _blend_postures(_FINGER_PREGRASP_SPREAD_R, _FINGER_SIDE_SUPPORT_R, 0.32)
_SIDE_SUPPORT_PARTIAL_L = _blend_postures(_FINGER_PREGRASP_SPREAD_L, _FINGER_SIDE_SUPPORT_L, 0.38)
_SIDE_SUPPORT_PARTIAL_R = _blend_postures(_FINGER_PREGRASP_SPREAD_R, _FINGER_SIDE_SUPPORT_R, 0.38)

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
    _frozen_at_contact: dict[str, float] = field(default_factory=dict)

    def reset(self, qpos_snapshot: Mapping[str, float]) -> None:
        self._state = {
            **{jn: float(qpos_snapshot[jn]) for jn in DEX3_FINGER_JOINTS_LEFT},
            **{jn: float(qpos_snapshot[jn]) for jn in DEX3_FINGER_JOINTS_RIGHT},
        }
        self._frozen_at_contact = {}

    def freeze_digit_joints(self, joint_names: tuple[str, ...] | list[str]) -> None:
        """Latch current low-pass targets so digits stay fixed on the box."""
        for jn in joint_names:
            if jn in self._state:
                self._frozen_at_contact[jn] = float(self._state[jn])

    def unfreeze_digit_joints(self, joint_names: tuple[str, ...] | list[str]) -> None:
        """Release a prior finger latch so closing can continue."""
        for jn in joint_names:
            self._frozen_at_contact.pop(jn, None)

    def frozen_joint_snapshot(self) -> dict[str, float]:
        return dict(self._frozen_at_contact)

    def _clip_step(self, cur: float, tgt: float) -> float:
        d = float(np.clip(tgt - cur, -self.max_delta_rad, self.max_delta_rad))
        return cur + d

    def targets_progressive_side_grasp(self, blend_u: float, *, side: str) -> dict[str, float]:
        """Blend spread → side support → lift hold for one hand (``blend_u`` in [0, 1])."""
        u = float(np.clip(blend_u, 0.0, 1.0))
        if side == "right":
            pre, mid, full = (
                _FINGER_PREGRASP_SPREAD_R,
                _FINGER_SIDE_SUPPORT_R,
                _FINGER_SIDE_REACH_MAX_R,
            )
            other_pre, other_mid, other_full = (
                _FINGER_PREGRASP_SPREAD_L,
                _FINGER_SIDE_SUPPORT_L,
                _FINGER_SIDE_REACH_MAX_L,
            )
        elif side == "left":
            pre, mid, full = (
                _FINGER_PREGRASP_SPREAD_L,
                _FINGER_SIDE_SUPPORT_L,
                _FINGER_SIDE_REACH_MAX_L,
            )
            other_pre, other_mid, other_full = (
                _FINGER_PREGRASP_SPREAD_R,
                _FINGER_SIDE_SUPPORT_R,
                _FINGER_SIDE_REACH_MAX_R,
            )
        else:
            raise ValueError(side)
        if u <= 0.5:
            uu = u / 0.5
            hand = _blend_postures(pre, mid, uu)
            other = dict(other_pre)
        else:
            uu = (u - 0.5) / 0.5
            hand = _blend_postures(mid, full, uu)
            other = dict(other_mid if u <= 0.75 else other_full)
        if side == "right":
            return {**other, **hand}
        return {**hand, **other}

    def targets_for_mode(self, mode: str) -> dict[str, float]:
        if mode in ("open_hand", "open"):
            return {**_FINGER_OPEN_HAND_L, **_FINGER_OPEN_HAND_R}
        if mode in ("pregrasp_spread", "pregrasp"):
            return {**_FINGER_PREGRASP_SPREAD_L, **_FINGER_PREGRASP_SPREAD_R}
        if mode == "light_side_prepare":
            return {**_LIGHT_SIDE_PREPARE_L, **_LIGHT_SIDE_PREPARE_R}
        if mode == "side_support_partial":
            return {**_SIDE_SUPPORT_PARTIAL_L, **_SIDE_SUPPORT_PARTIAL_R}
        if mode in ("side_support_grasp", "close", "closed", "grasp"):
            return {**_FINGER_SIDE_SUPPORT_L, **_FINGER_SIDE_SUPPORT_R}
        if mode == "lift_hold_grasp":
            return {**_FINGER_LIFT_HOLD_L, **_FINGER_LIFT_HOLD_R}
        if mode in ("box_stable_support_grasp", "stable_box_grasp"):
            return {**_FINGER_BOX_STABLE_SUPPORT_L, **_FINGER_BOX_STABLE_SUPPORT_R}
        if mode == "release":
            return {**_FINGER_RELEASE_L, **_FINGER_RELEASE_R}
        raise ValueError(f"unknown finger mode {mode!r}")

    def state_snapshot(self) -> dict[str, float]:
        """Current low-passed finger joint targets (for hold-freeze without advancing posture)."""
        return dict(self._state)

    def step(self, mode: str) -> dict[str, float]:
        raw = self.targets_for_mode(mode)
        return self._step_raw(raw, max_delta_by_joint=None)

    def step_contact_aware(
        self,
        mode: str,
        *,
        digit_policy: Mapping[str, str],
        index_max_delta_rad: float | None = None,
        close_max_delta_rad: float | None = None,
        close_blend_u: float | None = None,
        close_blend_side: str = "right",
        close_blend_by_role: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Advance semantic digits independently.

        ``digit_policy`` keys are ``{side}_{thumb,index,middle}``; values:
        ``close`` follows the requested posture, ``hold`` freezes the current
        target, and ``open`` eases back toward the pregrasp spread.
        """
        if close_blend_by_role and any(a == "close" for a in digit_policy.values()):
            raw = self.targets_for_mode(mode)
            for key, action in digit_policy.items():
                if action != "close":
                    continue
                side, role = key.split("_", 1)
                u = float(close_blend_by_role.get(role, close_blend_u or 0.55))
                partial = self.targets_progressive_side_grasp(u, side=side)
                for jn in _DIGIT_JOINT_GROUPS.get((side, role), ()):
                    raw[jn] = float(partial[jn])
        elif close_blend_u is not None and any(a == "close" for a in digit_policy.values()):
            raw = self.targets_progressive_side_grasp(
                float(close_blend_u), side=str(close_blend_side)
            )
        else:
            raw = self.targets_for_mode(mode)
        pre = self.targets_for_mode("pregrasp_spread")
        max_delta_by: dict[str, float] = {}
        if index_max_delta_rad is not None and float(index_max_delta_rad) > 0.0:
            lim = float(index_max_delta_rad)
            for jn in _DIGIT_JOINT_GROUPS[("right", "index")] + _DIGIT_JOINT_GROUPS[("left", "index")]:
                max_delta_by[jn] = lim
        close_lim = (
            float(close_max_delta_rad)
            if close_max_delta_rad is not None and float(close_max_delta_rad) > 0.0
            else None
        )
        for key, action in digit_policy.items():
            side, role = key.split("_", 1)
            joints = _DIGIT_JOINT_GROUPS.get((side, role), ())
            if action == "hold":
                for jn in joints:
                    raw[jn] = float(self._state.get(jn, raw[jn]))
            elif action == "open":
                for jn in joints:
                    cur = float(self._state.get(jn, raw[jn]))
                    raw[jn] = 0.68 * cur + 0.32 * float(pre[jn])
            elif action == "close":
                if close_lim is not None:
                    for jn in joints:
                        max_delta_by[jn] = close_lim
                continue
            else:
                raise ValueError(f"unknown Dex3 digit action {action!r} for {key!r}")
        return self._step_raw(raw, max_delta_by_joint=max_delta_by or None)

    def _step_raw(
        self,
        raw: Mapping[str, float],
        *,
        max_delta_by_joint: dict[str, float] | None = None,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for jn, tgt in raw.items():
            if jn in self._frozen_at_contact:
                fv = float(self._frozen_at_contact[jn])
                out[jn] = fv
                self._state[jn] = fv
                continue
            prev = float(self._state.get(jn, tgt))
            blended = (1.0 - self.lp_alpha) * prev + self.lp_alpha * float(tgt)
            lim = (
                float(max_delta_by_joint[jn])
                if max_delta_by_joint and jn in max_delta_by_joint
                else float(self.max_delta_rad)
            )
            d = float(np.clip(blended - prev, -lim, lim))
            out_jn = prev + d
            out[jn] = out_jn
            self._state[jn] = out[jn]
        return out


_DIGIT_JOINT_GROUPS: dict[tuple[str, str], tuple[str, ...]] = {
    ("left", "thumb"): (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ),
    ("left", "index"): (
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ),
    ("left", "middle"): (
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
    ),
    ("right", "thumb"): (
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ),
    ("right", "index"): (
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ),
    ("right", "middle"): (
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ),
}


def dex3_digit_joint_names(side: str, role: str) -> tuple[str, ...]:
    return _DIGIT_JOINT_GROUPS.get((side, role), ())
