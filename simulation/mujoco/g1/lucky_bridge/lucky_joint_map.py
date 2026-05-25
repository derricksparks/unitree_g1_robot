"""Joint ownership and MuJoCo mapping for Lucky G1 policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


LEG_JOINT_TOKENS = ("hip", "knee", "ankle")
WAIST_JOINT_TOKENS = ("waist",)
ARM_JOINT_TOKENS = ("shoulder", "elbow", "wrist")


@dataclass(frozen=True)
class LuckyJointMap:
    joint_names: tuple[str, ...]
    qpos_indices: dict[str, int]
    qvel_indices: dict[str, int]
    actuator_ids: dict[str, int]
    default_joint_pos: np.ndarray
    action_scales: np.ndarray
    leg_indices: tuple[int, ...]
    waist_indices: tuple[int, ...]
    arm_indices: tuple[int, ...]
    controlled_indices: tuple[int, ...]
    missing_actuators: tuple[str, ...]

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def mapping_ok(self) -> bool:
        return len(self.missing_actuators) == 0


def is_leg_joint(name: str) -> bool:
    return any(token in name for token in LEG_JOINT_TOKENS)


def is_waist_joint(name: str) -> bool:
    return any(token in name for token in WAIST_JOINT_TOKENS)


def is_arm_joint(name: str) -> bool:
    return any(token in name for token in ARM_JOINT_TOKENS)


def build_lucky_joint_map(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    control_waist: bool = True,
) -> LuckyJointMap:
    joint_names = tuple(str(n) for n in config["joint_names"])
    defaults = config.get("default_joint_pos", {})
    scales = config.get("action_scales", {})

    qpos_indices: dict[str, int] = {}
    qvel_indices: dict[str, int] = {}
    actuator_ids: dict[str, int] = {}
    missing: list[str] = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid >= 0:
            qpos_indices[name] = int(model.jnt_qposadr[jid])
            qvel_indices[name] = int(model.jnt_dofadr[jid])
        if aid >= 0:
            actuator_ids[name] = int(aid)
        else:
            missing.append(name)

    default_joint_pos = np.array(
        [float(defaults.get(name, 0.0)) for name in joint_names],
        dtype=np.float32,
    )
    action_scales = np.array(
        [float(scales.get(name, 1.0)) for name in joint_names],
        dtype=np.float32,
    )
    leg_indices = tuple(i for i, name in enumerate(joint_names) if is_leg_joint(name))
    waist_indices = tuple(i for i, name in enumerate(joint_names) if is_waist_joint(name))
    arm_indices = tuple(i for i, name in enumerate(joint_names) if is_arm_joint(name))
    controlled = tuple(sorted((*leg_indices, *(waist_indices if control_waist else ()))))
    return LuckyJointMap(
        joint_names=joint_names,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        actuator_ids=actuator_ids,
        default_joint_pos=default_joint_pos,
        action_scales=action_scales,
        leg_indices=leg_indices,
        waist_indices=waist_indices,
        arm_indices=arm_indices,
        controlled_indices=controlled,
        missing_actuators=tuple(missing),
    )


def action_to_joint_targets(action: np.ndarray, joint_map: LuckyJointMap) -> np.ndarray:
    raw = np.asarray(action, dtype=np.float32).reshape(joint_map.action_dim)
    return joint_map.default_joint_pos + raw * joint_map.action_scales
