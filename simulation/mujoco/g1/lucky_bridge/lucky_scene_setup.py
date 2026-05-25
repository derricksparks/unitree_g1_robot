"""Scene setup and ground-truth detection for Lucky warehouse milestones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


TABLE_BODY_NAME = "table"
TABLE_TOP_GEOM_NAME = "table_top"
BOX_BODY_NAME = "red_block"
BOX_GEOM_NAMES = ("red_cylinder", "red_cap_top", "red_cap_bot")
PLACE_TABLE_BODY_NAME = "table_white"


@dataclass(frozen=True)
class BoxDetection:
    box_detected: bool
    box_position_world: tuple[float, float, float]
    table_position_world: tuple[float, float, float]
    left_grasp_target_world: tuple[float, float, float]
    right_grasp_target_world: tuple[float, float, float]
    grasp_target_height_above_table_m: float
    grasp_targets_above_table: bool
    grasp_targets_clear_table_edge: bool
    detection_mode: str = "mujoco_ground_truth"


def _id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, obj_type, name))


def _geom_half_extents(model: mujoco.MjModel, geom_name: str) -> np.ndarray:
    gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return np.zeros(3, dtype=float)
    return np.asarray(model.geom_size[gid, :3], dtype=float).copy()


def table_surface_z(model: mujoco.MjModel) -> float:
    gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_TOP_GEOM_NAME)
    if gid < 0:
        return 0.733
    bid = int(model.geom_bodyid[gid])
    return float(model.body_pos[bid, 2] + model.geom_pos[gid, 2] + model.geom_size[gid, 2])


def table_front_edge_x(model: mujoco.MjModel) -> float:
    gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_TOP_GEOM_NAME)
    if gid < 0:
        return -0.049
    bid = int(model.geom_bodyid[gid])
    return float(model.body_pos[bid, 0] + model.geom_pos[gid, 0] - model.geom_size[gid, 0])


def box_half_extents(model: mujoco.MjModel) -> np.ndarray:
    size = _geom_half_extents(model, "red_cylinder")
    if size.size < 3:
        return np.array([0.09, 0.06, 0.05], dtype=float)
    return np.asarray(size[:3], dtype=float)


def compute_pick_point(model: mujoco.MjModel, *, pick_stand_off: float, box_y: float) -> tuple[float, float]:
    return (float(table_front_edge_x(model) - float(pick_stand_off)), float(box_y))


def setup_lucky_pick_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    table_x: float = 0.351,
    box_y: float = 0.026,
    box_size_xyz: tuple[float, float, float] = (0.18, 0.12, 0.10),
    box_edge_offset: float = 0.08,
) -> dict[str, Any]:
    """Place the existing Lucky table and red block into a stable pick scene."""
    table_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
    box_jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    table_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_TOP_GEOM_NAME)
    box_gids = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in BOX_GEOM_NAMES]

    if table_bid >= 0:
        model.body_pos[table_bid, 0] = float(table_x)
        model.body_pos[table_bid, 1] = 0.0
    if table_gid >= 0:
        model.geom_size[table_gid, :3] = [0.4, 0.25, 0.02]
        model.geom_contype[table_gid] = 1
        model.geom_conaffinity[table_gid] = 1
        model.geom_friction[table_gid, :3] = [1.6, 0.2, 0.02]
    for gid in range(model.ngeom):
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if name.startswith("table_leg_"):
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1
            model.geom_friction[gid, :3] = [1.2, 0.1, 0.01]

    half = np.asarray(box_size_xyz, dtype=float).reshape(3) * 0.5
    main_box_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_cylinder")
    if main_box_gid >= 0:
        model.geom_type[main_box_gid] = int(mujoco.mjtGeom.mjGEOM_BOX)
        model.geom_size[main_box_gid, :3] = half
        model.geom_pos[main_box_gid, :3] = 0.0
        model.geom_contype[main_box_gid] = 1
        model.geom_conaffinity[main_box_gid] = 1
        model.geom_friction[main_box_gid, :3] = [3.5, 0.25, 0.03]
    for cap_name in ("red_cap_top", "red_cap_bot"):
        cap_gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, cap_name)
        if cap_gid >= 0:
            model.geom_contype[cap_gid] = 0
            model.geom_conaffinity[cap_gid] = 0
            model.geom_size[cap_gid, :3] = 1e-4

    surface_z = table_surface_z(model)
    front_edge_x = table_front_edge_x(model)
    box_x = front_edge_x + max(float(box_edge_offset), float(half[0]) + 0.01)
    stable_box_z = surface_z + float(half[2]) + 0.003

    if box_jid >= 0:
        qadr = int(model.jnt_qposadr[box_jid])
        data.qpos[qadr : qadr + 7] = [float(box_x), float(box_y), stable_box_z, 1.0, 0.0, 0.0, 0.0]
        vadr = int(model.jnt_dofadr[box_jid])
        data.qvel[vadr : vadr + 6] = 0.0

    mujoco.mj_forward(model, data)

    table_collision = bool(table_gid >= 0 and model.geom_contype[table_gid] != 0 and model.geom_conaffinity[table_gid] != 0)
    box_collision = bool(main_box_gid >= 0 and model.geom_contype[main_box_gid] != 0 and model.geom_conaffinity[main_box_gid] != 0)
    table_full_top = bool(table_gid >= 0 and np.allclose(model.geom_size[table_gid, :3], np.array([0.4, 0.25, 0.02]), atol=1e-6))
    table_leg_collision = True
    for gid in range(model.ngeom):
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if name.startswith("table_leg_"):
            table_leg_collision = table_leg_collision and bool(model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0)
    box_pos = detect_box_ground_truth(model, data).box_position_world
    box_on_table = bool(abs(float(box_pos[2]) - stable_box_z) < 0.03)
    box_fell_through = bool(float(box_pos[2]) < surface_z - 0.02)
    center_edge_dist = float(float(box_pos[0]) - front_edge_x)
    table_edge_clearance = float(center_edge_dist)
    box_size = tuple(float(x) for x in box_size_xyz)
    graspable = bool(0.05 <= box_size[0] <= 0.30 and 0.05 <= box_size[1] <= 0.24 and 0.04 <= box_size[2] <= 0.18)
    return {
        "box_initial_pose_valid": bool(box_collision and table_collision and box_on_table and not box_fell_through),
        "box_on_table": box_on_table,
        "box_fell_through_table": box_fell_through,
        "table_collision_detected": table_collision,
        "pick_table_full_top_dimension": table_full_top,
        "pick_table_leg_collision_enabled": table_leg_collision,
        "pick_table_top_half_extents": [0.4, 0.25, 0.02],
        "box_collision_detected": box_collision,
        "box_size_xyz": list(box_size),
        "box_graspable_size": graspable,
        "table_front_edge_x": float(front_edge_x),
        "box_near_table_edge": bool(abs(center_edge_dist - float(box_edge_offset)) <= 0.03),
        "box_distance_from_table_front_edge_m": float(center_edge_dist),
        "table_edge_clearance_for_hands_m": table_edge_clearance,
        "table_surface_z": float(surface_z),
        "box_initial_z": float(box_pos[2]),
    }


def detect_box_ground_truth(model: mujoco.MjModel, data: mujoco.MjData) -> BoxDetection:
    box_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME)
    table_bid = _id(model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY_NAME)
    if box_bid < 0:
        zero = (0.0, 0.0, 0.0)
        return BoxDetection(False, zero, zero, zero, zero, 0.0, False, False)

    box_pos = np.asarray(data.xpos[box_bid, :3], dtype=float)
    table_pos = (
        np.asarray(data.xpos[table_bid, :3], dtype=float)
        if table_bid >= 0
        else np.zeros(3, dtype=float)
    )
    half = box_half_extents(model)
    surface_z = table_surface_z(model)
    front_edge_x = table_front_edge_x(model)
    target_x = float(front_edge_x - 0.015)
    target_z = float(box_pos[2] + min(0.045, 0.85 * float(half[2])))
    side_offset = float(half[1] + 0.045)
    left_target = np.array([target_x, box_pos[1] + side_offset, target_z], dtype=float)
    right_target = np.array([target_x, box_pos[1] - side_offset, target_z], dtype=float)
    height_above_table = float(target_z - surface_z)
    targets_above = bool(height_above_table > 0.02)
    targets_clear_edge = bool(target_x >= front_edge_x - 0.03 and height_above_table > 0.02)
    return BoxDetection(
        True,
        tuple(float(x) for x in box_pos),
        tuple(float(x) for x in table_pos),
        tuple(float(x) for x in left_target),
        tuple(float(x) for x in right_target),
        height_above_table,
        targets_above,
        targets_clear_edge,
    )
