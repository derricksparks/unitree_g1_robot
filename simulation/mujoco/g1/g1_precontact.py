"""Shared pre-grasp side-contact target and palm-plate penetration math for G1 reach/touch demos."""

from __future__ import annotations

from typing import Sequence

from itertools import product

import mujoco
import numpy as np

# Half-extents (m); must match ``box_geom`` ``size`` in ``g1_reach_box_scene.xml`` (0.05 per axis).
BOX_HALF_SIZE = np.array([0.05, 0.05, 0.05], dtype=float)
# Palm collision plate: half-thickness along the hand +x axis in ``right_palm_contact_geom`` MJCF local frame.
PALM_PLATE_HALF_THICKNESS_X = 0.012
# Half-width along palm-local ±y on ``*_palm_contact_geom`` boxes (matches MJCF ``size`` second axis).
PALM_PLATE_HALF_WIDTH_Y = 0.045
# Desired clearance gap from the −x mathematical near-face plane to the palm plate outer face nominal target.
PALM_SURFACE_CLEARANCE = 0.003
# Extra lateral clearance beyond box ±y faces for dual-arm palm centers (outside the obstacle).
OUTSIDE_Y_CLEARANCE = 0.01
# Deprecated name kept for callers that referenced the sphere radius — maps to palm half-thickness semantics.
PALM_CONTACT_RADIUS = PALM_PLATE_HALF_THICKNESS_X

# Palm target z as a fraction of full box height from the bottom face (lower third).
GRASP_HEIGHT_FRACTION_FROM_BOTTOM = 0.30

PENETRATION_FACE_TOLERANCE = 0.001
MAX_PENETRATION_DEPTH_OK_M = 0.005
# Dual-arm milestone: reject success if any monitored palm/proxy geom overlaps this deeply into the box AABB.
MAX_DUAL_BOX_PENETRATION_ANY_GEOM_M = 0.005

TOUCH_DISTANCE_M = 0.025

# When the palm is high above / beside the box, its OBB can still cross the infinite
# near-face plane — ignore penetration unless the palm center lies in this y-z corridor.
SIDE_CONTACT_CORRIDOR_PAD_Y_M = 0.03
SIDE_CONTACT_CORRIDOR_PAD_Z_M = 0.02

APPROACH_ABOVE_OFFSET = np.array([0.0, 0.0, 0.10], dtype=float)


def box_near_face_x(box_center: np.ndarray) -> float:
    return float(box_center[0]) - BOX_HALF_SIZE[0]


def palm_precontact_target_world(box_center: np.ndarray) -> np.ndarray:
    """Pre-grasp palm **plate center** outside the −x face, lower-third height—not box COM."""
    bc = np.asarray(box_center, dtype=float).reshape(3,)
    box_bottom_z = bc[2] - BOX_HALF_SIZE[2]
    box_height = 2.0 * BOX_HALF_SIZE[2]
    grasp_z = box_bottom_z + GRASP_HEIGHT_FRACTION_FROM_BOTTOM * box_height
    near_face_x = bc[0] - BOX_HALF_SIZE[0]
    palm_center_x = (
        near_face_x - PALM_PLATE_HALF_THICKNESS_X - PALM_SURFACE_CLEARANCE
    )
    return np.array([palm_center_x, bc[1], grasp_z], dtype=float)


def palm_plate_max_world_x_extent(
    model: mujoco.MjModel, data: mujoco.MjData, palm_geom_id: int
) -> float:
    """Largest world +x over palm-plate OBB corners (diagnostic — can exceed nominal face plane)."""
    c = np.asarray(data.geom_xpos[palm_geom_id, :3], dtype=float)
    R = np.asarray(data.geom_xmat[palm_geom_id], dtype=float).reshape(3, 3)
    h = np.asarray(model.geom_size[palm_geom_id, :3], dtype=float)
    max_x = -1e300
    for sx, sy, sz in product([-1.0, 1.0], repeat=3):
        offs = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=float)
        px = float(c[0] + (R @ offs)[0])
        max_x = max(max_x, px)
    return max_x


def palm_in_side_contact_corridor(
    palm_xyz: np.ndarray | Sequence[float],
    *,
    left_edge_y: float,
    right_edge_y: float,
    bottom_z: float,
    top_z: float,
    pad_y: float = SIDE_CONTACT_CORRIDOR_PAD_Y_M,
    pad_z: float = SIDE_CONTACT_CORRIDOR_PAD_Z_M,
) -> bool:
    cy = float(np.asarray(palm_xyz, dtype=float).reshape(3,)[1])
    cz = float(np.asarray(palm_xyz, dtype=float).reshape(3,)[2])
    return bool(
        float(left_edge_y) - pad_y <= cy <= float(right_edge_y) + pad_y
        and float(bottom_z) - pad_z <= cz <= float(top_z) + pad_z
    )


def palm_plate_outer_face_x_nominal(contact_geom_center_x_world: float) -> float:
    """Outward +x face of palm plate when local plate +x aligns with world +x (MJCF nominal)."""
    return float(contact_geom_center_x_world) + float(PALM_PLATE_HALF_THICKNESS_X)


def penetration_depth_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    palm_geom_id: int,
    near_face_x: float,
    left_edge_y: float,
    right_edge_y: float,
    bottom_z: float,
    top_z: float,
) -> float:
    """Penetration past the −x box face (+x into box), gated to side-contact corridor in y/z."""
    _ = model  # API symmetry / future use
    c = np.asarray(data.geom_xpos[palm_geom_id, :3], dtype=float)
    if not palm_in_side_contact_corridor(
        c,
        left_edge_y=left_edge_y,
        right_edge_y=right_edge_y,
        bottom_z=bottom_z,
        top_z=top_z,
    ):
        return 0.0
    ox = palm_plate_outer_face_x_nominal(float(c[0]))
    return max(0.0, float(ox) - float(near_face_x))


def penetration_warning(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    palm_geom_id: int,
    near_face_x: float,
    left_edge_y: float,
    right_edge_y: float,
    bottom_z: float,
    top_z: float,
    tol: float = PENETRATION_FACE_TOLERANCE,
) -> bool:
    _ = model
    c = np.asarray(data.geom_xpos[palm_geom_id, :3], dtype=float)
    if not palm_in_side_contact_corridor(
        c,
        left_edge_y=left_edge_y,
        right_edge_y=right_edge_y,
        bottom_z=bottom_z,
        top_z=top_z,
    ):
        return False
    ox = palm_plate_outer_face_x_nominal(float(c[0]))
    return float(ox) > float(near_face_x) + float(tol)


def geom_obb_corners_world(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    """Eight corners of an OBB geom in world frame (shape ``(8, 3)``)."""
    c = np.asarray(data.geom_xpos[geom_id, :3], dtype=float)
    R = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    h = np.asarray(model.geom_size[geom_id, :3], dtype=float)
    corners = []
    for sx, sy, sz in product([-1.0, 1.0], repeat=3):
        offs = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=float)
        corners.append(c + R @ offs)
    return np.vstack(corners)


def geom_max_penetration_into_world_axis_aligned_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> float:
    """
    Conservative penetration metric: maximum ``min(distance-to-each-face-inside)`` over geom corners
    that lie strictly inside the axis-aligned box (world frame).

    Corners outside the box contribute overlap via axis-aligned bbox intersection slab thickness when
    the geom's world AABB intersects the box interior (captures grazing shells).
    """
    corners = geom_obb_corners_world(model, data, geom_id)
    gmin = corners.min(axis=0)
    gmax = corners.max(axis=0)
    ox = min(float(gmax[0]), float(xmax)) - max(float(gmin[0]), float(xmin))
    oy = min(float(gmax[1]), float(ymax)) - max(float(gmin[1]), float(ymin))
    oz = min(float(gmax[2]), float(zmax)) - max(float(gmin[2]), float(zmin))
    slab = 0.0
    if ox > 0.0 and oy > 0.0 and oz > 0.0:
        slab = float(min(ox, oy, oz))

    vert_max = 0.0
    for i in range(corners.shape[0]):
        v = corners[i]
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
            d = min(
                x - xmin,
                xmax - x,
                y - ymin,
                ymax - y,
                z - zmin,
                zmax - z,
            )
            vert_max = max(vert_max, float(d))
    return float(max(slab, vert_max))
