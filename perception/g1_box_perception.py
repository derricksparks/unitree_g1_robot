"""
Simulated RGB-D-style box perception for locked-base G1 pre-grasp:
reads MuJoCo geom truth for the obstacle box (center, OBB extents, faces, grasp height).

Vision/perception selects the grasp point; motion control still respects the actuator model.
Later this can swap to noisy camera-aligned estimates without changing IK consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np

_G1_DIR_FOR_PRECONTACT = Path(__file__).resolve().parents[1] / "simulation" / "mujoco" / "g1"
if str(_G1_DIR_FOR_PRECONTACT) not in sys.path:
    sys.path.insert(0, str(_G1_DIR_FOR_PRECONTACT))

from g1_precontact import (
    OUTSIDE_Y_CLEARANCE,
    PALM_PLATE_HALF_THICKNESS_X,
    PALM_PLATE_HALF_WIDTH_Y,
    PALM_SURFACE_CLEARANCE,
)


@dataclass(frozen=True)
class G1BoxPerceptionResult:
    box_center: np.ndarray
    box_half_size: np.ndarray
    near_face_x: float
    far_face_x: float
    left_edge_y: float
    right_edge_y: float
    bottom_z: float
    top_z: float
    grasp_height_z: float
    palm_pregrasp_target: np.ndarray
    palm_approach_target: np.ndarray
    near_face_center: np.ndarray
    right_dual_pregrasp_target: np.ndarray
    left_dual_pregrasp_target: np.ndarray
    right_dual_approach_target: np.ndarray
    left_dual_approach_target: np.ndarray
    viz_near_face_center_world: np.ndarray
    viz_right_contact_world: np.ndarray
    viz_left_contact_world: np.ndarray
    viz_right_lower_edge_world: np.ndarray
    viz_left_lower_edge_world: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "box_center": self.box_center.astype(float).copy(),
            "box_half_size": self.box_half_size.astype(float).copy(),
            "near_face_x": float(self.near_face_x),
            "far_face_x": float(self.far_face_x),
            "left_edge_y": float(self.left_edge_y),
            "right_edge_y": float(self.right_edge_y),
            "box_y_min": float(self.left_edge_y),
            "box_y_max": float(self.right_edge_y),
            "box_x_min": float(self.near_face_x),
            "box_x_max": float(self.far_face_x),
            "bottom_z": float(self.bottom_z),
            "top_z": float(self.top_z),
            "grasp_height_z": float(self.grasp_height_z),
            "palm_pregrasp_target": self.palm_pregrasp_target.astype(float).copy(),
            "palm_approach_target": self.palm_approach_target.astype(float).copy(),
            "near_face_center": self.near_face_center.astype(float).copy(),
            "right_dual_pregrasp_target": self.right_dual_pregrasp_target.astype(float).copy(),
            "left_dual_pregrasp_target": self.left_dual_pregrasp_target.astype(float).copy(),
            "right_dual_approach_target": self.right_dual_approach_target.astype(float).copy(),
            "left_dual_approach_target": self.left_dual_approach_target.astype(float).copy(),
            "viz_near_face_center_world": self.viz_near_face_center_world.astype(float).copy(),
            "viz_right_contact_world": self.viz_right_contact_world.astype(float).copy(),
            "viz_left_contact_world": self.viz_left_contact_world.astype(float).copy(),
            "viz_right_lower_edge_world": self.viz_right_lower_edge_world.astype(float).copy(),
            "viz_left_lower_edge_world": self.viz_left_lower_edge_world.astype(float).copy(),
        }


VIS_SITE_BOX_CENTER = "g1_vis_box_center"
VIS_SITE_NEAR_FACE = "g1_vis_near_face_center"
VIS_SITE_PREGRASP = "g1_vis_palm_pregrasp"
VIS_SITE_APPROACH = "g1_vis_palm_approach"

SITE_BOX_NEAR_FACE_CENTER = "box_near_face_center_site"
SITE_BOX_RIGHT_CONTACT = "box_right_contact_site"
SITE_BOX_LEFT_CONTACT = "box_left_contact_site"
SITE_BOX_RIGHT_LOWER_EDGE = "box_right_lower_edge_site"
SITE_BOX_LEFT_LOWER_EDGE = "box_left_lower_edge_site"


def _geom_box_corners(
    model: mujoco.MjModel, data: mujoco.MjData, gid: int
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = int(model.geom_type[gid])
    if geom_type != int(mujoco.mjtGeom.mjGEOM_BOX):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
        raise ValueError(f"Geom {nm!r} is not mjGEOM_BOX (type={geom_type})")

    c = np.asarray(data.geom_xpos[gid, :3], dtype=float)
    R = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    h = np.asarray(model.geom_size[gid, :3], dtype=float)

    verts: list[np.ndarray] = []
    for sx, sy, sz in product([-1.0, 1.0], repeat=3):
        offs = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=float)
        verts.append(c + R @ offs)
    m = np.vstack(verts)
    return c.copy(), m


class G1BoxPerception:
    """Ground-truth 'vision' extraction for a labeled box geom in the MuJoCo state."""

    @staticmethod
    def detect_box(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        box_geom_name: str = "box_geom",
        palm_plate_half_thickness_x: float = PALM_PLATE_HALF_THICKNESS_X,
        palm_plate_half_width_y: float = PALM_PLATE_HALF_WIDTH_Y,
        clearance: float = PALM_SURFACE_CLEARANCE,
        outside_y_clearance: float = OUTSIDE_Y_CLEARANCE,
        grasp_fraction_from_bottom: float = 0.30,
        approach_above_dz_m: float = 0.10,
    ) -> dict[str, Any]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, box_geom_name)
        if gid < 0:
            raise ValueError(f"Geom {box_geom_name!r} not found")

        center, verts = _geom_box_corners(model, data, gid)
        xs = verts[:, 0]
        ys = verts[:, 1]
        zs = verts[:, 2]
        near_face_x = float(xs.min())
        far_face_x = float(xs.max())
        left_edge_y = float(ys.min())
        right_edge_y = float(ys.max())
        bottom_z = float(zs.min())
        top_z = float(zs.max())
        box_height = top_z - bottom_z

        grasp_height_z = bottom_z + grasp_fraction_from_bottom * box_height

        h = np.asarray(model.geom_size[gid, :3], dtype=float)
        R = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        hy = float(h[1])

        px = near_face_x - float(palm_plate_half_thickness_x) - float(clearance)
        cy = float(center[1])
        gz = float(grasp_height_z)

        palm_pregrasp_target = np.array([px, cy, gz], dtype=float)
        palm_approach_target = palm_pregrasp_target + np.array(
            [0.0, 0.0, float(approach_above_dz_m)], dtype=float
        )

        ry_t = cy - hy - float(palm_plate_half_width_y) - float(outside_y_clearance)
        ly_t = cy + hy + float(palm_plate_half_width_y) + float(outside_y_clearance)
        right_dual_pregrasp_target = np.array([px, ry_t, gz], dtype=float)
        left_dual_pregrasp_target = np.array([px, ly_t, gz], dtype=float)
        dzv = np.array([0.0, 0.0, float(approach_above_dz_m)], dtype=float)
        right_dual_approach_target = right_dual_pregrasp_target + dzv
        left_dual_approach_target = left_dual_pregrasp_target + dzv

        near_face_center = center + R @ np.array([-h[0], 0.0, 0.0], dtype=float)

        eps = 1e-7
        nf_pts = verts[np.abs(xs - near_face_x) < eps]
        if nf_pts.size > 0:
            nfc_tmp = nf_pts.mean(axis=0).copy()
            nfc_tmp[2] = gz
            viz_near_face_center_world = nfc_tmp
        else:
            viz_near_face_center_world = np.array([near_face_x, cy, gz], dtype=float)

        viz_right_contact_world = np.array([near_face_x, left_edge_y - outside_y_clearance, gz])
        viz_left_contact_world = np.array([near_face_x, right_edge_y + outside_y_clearance, gz])

        rr_corner = verts[
            (np.abs(xs - near_face_x) < eps)
            & (np.abs(ys - left_edge_y) < eps)
            & (np.abs(zs - bottom_z) < eps)
        ]
        viz_right_lower_edge_world = (
            rr_corner.mean(axis=0).copy()
            if rr_corner.shape[0] > 0
            else np.array([near_face_x, left_edge_y, bottom_z], dtype=float)
        )
        ll_corner = verts[
            (np.abs(xs - near_face_x) < eps)
            & (np.abs(ys - right_edge_y) < eps)
            & (np.abs(zs - bottom_z) < eps)
        ]
        viz_left_lower_edge_world = (
            ll_corner.mean(axis=0).copy()
            if ll_corner.shape[0] > 0
            else np.array([near_face_x, right_edge_y, bottom_z], dtype=float)
        )

        res = G1BoxPerceptionResult(
            box_center=center,
            box_half_size=h.copy(),
            near_face_x=near_face_x,
            far_face_x=far_face_x,
            left_edge_y=left_edge_y,
            right_edge_y=right_edge_y,
            bottom_z=bottom_z,
            top_z=top_z,
            grasp_height_z=grasp_height_z,
            palm_pregrasp_target=palm_pregrasp_target,
            palm_approach_target=palm_approach_target,
            near_face_center=near_face_center,
            right_dual_pregrasp_target=right_dual_pregrasp_target,
            left_dual_pregrasp_target=left_dual_pregrasp_target,
            right_dual_approach_target=right_dual_approach_target,
            left_dual_approach_target=left_dual_approach_target,
            viz_near_face_center_world=viz_near_face_center_world,
            viz_right_contact_world=viz_right_contact_world,
            viz_left_contact_world=viz_left_contact_world,
            viz_right_lower_edge_world=viz_right_lower_edge_world,
            viz_left_lower_edge_world=viz_left_lower_edge_world,
        )
        return res.as_dict()

    @staticmethod
    def sync_debug_marker_sites(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        perception: dict[str, Any],
        *,
        box_body_name: str = "reach_target_box",
    ) -> None:
        """Move optional perception debug sites into place (requires ``mj_forward`` on ``data``)."""
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
        if bid < 0:
            return

        R = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
        bpos = np.asarray(data.xpos[bid, :3], dtype=float)

        def _to_local(pw: np.ndarray) -> np.ndarray:
            pw = np.asarray(pw, dtype=float).reshape(3,)
            return R.T @ (pw - bpos)

        coords = [
            (VIS_SITE_BOX_CENTER, perception["box_center"]),
            (VIS_SITE_NEAR_FACE, perception["near_face_center"]),
            (VIS_SITE_PREGRASP, perception["palm_pregrasp_target"]),
            (VIS_SITE_APPROACH, perception["palm_approach_target"]),
            (SITE_BOX_NEAR_FACE_CENTER, perception["viz_near_face_center_world"]),
            (SITE_BOX_RIGHT_CONTACT, perception["viz_right_contact_world"]),
            (SITE_BOX_LEFT_CONTACT, perception["viz_left_contact_world"]),
            (SITE_BOX_RIGHT_LOWER_EDGE, perception["viz_right_lower_edge_world"]),
            (SITE_BOX_LEFT_LOWER_EDGE, perception["viz_left_lower_edge_world"]),
        ]
        for site_name, p_w in coords:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                continue
            model.site_pos[sid] = _to_local(p_w)
