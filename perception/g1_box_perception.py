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
    PALM_PLATE_HALF_THICKNESS_X,
    PALM_PLATE_HALF_WIDTH_Y,
    PALM_SURFACE_CLEARANCE,
)

# Dual-arm side grasp: nominal gap from each ±y face plane to the palm plate.
# Dex3 fingers need the palm to stay clearly outside the side faces so closure
# wraps around edges instead of turning into a palm-driven squeeze.
DUAL_SURFACE_CONTACT_CLEARANCE_Y_M = 0.012
# Additional inward motion allowed from contact solver / numerical compliance (diagnostic budget).
DUAL_ALLOWED_SURFACE_COMPRESSION_M = 0.003


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
    viz_right_middle_contact_world: np.ndarray
    viz_left_contact_world: np.ndarray
    viz_left_middle_contact_world: np.ndarray
    viz_right_lower_edge_world: np.ndarray
    viz_left_lower_edge_world: np.ndarray
    viz_front_contact_world: np.ndarray

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
            "viz_right_middle_contact_world": self.viz_right_middle_contact_world.astype(
                float
            ).copy(),
            "viz_left_contact_world": self.viz_left_contact_world.astype(float).copy(),
            "viz_left_middle_contact_world": self.viz_left_middle_contact_world.astype(
                float
            ).copy(),
            "viz_right_lower_edge_world": self.viz_right_lower_edge_world.astype(float).copy(),
            "viz_left_lower_edge_world": self.viz_left_lower_edge_world.astype(float).copy(),
            "viz_front_contact_world": self.viz_front_contact_world.astype(float).copy(),
        }


VIS_SITE_BOX_CENTER = "g1_vis_box_center"
VIS_SITE_NEAR_FACE = "g1_vis_near_face_center"
VIS_SITE_PREGRASP = "g1_vis_palm_pregrasp"
VIS_SITE_APPROACH = "g1_vis_palm_approach"

SITE_BOX_NEAR_FACE_CENTER = "box_near_face_center_site"
SITE_BOX_RIGHT_CONTACT = "box_right_contact_site"
SITE_BOX_RIGHT_MIDDLE_CONTACT = "box_right_middle_contact_site"
SITE_BOX_LEFT_CONTACT = "box_left_contact_site"
SITE_BOX_LEFT_MIDDLE_CONTACT = "box_left_middle_contact_site"
SITE_BOX_RIGHT_LOWER_EDGE = "box_right_lower_edge_site"
SITE_BOX_LEFT_LOWER_EDGE = "box_left_lower_edge_site"
SITE_BOX_FRONT_CONTACT = "box_front_contact_site"


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
        dual_side_clearance_y_m: float = DUAL_SURFACE_CONTACT_CLEARANCE_Y_M,
        dual_near_face_extra_clearance_x_m: float = 0.0,
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
        px_dual = px - float(dual_near_face_extra_clearance_x_m)
        cy = float(center[1])
        gz = float(grasp_height_z)

        palm_pregrasp_target = np.array([px, cy, gz], dtype=float)
        palm_approach_target = palm_pregrasp_target + np.array(
            [0.0, 0.0, float(approach_above_dz_m)], dtype=float
        )

        # Right arm → negative-y face (``left_edge_y``); left arm → positive-y face (``right_edge_y``).
        # Palm plate outer +x presses toward +box_y / −box_y respectively; palm centers stay outside ymin/ymax.
        ry_t = cy - hy - float(palm_plate_half_width_y) - float(dual_side_clearance_y_m)
        ly_t = cy + hy + float(palm_plate_half_width_y) + float(dual_side_clearance_y_m)
        right_dual_pregrasp_target = np.array([px_dual, ry_t, gz], dtype=float)
        left_dual_pregrasp_target = np.array([px_dual, ly_t, gz], dtype=float)
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

        # Debug anchors: face centers at grasp height (sites slide with perception sync).
        cxw = float(center[0])
        hx = float(h[0])
        # Right-hand index/middle column on the −y face (matches Dex3 reach, not box center x).
        finger_col_x = cxw - 0.22 * hx
        # Index/middle sit on slightly different z on the face; blue site is their midpoint.
        # Dex3 palm: index/middle bases offset ±0.0285 m in hand z; use reachable z on the face.
        right_digit_z_sep_m = 0.032
        right_index_face = np.array([finger_col_x, left_edge_y, gz], dtype=float)
        right_middle_face = np.array(
            [finger_col_x, left_edge_y, gz + right_digit_z_sep_m], dtype=float
        )
        viz_right_contact_world = right_index_face.copy()
        viz_right_middle_contact_world = right_middle_face.copy()
        left_digit_z_sep_m = right_digit_z_sep_m
        viz_left_contact_world = np.array([finger_col_x, right_edge_y, gz])
        viz_left_middle_contact_world = np.array(
            [finger_col_x, right_edge_y, gz + left_digit_z_sep_m], dtype=float
        )
        viz_front_contact_world = np.array([near_face_x, cy, gz])

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
            viz_right_middle_contact_world=viz_right_middle_contact_world,
            viz_left_contact_world=viz_left_contact_world,
            viz_left_middle_contact_world=viz_left_middle_contact_world,
            viz_right_lower_edge_world=viz_right_lower_edge_world,
            viz_left_lower_edge_world=viz_left_lower_edge_world,
            viz_front_contact_world=viz_front_contact_world,
        )
        out = res.as_dict()
        # Dex3 semantic grasp hints (world): index/middle on ±y faces, thumbs oppose inward/up, lower support.
        hz = float(grasp_height_z)
        hx, hz_box = float(h[0]), float(h[2])
        lo_z = float(bottom_z) + 0.18 * float(top_z - bottom_z)
        off_face = 0.013 + 0.12 * float(dual_side_clearance_y_m)
        # Per-digit face targets; blue site is the midpoint (``viz_right_contact_world``).
        out["dex3_right_index_target_world"] = right_index_face + np.array(
            [0.0, -off_face * 0.5, 0.0], dtype=float
        )
        out["dex3_right_middle_target_world"] = right_middle_face + np.array(
            [0.0, -off_face * 0.5, 0.0], dtype=float
        )
        out["dex3_right_thumb_target_world"] = np.array(
            [cxw - 0.42 * hx, float(left_edge_y) - off_face * 2.0, hz + 0.025 * hz_box], dtype=float
        )
        out["dex3_right_lower_support_target_world"] = np.array(
            [cxw + 0.05 * hx, float(left_edge_y) - off_face * 0.5, lo_z], dtype=float
        )
        out["dex3_left_index_target_world"] = np.array(
            [cxw - 0.22 * hx, float(right_edge_y) + off_face * 0.5, hz], dtype=float
        )
        out["dex3_left_middle_target_world"] = np.array(
            [cxw - 0.22 * hx, float(right_edge_y) + off_face * 0.5, hz], dtype=float
        )
        out["dex3_left_thumb_target_world"] = np.array(
            [cxw - 0.42 * hx, float(right_edge_y) + off_face * 2.0, hz + 0.025 * hz_box], dtype=float
        )
        out["dex3_left_lower_support_target_world"] = np.array(
            [cxw + 0.05 * hx, float(right_edge_y) + off_face * 0.5, lo_z], dtype=float
        )
        return out

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
            (SITE_BOX_RIGHT_MIDDLE_CONTACT, perception["viz_right_middle_contact_world"]),
            (SITE_BOX_LEFT_CONTACT, perception["viz_left_contact_world"]),
            (SITE_BOX_LEFT_MIDDLE_CONTACT, perception["viz_left_middle_contact_world"]),
            (SITE_BOX_RIGHT_LOWER_EDGE, perception["viz_right_lower_edge_world"]),
            (SITE_BOX_LEFT_LOWER_EDGE, perception["viz_left_lower_edge_world"]),
            (SITE_BOX_FRONT_CONTACT, perception["viz_front_contact_world"]),
        ]
        for site_name, p_w in coords:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if sid < 0:
                continue
            model.site_pos[sid] = _to_local(p_w)
