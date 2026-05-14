"""Dex3 / articulated-hand contact helpers for the dual-arm pipeline (no proxy fingers)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


def enumerate_hand_contact_geoms(
    model: mujoco.MjModel,
    *,
    side: str,
) -> list[int]:
    """Geom ids under ``{side}_*`` hand / wrist bodies that can touch the box (non-zero contype/conaffinity)."""
    if side not in ("left", "right"):
        raise ValueError(side)
    px = f"{side}_"
    out: list[int] = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = model.body(bid).name or ""
        if not bname.startswith(px):
            continue
        if "hand" not in bname and bname != f"{side}_wrist_yaw_link":
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        out.append(int(gid))
    palm_gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_palm_contact_geom"
    )
    if palm_gid >= 0 and palm_gid not in out:
        out.append(int(palm_gid))
    return sorted(set(out))


def fingertip_contact_count(
    data: mujoco.MjData,
    *,
    box_gid: int,
    finger_geom_ids: set[int],
) -> int:
    n = 0
    for cid in range(data.ncon):
        c = data.contact[cid]
        if box_gid not in (c.geom1, c.geom2):
            continue
        other = int(c.geom2 if c.geom1 == box_gid else c.geom1)
        if other in finger_geom_ids:
            n += 1
    return int(n)


def palm_contact_count(
    data: mujoco.MjData,
    *,
    box_gid: int,
    palm_geom_id: int,
) -> int:
    if palm_geom_id < 0:
        return 0
    n = 0
    for cid in range(data.ncon):
        c = data.contact[cid]
        if box_gid not in (c.geom1, c.geom2):
            continue
        other = int(c.geom2 if c.geom1 == box_gid else c.geom1)
        if other == palm_geom_id:
            n += 1
    return int(n)


def hand_box_contact_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_gid: int,
    side: str,
    palm_geom_id: int,
) -> dict[str, Any]:
    fg = set(enumerate_hand_contact_geoms(model, side=side))
    if palm_geom_id >= 0:
        fg.discard(palm_geom_id)
    return {
        "side": side,
        "finger_geom_count": len(fg),
        "fingertip_contacts": fingertip_contact_count(data, box_gid=box_gid, finger_geom_ids=fg),
        "palm_contacts": palm_contact_count(data, box_gid=box_gid, palm_geom_id=palm_geom_id),
    }


def fingertip_collision_geom_ids(model: mujoco.MjModel, *, side: str) -> set[int]:
    """Distal link collision geoms (thumb / index / middle) for clearance metrics."""
    if side not in ("left", "right"):
        raise ValueError(side)
    px = f"{side}_"
    tags = ("thumb_2_link", "index_1_link", "middle_1_link")
    out: set[int] = set()
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = model.body(bid).name or ""
        if not bname.startswith(px):
            continue
        if not any(t in bname for t in tags):
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        out.add(int(gid))
    return out


def min_geom_pair_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gids_a: set[int],
    gids_b: set[int],
    *,
    dist_cap: float = 0.2,
) -> float:
    """Minimum MuJoCo convex distance between any geom in ``gids_a`` and any in ``gids_b``."""
    fromto = np.zeros(6, dtype=np.float64)
    best = float("inf")
    for ga in gids_a:
        for gb in gids_b:
            if ga < 0 or gb < 0 or ga == gb:
                continue
            d = float(
                mujoco.mj_geomDistance(model, data, int(ga), int(gb), float(dist_cap), fromto)
            )
            best = min(best, d)
    return float(best)


def min_within_set_geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gids: set[int],
    *,
    dist_cap: float = 0.12,
) -> float:
    """Minimum distance among distinct geoms in one set (same-hand finger clearance)."""
    fromto = np.zeros(6, dtype=np.float64)
    best = float("inf")
    gl = sorted(g for g in gids if g >= 0)
    for i, ga in enumerate(gl):
        for gb in gl[i + 1 :]:
            d = float(
                mujoco.mj_geomDistance(model, data, int(ga), int(gb), float(dist_cap), fromto)
            )
            best = min(best, d)
    return float(best)


def dex3_finger_clearance_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, float]:
    """Cross-hand and same-hand fingertip separation (m); larger is safer."""
    r = fingertip_collision_geom_ids(model, side="right")
    l = fingertip_collision_geom_ids(model, side="left")
    return {
        "cross_hand_fingertip_min_m": min_geom_pair_distance(model, data, r, l, dist_cap=0.25),
        "right_hand_fingertip_min_m": min_within_set_geom_distance(model, data, r, dist_cap=0.12),
        "left_hand_fingertip_min_m": min_within_set_geom_distance(model, data, l, dist_cap=0.12),
    }


def dex3_per_hand_grasp_contact_ready(*, palm_contacts: int, fingertip_contacts: int) -> bool:
    """At least one finger contact plus either palm or a second finger line (side support)."""
    pc = int(palm_contacts)
    fc = int(fingertip_contacts)
    return bool(fc >= 1 and (pc >= 1 or fc >= 2))
