"""Dex3 / articulated-hand contact helpers for the dual-arm pipeline (no proxy fingers)."""

from __future__ import annotations

from typing import Any, Mapping

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


def digit_contact_counts_by_role(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_gid: int,
    side: str,
) -> dict[str, int]:
    """Per semantic Dex3 digit contact counts against the box."""
    digits = dex3_digit_geom_ids_by_role(model, side=side)
    return {
        role: fingertip_contact_count(data, box_gid=box_gid, finger_geom_ids=gids)
        for role, gids in digits.items()
    }


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


def dex3_digit_geom_ids_by_role(model: mujoco.MjModel, *, side: str) -> dict[str, set[int]]:
    """Thumb / index / middle collision geoms for fingertip–box distance metrics."""
    if side not in ("left", "right"):
        raise ValueError(side)
    px = f"{side}_"
    roles: dict[str, tuple[str, ...]] = {
        "thumb": ("thumb_2_link",),
        "index": ("index_0_link", "index_1_link"),
        "middle": ("middle_0_link", "middle_1_link"),
    }
    out: dict[str, set[int]] = {k: set() for k in roles}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = model.body(bid).name or ""
        if not bname.startswith(px):
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        for role, tags in roles.items():
            if any(t in bname for t in tags):
                out[role].add(int(gid))
    return out


def dex3_digit_box_distances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    box_gid: int,
    side: str,
    dist_cap: float = 0.18,
) -> dict[str, float]:
    """Minimum convex distance (m) from each digit role's geoms to the box geom."""
    digits = dex3_digit_geom_ids_by_role(model, side=side)
    box_set = {int(box_gid)}
    return {
        "thumb": min_geom_pair_distance(model, data, digits["thumb"], box_set, dist_cap=dist_cap),
        "index": min_geom_pair_distance(model, data, digits["index"], box_set, dist_cap=dist_cap),
        "middle": min_geom_pair_distance(model, data, digits["middle"], box_set, dist_cap=dist_cap),
    }


def dex3_digit_geom_centroid_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    role: str,
) -> np.ndarray | None:
    """World centroid of collision geoms for a semantic digit (None if no geoms)."""
    gids = dex3_digit_geom_ids_by_role(model, side=side).get(role, set())
    if not gids:
        return None
    pts = np.array([np.asarray(data.geom_xpos[int(gid)], dtype=float) for gid in gids])
    return pts.mean(axis=0)


def dex3_digit_side_face_gaps(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    face_y: float,
    approach_from_negative_y: bool,
) -> dict[str, float]:
    """
    Signed gap (m) from each digit centroid to the box ±y side face at ``face_y``.

    Positive = outside the box along the approach direction; ``<= 0`` means the digit
    centroid has reached or passed the face (intersection / penetration in y).
    """
    gaps: dict[str, float] = {}
    for role in ("thumb", "index", "middle"):
        c = dex3_digit_geom_centroid_world(model, data, side=side, role=role)
        if c is None:
            gaps[role] = float("inf")
            continue
        y = float(c[1])
        fy = float(face_y)
        if approach_from_negative_y:
            gaps[role] = fy - y
        else:
            gaps[role] = y - fy
    return gaps


def dex3_digit_distal_world_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    role: str,
) -> np.ndarray | None:
    """World xyz of the distal collision geoms' centroid for one digit."""
    gids = dex3_digit_distal_geom_ids_by_role(model, side=side).get(role, set())
    if not gids:
        return None
    pts = np.array([np.asarray(data.geom_xpos[int(gid)], dtype=float) for gid in gids])
    return pts.mean(axis=0)


def project_point_to_box_side_face(
    point_xyz: np.ndarray,
    *,
    face_y: float,
) -> np.ndarray:
    """Project a world point onto the box side face plane at ``y = face_y``."""
    p = np.asarray(point_xyz, dtype=float).reshape(3).copy()
    p[1] = float(face_y)
    return p


def mujoco_site_world_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> np.ndarray | None:
    """World position of a named site (``None`` if missing)."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        return None
    return np.asarray(data.site_xpos[sid, :3], dtype=float).copy()


def dex3_digit_distal_geom_ids_by_role(model: mujoco.MjModel, *, side: str) -> dict[str, set[int]]:
    """Distal collision geoms per digit (fingertip region) for contact-site distance."""
    if side not in ("left", "right"):
        raise ValueError(side)
    px = f"{side}_"
    roles: dict[str, tuple[str, ...]] = {
        "thumb": ("thumb_2_link",),
        "index": ("index_1_link",),
        "middle": ("middle_1_link",),
    }
    out: dict[str, set[int]] = {k: set() for k in roles}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = model.body(bid).name or ""
        if not bname.startswith(px):
            continue
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        if ct == 0 and ca == 0:
            continue
        for role, tags in roles.items():
            if any(t in bname for t in tags):
                out[role].add(int(gid))
    return out


def dex3_digit_target_distances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xyz: np.ndarray,
    *,
    side: str,
    distal_only: bool = False,
    face_y: float | None = None,
) -> dict[str, float]:
    """Minimum distance (m) from digit collision geoms to a world target point."""
    tgt = np.asarray(target_xyz, dtype=float).reshape(3)
    if face_y is not None:
        tgt = project_point_to_box_side_face(tgt, face_y=float(face_y))
    digits = (
        dex3_digit_distal_geom_ids_by_role(model, side=side)
        if distal_only
        else dex3_digit_geom_ids_by_role(model, side=side)
    )
    out: dict[str, float] = {}
    for role, gids in digits.items():
        if not gids:
            out[role] = float("inf")
            continue
        best = float("inf")
        for gid in gids:
            p = np.asarray(data.geom_xpos[int(gid), :3], dtype=float)
            if face_y is not None:
                p = project_point_to_box_side_face(p, face_y=float(face_y))
            best = min(best, float(np.linalg.norm(p - tgt)))
        out[role] = float(best)
    return out


def dex3_digit_contact_point_distances(
    perception: Mapping[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
) -> dict[str, float]:
    """Euclidean distance (m) from digit geoms to perception Dex3 contact targets."""
    out: dict[str, float] = {}
    for role in ("index", "middle", "thumb"):
        key = f"dex3_{side}_{role}_target_world"
        tgt = perception.get(key)
        if tgt is None:
            out[role] = float("inf")
            continue
        out[role] = float(
            dex3_digit_target_distances(model, data, np.asarray(tgt, dtype=float), side=side).get(
                role, float("inf")
            )
        )
    return out


def dex3_digit_at_contact_site(
    *,
    site_distance_m: float,
    epsilon_m: float = 1e-3,
) -> bool:
    """True when the digit is within ``epsilon_m`` of the box contact site (zero gap)."""
    return float(site_distance_m) <= float(epsilon_m)


def tune_digit_progressive_blend(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    finger_ctrl: Any,
    *,
    side: str,
    role: str,
    box_gid: int,
    hinge_qpos_adrs: Mapping[str, int],
    contact_site_xyz: np.ndarray | None = None,
    face_y: float | None = None,
    u_samples: tuple[float, ...] = (
        0.36,
        0.40,
        0.43,
        0.46,
        0.48,
        0.50,
        0.52,
        0.55,
    ),
) -> tuple[float, float]:
    """
    Pick the progressive side-grasp blend ``u`` that minimizes distance to the box or contact site.

    Returns ``(best_u, best_metric_m)`` (site distance when ``contact_site_xyz`` is set).
    """
    from g1_dex3_finger_control import dex3_digit_joint_names

    jnames = dex3_digit_joint_names(side, role)
    if not jnames:
        return 0.5, float("inf")
    saved = {
        jn: float(data.qpos[int(hinge_qpos_adrs[jn])])
        for jn in jnames
        if jn in hinge_qpos_adrs
    }
    best_u = float(u_samples[len(u_samples) // 2])
    best_d = float("inf")
    for u in u_samples:
        partial = finger_ctrl.targets_progressive_side_grasp(float(u), side=side)
        for jn in jnames:
            if jn not in hinge_qpos_adrs:
                continue
            data.qpos[int(hinge_qpos_adrs[jn])] = float(partial[jn])
        mujoco.mj_forward(model, data)
        if contact_site_xyz is not None:
            d = float(
                dex3_digit_target_distances(
                    model,
                    data,
                    contact_site_xyz,
                    side=side,
                    distal_only=True,
                    face_y=face_y,
                ).get(role, float("inf"))
            )
        else:
            d = float(
                dex3_digit_box_distances(model, data, box_gid=int(box_gid), side=side).get(
                    role, float("inf")
                )
            )
        if d < best_d:
            best_d = d
            best_u = float(u)
    for jn, qv in saved.items():
        data.qpos[int(hinge_qpos_adrs[jn])] = float(qv)
    partial = finger_ctrl.targets_progressive_side_grasp(best_u, side=side)
    for jn in jnames:
        if jn in hinge_qpos_adrs:
            data.qpos[int(hinge_qpos_adrs[jn])] = float(partial[jn])
    mujoco.mj_forward(model, data)
    return best_u, float(best_d)


def snap_grasp_digits_to_world_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    roles: tuple[str, ...],
    target_xyz: np.ndarray,
    joint_names_by_role: Mapping[str, tuple[str, ...] | list[str]],
    max_iters: int = 64,
    step_rad: float = 0.01,
    face_y: float | None = None,
) -> dict[str, float]:
    """Coordinate descent on multiple digits minimizing sum of distal site distances."""
    tgt = np.asarray(target_xyz, dtype=float).reshape(3)
    all_joints: list[str] = []
    for role in roles:
        all_joints.extend(list(joint_names_by_role.get(role, ())))

    def _dists() -> dict[str, float]:
        return dex3_digit_target_distances(
            model, data, tgt, side=side, distal_only=True, face_y=face_y
        )

    def _max_dist() -> float:
        ds = _dists()
        return float(max(float(ds.get(r, float("inf"))) for r in roles))

    best_cost = _max_dist()
    for _ in range(int(max_iters)):
        improved = False
        for jn in all_joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            adr = int(model.jnt_qposadr[jid])
            lo = float(model.jnt_range[jid, 0])
            hi = float(model.jnt_range[jid, 1])
            cur = float(data.qpos[adr])
            for delta in (float(step_rad), -float(step_rad), 1.5 * float(step_rad)):
                trial = float(np.clip(cur + delta, lo, hi))
                if abs(trial - cur) < 1e-9:
                    continue
                data.qpos[adr] = trial
                mujoco.mj_forward(model, data)
                c_try = _max_dist()
                if c_try < best_cost - 1e-8:
                    best_cost = c_try
                    cur = trial
                    improved = True
            data.qpos[adr] = cur
        if not improved:
            break
    mujoco.mj_forward(model, data)
    return {r: float(_dists().get(r, float("inf"))) for r in roles}


def nudge_hinge_joint_toward_site(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_name: str,
    side: str,
    role: str,
    target_xyz: np.ndarray,
    step_rad: float = 0.012,
    trials: int = 28,
) -> float:
    """1-D search on a single hinge to reduce one digit's distal distance to a site."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float("inf")
    adr = int(model.jnt_qposadr[jid])
    lo = float(model.jnt_range[jid, 0])
    hi = float(model.jnt_range[jid, 1])
    cur = float(data.qpos[adr])
    tgt = np.asarray(target_xyz, dtype=float).reshape(3)

    def _dist() -> float:
        return float(
            dex3_digit_target_distances(model, data, tgt, side=side, distal_only=True).get(
                role, float("inf")
            )
        )

    best_d = _dist()
    best_q = cur
    for i in range(int(trials)):
        t = (float(i) / max(int(trials) - 1, 1) - 0.5) * 2.0
        trial = float(np.clip(cur + t * float(step_rad) * 8.0, lo, hi))
        data.qpos[adr] = trial
        mujoco.mj_forward(model, data)
        d = _dist()
        if d < best_d:
            best_d = d
            best_q = trial
    data.qpos[adr] = best_q
    mujoco.mj_forward(model, data)
    return float(best_d)


def snap_digit_joints_to_world_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    role: str,
    target_xyz: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    max_iters: int = 48,
    step_rad: float = 0.011,
    face_y: float | None = None,
) -> float:
    """Coordinate descent: minimize distal-geom distance to ``target_xyz``."""
    if not joint_names:
        return float("inf")
    tgt = np.asarray(target_xyz, dtype=float).reshape(3)

    def _dist() -> float:
        return float(
            dex3_digit_target_distances(
                model, data, tgt, side=side, distal_only=True, face_y=face_y
            ).get(role, float("inf"))
        )

    best_d = _dist()
    for _ in range(int(max_iters)):
        improved = False
        for jn in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            adr = int(model.jnt_qposadr[jid])
            lo = float(model.jnt_range[jid, 0])
            hi = float(model.jnt_range[jid, 1])
            cur = float(data.qpos[adr])
            for delta in (float(step_rad), -float(step_rad), 1.6 * float(step_rad)):
                trial = float(np.clip(cur + delta, lo, hi))
                if abs(trial - cur) < 1e-9:
                    continue
                data.qpos[adr] = trial
                mujoco.mj_forward(model, data)
                d_try = _dist()
                if d_try < best_d - 1e-8:
                    best_d = d_try
                    cur = trial
                    improved = True
            data.qpos[adr] = cur
        if not improved:
            break
    mujoco.mj_forward(model, data)
    return float(best_d)


def nudge_digit_joints_toward_box(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    side: str,
    role: str,
    box_gid: int,
    joint_names: tuple[str, ...] | list[str],
    step_rad: float = 0.014,
    trials: int = 2,
    target_xyz: np.ndarray | None = None,
    target_weight: float = 0.35,
) -> float:
    """
    Hill-climb digit hinge angles to reduce convex distance to the box.

    Returns the best (minimum) box distance achieved for ``role``.
    """
    if not joint_names:
        return float("inf")

    def _metric() -> float:
        d_box = float(
            dex3_digit_box_distances(model, data, box_gid=int(box_gid), side=side).get(
                role, float("inf")
            )
        )
        if target_xyz is None:
            return d_box
        d_pt = float(
            dex3_digit_target_distances(
                model, data, target_xyz, side=side, distal_only=True
            ).get(role, float("inf"))
        )
        w = float(np.clip(target_weight, 0.0, 1.0))
        return w * d_pt + (1.0 - w) * d_box

    best_d = _metric()
    best_q: dict[str, float] = {}
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        cur = float(data.qpos[adr])
        best_q[jn] = cur
        for _ in range(int(trials)):
            for delta in (float(step_rad), -float(step_rad), 2.0 * float(step_rad)):
                trial = float(np.clip(cur + delta, lo, hi))
                if abs(trial - cur) < 1e-9:
                    continue
                data.qpos[adr] = trial
                mujoco.mj_forward(model, data)
                d_try = _metric()
                d_box_try = float(
                    dex3_digit_box_distances(model, data, box_gid=int(box_gid), side=side).get(
                        role, float("inf")
                    )
                )
                if d_box_try < -0.012:
                    continue
                if d_try < best_d - 1e-7:
                    best_d = d_try
                    cur = trial
                    best_q[jn] = trial
            data.qpos[adr] = cur

    for jn, qv in best_q.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(qv)
    mujoco.mj_forward(model, data)
    return float(best_d)


def dex3_digit_intersects_box(
    *,
    box_distance_m: float,
    contact_count: int,
    intersect_epsilon_m: float = 1e-4,
    face_gap_m: float | None = None,
) -> bool:
    """
    True when the digit has reached the box surface (zero convex gap or penetration).

    MuJoCo ``contact_count`` alone is not sufficient — collision margins can report
    contacts while ``mj_geomDistance`` is still positive.
    """
    d = float(box_distance_m)
    if np.isfinite(d) and d <= float(intersect_epsilon_m):
        return True
    if face_gap_m is not None and np.isfinite(face_gap_m) and float(face_gap_m) <= float(
        intersect_epsilon_m
    ):
        return True
    if int(contact_count) > 0 and np.isfinite(d) and d <= max(float(intersect_epsilon_m), 0.0025):
        return True
    return False


def dex3_per_hand_lift_ready_relaxed(
    *,
    side_face_ok: bool,
    palm_contacts: int,
    fingertip_contacts: int,
    min_digit_to_box_m: float,
    near_threshold_m: float = 0.032,
) -> bool:
    """
    Grasp readiness for lift gating: side geometry satisfied and either real finger contact
    or a fingertip digit within ``near_threshold_m`` of the box.
    """
    if not bool(side_face_ok):
        return False
    fc = int(fingertip_contacts)
    if fc >= 1:
        return True
    if float(min_digit_to_box_m) + 1e-9 <= float(near_threshold_m):
        return True
    if int(palm_contacts) >= 1 and float(min_digit_to_box_m) + 1e-9 <= float(near_threshold_m) * 1.35:
        return True
    return False
