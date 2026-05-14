#!/usr/bin/env python3
"""Scan and load Unitree G1 MJCF assets under assets/unitree_g1/; write JSON inspection report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_G1_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_UNITREE_DIR = _G1_DIR / "assets" / "unitree_g1"
_DEFAULT_OUT = _REPO_ROOT / "results" / "unitree_g1_asset_inspection.json"


def _iter_joint_names(model: Any) -> list[str]:
    out: list[str] = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm:
            out.append(str(nm))
    return sorted(out)


def _iter_body_names(model: Any) -> list[str]:
    out: list[str] = []
    for b in range(model.nbody):
        nm = model.body(b).name
        if nm:
            out.append(str(nm))
    return sorted(out)


def _iter_actuator_names(model: Any) -> list[str]:
    out: list[str] = []
    for a in range(model.nu):
        nm = model.actuator(a).name
        if nm:
            out.append(str(nm))
    return sorted(out)


def _hand_filter(names: list[str], *, side: str) -> list[str]:
    s = side + "_"
    keys = ("hand", "thumb", "index", "middle", "palm")
    return sorted(n for n in names if n.startswith(s) and any(k in n for k in keys))


def _geom_hand_summary(model: Any, *, side: str) -> dict[str, Any]:
    """Per-geom rows attached to bodies whose names look like hand links."""
    rows: list[dict[str, Any]] = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = model.body(bid).name or ""
        if not (bname.startswith(f"{side}_") and "hand" in bname):
            continue
        rows.append(
            {
                "geom_id": gid,
                "name": model.geom(gid).name or "",
                "body": bname,
                "type": int(model.geom_type[gid]),
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
                "group": int(model.geom_group[gid]),
            }
        )
    n_coll = sum(1 for r in rows if r["contype"] != 0 or r["conaffinity"] != 0)
    return {"geom_count_on_hand_bodies": len(rows), "collision_geom_count": n_coll, "geoms": rows}


def _try_load(path: Path) -> dict[str, Any]:
    try:
        import mujoco
    except ImportError as e:
        return {"path": str(path), "ok": False, "error": f"mujoco_import: {e}"}

    try:
        m = mujoco.MjModel.from_xml_path(str(path.resolve()))
    except Exception as e:
        return {"path": str(path), "ok": False, "error": f"{type(e).__name__}: {e}"}

    jnames = _iter_joint_names(m)
    hand_l_j = sorted(n for n in jnames if "left_hand" in n)
    hand_r_j = sorted(n for n in jnames if "right_hand" in n)
    bnames = _iter_body_names(m)
    anames = _iter_actuator_names(m)
    hand_l_act = sorted(n for n in anames if "left_hand" in n)
    hand_r_act = sorted(n for n in anames if "right_hand" in n)

    return {
        "path": str(path),
        "ok": True,
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "nbody": int(m.nbody),
        "njnt": int(m.njnt),
        "ngeom": int(m.ngeom),
        "nsite": int(m.nsite),
        "ntendon": int(getattr(m, "ntendon", 0)),
        "joint_names": jnames,
        "body_names": bnames,
        "actuator_names": anames,
        "hand_joint_names_left": hand_l_j,
        "hand_joint_names_right": hand_r_j,
        "hand_actuator_names_left": hand_l_act,
        "hand_actuator_names_right": hand_r_act,
        "hand_body_names_left": _hand_filter(bnames, side="left"),
        "hand_body_names_right": _hand_filter(bnames, side="right"),
        "hand_geom_summary_left": _geom_hand_summary(m, side="left"),
        "hand_geom_summary_right": _geom_hand_summary(m, side="right"),
        "site_names": sorted(
            str(m.site(s).name)
            for s in range(m.nsite)
            if m.site(s).name
        ),
    }


def _scan_folder(unitree_dir: Path) -> list[Path]:
    cands: list[Path] = []
    for pat in ("*.xml", "*.mjcf"):
        cands.extend(sorted(unitree_dir.glob(pat)))
    # de-dup, stable
    seen: set[str] = set()
    out: list[Path] = []
    for p in cands:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _compare_to_pipeline_model(unitree_report: dict[str, Any], pipeline_xml: Path) -> dict[str, Any]:
    try:
        import mujoco

        pm = mujoco.MjModel.from_xml_path(str(pipeline_xml.resolve()))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    p_joints = set(_iter_joint_names(pm))
    p_bodies = set(_iter_body_names(pm))
    p_sites = {str(pm.site(s).name) for s in range(pm.nsite) if pm.site(s).name}

    out: dict[str, Any] = {
        "ok": True,
        "pipeline_xml": str(pipeline_xml),
        "pipeline_nu": int(pm.nu),
        "pipeline_joint_count": len(p_joints),
        "pipeline_site_names": sorted(p_sites),
    }

    for key in ("g1_with_hands.xml", "g1.xml"):
        rec = next((r for r in unitree_report.get("loads", []) if r.get("path", "").endswith(key)), None)
        if not rec or not rec.get("ok"):
            out[key] = {"skipped": True}
            continue
        uj = set(rec.get("joint_names", []))
        ub = set(rec.get("body_names", []))
        us = set(rec.get("site_names", []))
        out[key] = {
            "joint_names_intersection_pipeline": sorted(p_joints & uj),
            "joint_names_only_in_pipeline": sorted(p_joints - uj)[:80],
            "joint_names_only_in_unitree": sorted(uj - p_joints)[:80],
            "body_names_intersection_pipeline": sorted(p_bodies & ub)[:60],
            "body_names_only_in_pipeline": sorted(p_bodies - ub)[:40],
            "site_names_intersection_pipeline": sorted(p_sites & us),
            "site_names_only_in_pipeline": sorted(p_sites - us),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--unitree-dir",
        type=Path,
        default=_DEFAULT_UNITREE_DIR,
        help="Path to assets/unitree_g1 (default: next to this script).",
    )
    ap.add_argument(
        "--pipeline-xml",
        type=Path,
        default=_G1_DIR / "assets" / "g1_position_actuated.xml",
        help="Current manipulation MJCF for name comparison.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="JSON report output path.",
    )
    ns = ap.parse_args(argv)

    unitree_dir: Path = ns.unitree_dir
    out_path: Path = ns.out

    report: dict[str, Any] = {
        "unitree_dir": str(unitree_dir.resolve()),
        "folder_exists": unitree_dir.is_dir(),
        "candidates": [],
        "loads": [],
        "mesh_stl_count": 0,
        "urdf_files": [],
        "comparison_vs_pipeline": {},
    }

    if not unitree_dir.is_dir():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"error": "unitree_dir_missing", "path": str(unitree_dir)}, indent=2))
        return 1

    assets_sub = unitree_dir / "assets"
    if assets_sub.is_dir():
        report["mesh_stl_count"] = len(list(assets_sub.glob("*.STL"))) + len(list(assets_sub.glob("*.stl")))

    report["urdf_files"] = [str(p) for p in sorted(unitree_dir.rglob("*.urdf"))]

    cands = _scan_folder(unitree_dir)
    report["candidates"] = [str(p) for p in cands]

    for p in cands:
        report["loads"].append(_try_load(p))

    report["comparison_vs_pipeline"] = _compare_to_pipeline_model(report, ns.pipeline_xml)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
