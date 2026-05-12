#!/usr/bin/env python3
"""
Structured inspection of a MuJoCo MJCF/XML model for Unitree G1 integration.

Standalone; not imported by the pick-and-place prototype.

Usage::
    python simulation/mujoco/g1/inspect_g1_model.py --path simulation/mujoco/g1/assets/g1.xml
    python simulation/mujoco/g1/inspect_g1_model.py --path simulation/mujoco/world.xml --output-json inspect.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_G1_DIR = Path(__file__).resolve().parent
if str(_G1_DIR) not in sys.path:
    sys.path.insert(0, str(_G1_DIR))

import mujoco  # noqa: E402

from load_g1_model import load_model  # noqa: E402


def _joint_type_str(model: mujoco.MjModel, jid: int) -> str:
    jtype = mujoco.mjtJoint(int(model.jnt_type[jid]))
    for attr in sorted(dir(mujoco.mjtJoint)):
        if attr.startswith("_"):
            continue
        if getattr(mujoco.mjtJoint, attr) == jtype:
            return attr
    return repr(jtype)


def _joint_qpos_span(model: mujoco.MjModel, jid: int) -> tuple[int, int]:
    start = int(model.jnt_qposadr[jid])
    end = int(model.jnt_qposadr[jid + 1]) if jid + 1 < model.njnt else model.nq
    return start, max(0, end - start)


def _joint_qvel_span(model: mujoco.MjModel, jid: int) -> tuple[int, int]:
    start = int(model.jnt_dofadr[jid])
    end = int(model.jnt_dofadr[jid + 1]) if jid + 1 < model.njnt else model.nv
    return start, max(0, end - start)


def build_inspection_report(model: mujoco.MjModel) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of joints, bodies, and actuators."""
    bodies_out: list[dict[str, Any]] = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        parent_id = int(model.body_parentid[bid])
        parent_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            if parent_id >= 0
            else ""
        )
        bodies_out.append(
            {
                "id": bid,
                "name": name,
                "parent_id": parent_id,
                "parent_name": parent_name or None,
            }
        )

    joints_out: list[dict[str, Any]] = []
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        body_id = int(model.jnt_bodyid[jid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        qa, qc = _joint_qpos_span(model, jid)
        va, vc = _joint_qvel_span(model, jid)
        joints_out.append(
            {
                "id": jid,
                "name": jname,
                "body_name": body_name,
                "type": _joint_type_str(model, jid),
                "qpos_adr": qa,
                "qpos_dim": qc,
                "qvel_adr": va,
                "qvel_dim": vc,
            }
        )

    actuators_out: list[dict[str, Any]] = []
    for aid in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        actuators_out.append({"id": aid, "name": aname})

    return {
        "summary": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nu": int(model.nu),
            "nq": int(model.nq),
            "nv": int(model.nv),
        },
        "bodies": bodies_out,
        "joints": joints_out,
        "actuators": actuators_out,
    }


def print_inspection(report: dict[str, Any]) -> None:
    s = report["summary"]
    print(f"Number of bodies:    {s['nbody']}")
    print(f"Number of joints:    {s['njnt']}")
    print(f"Number of actuators: {s['nu']}")
    print(f"nq={s['nq']} nv={s['nv']}")
    print()

    print("All body names:")
    for b in report["bodies"]:
        nm = b["name"] if b["name"] else "<unnamed>"
        print(f"  id={b['id']:4d}  name={nm!r}")
    print()

    print("All joints (name, type, qpos adr:dim, qvel adr:dim, body):")
    for j in report["joints"]:
        nm = j["name"] if j["name"] else "<unnamed>"
        print(
            f"  id={j['id']:4d}  {nm!r:32s}  type={j['type']:16s}  "
            f"qpos[{j['qpos_adr']}:{j['qpos_adr'] + j['qpos_dim']}]  "
            f"qvel[{j['qvel_adr']}:{j['qvel_adr'] + j['qvel_dim']}]  "
            f"body={j['body_name']!r}"
        )
    print()

    print("All actuator names:")
    for a in report["actuators"]:
        nm = a["name"] if a["name"] else "<unnamed>"
        print(f"  id={a['id']:4d}  name={nm!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MuJoCo MJCF model (joints, bodies, actuators) for G1 mapping."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to MJCF/XML model file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        metavar="FILE",
        help="Write full inspection dict as JSON to this path.",
    )
    args = parser.parse_args(argv)

    path = Path(args.path).resolve()
    model = load_model(path)
    report = build_inspection_report(model)
    report["source_model"] = str(path)

    print(f"Loaded: {path}\n")
    print_inspection(report)

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
