#!/usr/bin/env python3
"""
Inject MuJoCo <position> actuators for each hinge joint in a G1 MJCF.

Skips floating_base_joint (free joint is a separate <freejoint> element).
Idempotent: re-running does not duplicate actuators for the same joint.

Usage::
    python simulation/mujoco/g1/add_position_actuators.py \\
      --input simulation/mujoco/g1/assets/g1.xml \\
      --output simulation/mujoco/g1/assets/g1_position_actuated.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FLOATING_BASE_JOINT_NAME = "floating_base_joint"


def _hinge_joint_names(root: ET.Element) -> list[str]:
    """Document-order hinge joints under <mujoco> (excludes freejoint)."""
    names: list[str] = []
    for el in root.iter():
        if el.tag != "joint":
            continue
        jtype = el.get("type", "hinge")
        if jtype != "hinge":
            continue
        name = el.get("name")
        if not name or name == FLOATING_BASE_JOINT_NAME:
            continue
        names.append(name)
    return names


def _ensure_actuator_section(mujoco_el: ET.Element) -> ET.Element:
    act = mujoco_el.find("actuator")
    if act is not None:
        return act
    act = ET.Element("actuator")
    insert_at = None
    for idx, child in enumerate(mujoco_el):
        if child.tag in ("contact", "sensor", "size", "visual", "statistic"):
            insert_at = idx
            break
    if insert_at is None:
        mujoco_el.append(act)
    else:
        mujoco_el.insert(insert_at, act)
    return act


def _existing_position_joint_targets(actuator_el: ET.Element) -> set[str]:
    joints: set[str] = set()
    for child in actuator_el:
        if child.tag != "position":
            continue
        jn = child.get("joint")
        if jn:
            joints.add(jn)
    return joints


def inject_position_actuators(
    tree: ET.ElementTree,
    *,
    kp: float = 50.0,
) -> ET.ElementTree:
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError("Root element must be <mujoco>")

    hinges = _hinge_joint_names(root)
    actuator_el = _ensure_actuator_section(root)
    existing = _existing_position_joint_targets(actuator_el)

    for jname in hinges:
        if jname in existing:
            continue
        act_name = f"{jname}_pos_actuator"
        pos_el = ET.Element(
            "position",
            {"name": act_name, "joint": jname, "kp": str(kp)},
        )
        actuator_el.append(pos_el)
        existing.add(jname)

    return tree


def add_position_actuators_file(
    input_path: Path,
    output_path: Path,
    *,
    kp: float = 50.0,
) -> None:
    tree = ET.parse(str(input_path))
    inject_position_actuators(tree, kp=kp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Add position actuators for hinge joints (skip floating base)."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kp", type=float, default=50.0, help="Position actuator kp (default 50)")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    add_position_actuators_file(args.input.resolve(), args.output.resolve(), kp=args.kp)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
