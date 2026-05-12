"""Tests for G1 MJCF position actuator injection (assets/g1.xml only)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_XML = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "assets" / "g1.xml"
_G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
if str(_G1_DIR) not in sys.path:
    sys.path.insert(0, str(_G1_DIR))

from add_position_actuators import (  # noqa: E402
    add_position_actuators_file,
)

pytestmark = pytest.mark.skipif(
    not G1_XML.is_file(),
    reason="simulation/mujoco/g1/assets/g1.xml not present",
)


def _count_position_actuators(xml_path: Path) -> int:
    tree = ET.parse(str(xml_path))
    act = tree.getroot().find("actuator")
    if act is None:
        return 0
    return sum(1 for c in act if c.tag == "position")


def test_inject_adds_29_actuators() -> None:
    # meshdir="assets" is resolved relative to the XML file directory; tmp_path breaks mesh load.
    out = G1_XML.parent / "_pytest_g1_position_actuated.xml"
    try:
        add_position_actuators_file(G1_XML, out, kp=50.0)
        assert out.is_file()
        assert _count_position_actuators(out) == 29

        model = mujoco.MjModel.from_xml_path(str(out))
        assert model.nu == 29
        assert model.njnt == 30

        floating_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        assert floating_id >= 0
        for aid in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            assert aname is not None
            assert aname.endswith("_pos_actuator")
            trnid = model.actuator_trnid[aid, 0]
            assert trnid != floating_id
    finally:
        if out.is_file():
            out.unlink()


def test_idempotent_second_run() -> None:
    out = G1_XML.parent / "_pytest_g1_position_actuated_idem.xml"
    try:
        add_position_actuators_file(G1_XML, out)
        add_position_actuators_file(out, out)
        assert _count_position_actuators(out) == 29
        model = mujoco.MjModel.from_xml_path(str(out))
        assert model.nu == 29
    finally:
        if out.is_file():
            out.unlink()


def test_kp_parameter_in_xml(tmp_path: Path) -> None:
    out = tmp_path / "g1_kp.xml"
    add_position_actuators_file(G1_XML, out, kp=123.0)
    text = out.read_text(encoding="utf-8")
    assert 'kp="123.0"' in text or 'kp="123"' in text
