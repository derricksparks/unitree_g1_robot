"""Smoke test for inspect_g1_model report (prototype world.xml); does not alter sim runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
if str(_G1_DIR) not in sys.path:
    sys.path.insert(0, str(_G1_DIR))

from inspect_g1_model import build_inspection_report  # noqa: E402
from load_g1_model import load_model  # noqa: E402


def test_inspection_json_shape_world_xml() -> None:
    world = PROJECT_ROOT / "simulation" / "mujoco" / "world.xml"
    model = load_model(world)
    report = build_inspection_report(model)

    assert "summary" in report
    assert "bodies" in report
    assert "joints" in report
    assert "actuators" in report

    assert isinstance(report["bodies"], list)
    assert isinstance(report["joints"], list)
    assert isinstance(report["actuators"], list)
    assert len(report["bodies"]) == report["summary"]["nbody"]
    assert len(report["joints"]) == report["summary"]["njnt"]
    assert len(report["actuators"]) == report["summary"]["nu"]

    assert report["summary"]["nbody"] > 0
    assert report["summary"]["njnt"] > 0

    dumped = json.loads(json.dumps(report))
    assert "bodies" in dumped
    assert "joints" in dumped
    assert "actuators" in dumped
