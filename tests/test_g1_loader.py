"""Smoke tests for optional G1 model loader (does not run the main simulation)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
if str(_G1_DIR) not in sys.path:
    sys.path.insert(0, str(_G1_DIR))

from load_g1_model import load_model, print_model_summary


def test_load_model_smoke_world_xml(capsys: pytest.CaptureFixture[str]) -> None:
    world_path = PROJECT_ROOT / "simulation" / "mujoco" / "world.xml"
    assert world_path.is_file(), f"missing prototype scene: {world_path}"

    model = load_model(world_path)
    assert model.nbody > 0
    assert model.njnt > 0

    print_model_summary(model)
    out = capsys.readouterr().out
    assert "Bodies" in out
    assert "Joints" in out
    assert "Actuators" in out
