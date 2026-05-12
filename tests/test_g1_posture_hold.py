"""Smoke test for G1 posture hold (separate from sliding-base prototype)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_POS_XML = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "g1"
    / "assets"
    / "g1_position_actuated.xml"
)
RUN_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_posture_hold.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not built (run add_position_actuators.py)",
)


def _load_posture_module():
    spec = importlib.util.spec_from_file_location("run_g1_posture_hold", RUN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_posture_hold")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_posture_hold_headless_smoke():
    mod = _load_posture_module()
    out = mod.run_posture_hold(headless=True, timeout=0.5, verbose=False)
    assert isinstance(out, dict)
    assert out["model_path"].endswith("g1_position_actuated.xml")
    assert 0.45 <= out["sim_time"] <= 0.52
    assert abs(out["pelvis_xyz"][2] - mod.DEFAULT_PELVIS_Z) < 1e-5
    assert len(out["pelvis_xyz"]) == 3
    assert isinstance(out["pelvis_roll_rad"], float)
    assert isinstance(out["pelvis_pitch_rad"], float)
    assert out["max_joint_error_rad"] >= 0.0
    assert out["fix_base_in_world"] is True
