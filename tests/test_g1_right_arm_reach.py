"""Smoke test: G1 right-arm reach (locked base), separate from pick-place prototype."""

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
REACH_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_right_arm_reach.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def _load_reach_module():
    spec = importlib.util.spec_from_file_location("run_g1_right_arm_reach", REACH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_right_arm_reach")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_right_arm_reach_headless_smoke():
    mod = _load_reach_module()
    out = mod.run_right_arm_reach(headless=True, timeout=1.0, verbose=False)
    assert isinstance(out, dict)
    assert out["model_path"].endswith("g1_position_actuated.xml")
    assert abs(out["sim_time"] - 1.0) < 0.06
    assert out["fix_base_in_world"] is True
    assert out["max_right_arm_target_delta_from_neutral"] > 0.12
