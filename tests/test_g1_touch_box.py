"""Smoke test: G1 locked-base palm touch/contact demo."""

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
TOUCH_BOX_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_touch_box.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def _load_touch_box_module():
    spec = importlib.util.spec_from_file_location("run_g1_touch_box", TOUCH_BOX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_touch_box")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_touch_box_headless():
    mod = _load_touch_box_module()
    out = mod.run_g1_touch_box(headless=True, timeout=8.0, verbose=False)
    assert isinstance(out, dict)
    assert out["initial_contact_before_approach"] is False
    assert out["max_penetration_depth"] <= out["max_penetration_depth_ok_m"]
    assert out["early_phase_mission_violation"] is False
    assert out["penetration_during_touch_hold"] is False
    assert out["success"] is True
    assert (
        out["min_palm_touch_target_distance"] < 0.025
        or out["max_contact_count"] > 0
    )
