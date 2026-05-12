"""Smoke test: G1 locked-base reach-to-box (IK + composite scene)."""

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
REACH_BOX_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_reach_box.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def _load_reach_box_module():
    spec = importlib.util.spec_from_file_location("run_g1_reach_box", REACH_BOX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_reach_box")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_reach_box_headless():
    mod = _load_reach_box_module()
    out = mod.run_g1_reach_box(headless=True, timeout=8.0, verbose=False)
    assert isinstance(out, dict)
    _mp = str(out["model_path"])
    assert _mp.endswith(
        "g1_reach_box_scene.xml"
    ) or _mp.endswith("g1_reach_box_scene_single_arm_offset.xml")
    assert abs(out["sim_time"] - 8.0) < 0.1
    assert out["fix_base_in_world"] is True
    assert out["ik_site_name"] == "right_palm_site"
    assert out["success"] is True
    assert out["min_palm_box_distance"] < 0.09
    assert out["min_palm_precontact_distance_reach_segment"] < 0.09
