"""Smoke test: direct dual-arm side reach with real contact validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import math
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_direct_side_reach.py"
DEX3_SCENE = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "g1"
    / "g1_reach_box_scene_dex3.xml"
)

pytestmark = pytest.mark.skipif(
    not DEX3_SCENE.is_file(),
    reason="Dex3 dual reach scene not present",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_g1_direct_side_reach", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_direct_side_reach")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_contact_validation_invariants(out: dict) -> None:
    assert out["scripted_lift_enabled"] is False
    assert out["debug_sites_moved_to_fingers"] is False
    assert out["target_sites_fixed_on_box"] is True
    assert float(out["box_lift_delta_z_m"]) < 0.02
    if float(out["stable_dual_contact_time_s"]) <= 0.0:
        assert out["success"] is False
    if (
        int(out["right_fingertip_contact_count_max"]) == 0
        and int(out["left_fingertip_contact_count_max"]) == 0
        and float(out["right_min_finger_box_distance"]) > 0.003
        and float(out["left_min_finger_box_distance"]) > 0.003
    ):
        assert out["success"] is False
    if out["success"]:
        assert float(out["stable_dual_contact_time_s"]) > 0.10
        assert out["right_real_finger_contact"] or out["left_real_finger_contact"]
        assert float(out["max_box_penetration_any_geom"]) <= 0.010


def test_g1_direct_side_reach_headless_default():
    mod = _load_module()
    out = mod.run_g1_direct_side_reach(
        headless=True, timeout=10.0, verbose=False, enable_lift=False
    )
    assert isinstance(out, dict)
    _assert_contact_validation_invariants(out)
    assert out["wrist_orientation_frozen"] is True or out["min_palm_to_contact_target_m"] < 0.04
    assert out["max_palm_penetration_m"] < 0.035
    assert math.isfinite(float(out["box_slip_xy_m"]))
    assert math.isfinite(float(out["box_max_tilt_deg"]))


def test_g1_direct_side_reach_no_assist_runs():
    mod = _load_module()
    out = mod.run_g1_direct_side_reach(
        headless=True, timeout=2.0, verbose=False, no_assist=True, enable_lift=False
    )
    assert isinstance(out, dict)
    assert out["no_assist_mode"] is True
    assert out["scripted_lift_enabled"] is False
    _assert_contact_validation_invariants(out)


def test_g1_direct_side_reach_scripted_lift_visualization_only():
    mod = _load_module()
    out = mod.run_g1_direct_side_reach(
        headless=True, timeout=10.0, verbose=False, scripted_lift=True
    )
    assert isinstance(out, dict)
    assert out["scripted_lift_enabled"] is True
    assert out["debug_sites_moved_to_fingers"] is False
    # Scripted lift must not count as validated grasp success.
    if out["success"]:
        assert out["real_grasp_contact_success"] is True
        assert float(out["stable_dual_contact_time_s"]) > 0.10


def test_g1_direct_side_reach_physical_lift():
    mod = _load_module()
    out = mod.run_g1_direct_side_reach(
        headless=True, timeout=18.0, verbose=False, enable_lift=True
    )
    assert isinstance(out, dict)
    assert out["lift_enabled"] is True
    assert out["scripted_lift_enabled"] is False
    if out["lift_armed"]:
        assert out["lift_phase_reached"] or float(out["box_lift_delta_z_m"]) > 0.01
    if out["lift_success"]:
        assert float(out["box_lift_delta_z_m"]) >= mod.LIFT_SUCCESS_DELTA_Z_M
