"""Smoke tests for G1 payload-aware foot-clearance validation."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_payload_foot_clearance.py"
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
    spec = importlib.util.spec_from_file_location("run_g1_payload_foot_clearance", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_payload_foot_clearance")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_payload_foot_clearance_headless_metrics():
    mod = _load_module()
    out = mod.run_g1_payload_foot_clearance(headless=True, timeout=14.0, verbose=False)
    required = {
        "payload_foot_clearance_validation_only",
        "no_locomotion_attempted",
        "phases_completed",
        "right_foot_clearance_success",
        "left_foot_clearance_success",
        "max_right_foot_clearance_m",
        "max_left_foot_clearance_m",
        "right_foot_returned_to_ground",
        "left_foot_returned_to_ground",
        "com_inside_support_all_phases",
        "min_com_margin_m",
        "min_com_margin_single_support_m",
        "min_support_foot_force_n",
        "max_pelvis_lateral_shift_m",
        "max_hip_roll_rad",
        "max_ankle_roll_rad",
        "payload_mass_kg",
        "payload_moment_nm",
        "success",
    }
    assert required.issubset(out)
    assert isinstance(out["success"], bool)
    assert math.isfinite(float(out["min_com_margin_m"]))
    assert out["no_locomotion_attempted"] is True
    assert out["success"] is True
    assert float(out["max_right_foot_clearance_m"]) >= 0.015
    assert float(out["max_left_foot_clearance_m"]) >= 0.015
    assert float(out["min_support_foot_force_n"]) > 120.0
    assert float(out["min_com_margin_single_support_m"]) > 0.005


def test_g1_payload_foot_clearance_phase_contract_and_vertical_lift():
    mod = _load_module()
    out = mod.run_g1_payload_foot_clearance(headless=True, timeout=14.0, verbose=False)
    assert out["phase_sequence"] == [
        "STAND_NEUTRAL",
        "LOAD_HOLD",
        "TORSO_COMPENSATE",
        "SHIFT_TO_LEFT_SUPPORT",
        "UNLOAD_RIGHT_FOOT",
        "LIFT_RIGHT_FOOT_SMALL",
        "HOLD_RIGHT_FOOT_CLEAR",
        "LOWER_RIGHT_FOOT",
        "RETURN_CENTER",
        "SHIFT_TO_RIGHT_SUPPORT",
        "UNLOAD_LEFT_FOOT",
        "LIFT_LEFT_FOOT_SMALL",
        "HOLD_LEFT_FOOT_CLEAR",
        "LOWER_LEFT_FOOT",
        "RETURN_CENTER_FINAL",
        "DONE",
    ]
    assert out["right_foot_returned_to_ground"] is True
    assert out["left_foot_returned_to_ground"] is True
    assert float(out["max_hip_roll_rad"]) <= 0.12
    assert float(out["max_ankle_roll_rad"]) <= 0.10
    assert float(out["max_pelvis_lateral_shift_m"]) <= 0.045
    assert float(out["max_swing_foot_xy_drift_m"]) < 0.003
    assert not any("STEP" in phase for phase in out["phase_sequence"])
    assert not any("WALK" in phase for phase in out["phase_sequence"])
    assert not any("FORWARD" in phase for phase in out["phase_sequence"])


def test_g1_payload_foot_clearance_script_output_includes_metrics():
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--headless",
            "--timeout",
            "14",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    stdout = proc.stdout
    for token in (
        "payload_foot_clearance_validation_only:",
        "phases_completed:",
        "right_foot_clearance_success:",
        "left_foot_clearance_success:",
        "max_right_foot_clearance_m:",
        "max_left_foot_clearance_m:",
        "right_foot_returned_to_ground:",
        "left_foot_returned_to_ground:",
        "com_inside_support_all_phases:",
        "min_com_margin_m:",
        "min_com_margin_single_support_m:",
        "min_support_foot_force_n:",
        "max_pelvis_lateral_shift_m:",
        "max_hip_roll_rad:",
        "max_ankle_roll_rad:",
        "payload_mass_kg:",
        "payload_moment_nm:",
        "success:",
    ):
        assert token in stdout
