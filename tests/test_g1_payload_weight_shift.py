"""Smoke tests for G1 payload-aware weight shifting."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_payload_weight_shift.py"
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
    spec = importlib.util.spec_from_file_location("run_g1_payload_weight_shift", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_payload_weight_shift")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_payload_weight_shift_headless_metrics():
    mod = _load_module()
    out = mod.run_g1_payload_weight_shift(headless=True, timeout=10.0, verbose=False)
    required = {
        "payload_weight_shift_validation_only",
        "no_locomotion_attempted",
        "phases_completed",
        "combined_robot_box_com",
        "support_polygon_xy",
        "com_inside_support_all_phases",
        "min_com_margin_m",
        "min_com_margin_left_shift_m",
        "min_com_margin_right_shift_m",
        "left_foot_force_n",
        "right_foot_force_n",
        "min_left_foot_force_n",
        "min_right_foot_force_n",
        "max_left_foot_force_ratio",
        "max_right_foot_force_ratio",
        "max_hip_roll_rad",
        "max_ankle_roll_rad",
        "payload_mass_kg",
        "payload_moment_nm",
        "torso_pitch_compensation_rad",
        "success",
    }
    assert required.issubset(out)
    assert isinstance(out["com_inside_support_all_phases"], bool)
    assert isinstance(out["success"], bool)
    assert out["success"] is True
    assert math.isfinite(float(out["min_com_margin_m"]))
    assert float(out["min_com_margin_m"]) > 0.02
    assert float(out["min_left_foot_force_n"]) > 20.0
    assert float(out["min_right_foot_force_n"]) > 20.0


def test_g1_payload_weight_shift_phase_contract():
    mod = _load_module()
    out = mod.run_g1_payload_weight_shift(headless=True, timeout=10.0, verbose=False)
    assert out["no_locomotion_attempted"] is True
    assert out["phase_sequence"] == [
        "STAND_NEUTRAL",
        "LOAD_HOLD",
        "TORSO_COMPENSATE",
        "SHIFT_LEFT",
        "HOLD_LEFT",
        "SHIFT_RIGHT",
        "HOLD_RIGHT",
        "RETURN_CENTER",
        "DONE",
    ]
    assert 0.0 < float(out["max_hip_roll_rad"]) <= 0.05
    assert 0.0 < float(out["max_ankle_roll_rad"]) <= 0.035
    assert float(out["max_left_foot_force_ratio"]) > 0.55
    assert float(out["max_right_foot_force_ratio"]) > 0.55
    assert out["physical_grasp_validation"] is False
    forbidden = {"STEP", "WALK", "LOCOMOTION"}
    assert not any(
        any(token in phase for token in forbidden) for phase in out["phase_sequence"]
    )


def test_g1_payload_weight_shift_script_output_includes_metrics():
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--headless",
            "--timeout",
            "10",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    stdout = proc.stdout
    for token in (
        "com_inside_support_all_phases:",
        "phases_completed:",
        "combined_robot_box_com:",
        "support_polygon_xy:",
        "min_com_margin_m:",
        "min_com_margin_left_shift_m:",
        "min_com_margin_right_shift_m:",
        "left_foot_force_n:",
        "right_foot_force_n:",
        "min_left_foot_force_n:",
        "min_right_foot_force_n:",
        "max_left_foot_force_ratio:",
        "max_right_foot_force_ratio:",
        "max_hip_roll_rad:",
        "max_ankle_roll_rad:",
        "payload_mass_kg:",
        "payload_moment_nm:",
        "torso_pitch_compensation_rad:",
        "no_locomotion_attempted:",
        "success:",
    ):
        assert token in stdout
