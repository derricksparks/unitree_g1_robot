"""Smoke tests for G1 payload-aware quasi-static first step."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_payload_first_step.py"
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
    spec = importlib.util.spec_from_file_location("run_g1_payload_first_step", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_payload_first_step")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_payload_first_step_headless_metrics():
    mod = _load_module()
    out = mod.run_g1_payload_first_step(headless=True, timeout=18.0, verbose=False)
    required = {
        "payload_first_step_validation_only",
        "no_continuous_walking_attempted",
        "no_rl_used",
        "phases_completed",
        "right_step_attempted",
        "right_step_success",
        "left_step_attempted",
        "left_step_success",
        "left_step_skipped_reason",
        "right_step_length_m",
        "left_step_length_m",
        "max_right_foot_clearance_m",
        "max_left_foot_clearance_m",
        "max_swing_foot_lateral_drift_m",
        "right_foot_landed",
        "left_foot_landed",
        "com_inside_support_all_phases",
        "min_com_margin_m",
        "min_com_margin_single_support_m",
        "min_com_margin_double_support_after_landing_m",
        "min_support_foot_force_n",
        "max_pelvis_lateral_shift_m",
        "payload_mass_kg",
        "payload_moment_nm",
        "torso_pitch_compensation_rad",
        "success",
    }
    assert required.issubset(out)
    assert isinstance(out["success"], bool)
    assert out["no_continuous_walking_attempted"] is True
    assert out["no_rl_used"] is True
    assert math.isfinite(float(out["right_step_length_m"]))
    assert float(out["right_step_length_m"]) >= 0.015
    assert float(out["right_step_length_m"]) <= 0.025
    assert float(out["max_swing_foot_lateral_drift_m"]) < 0.006
    assert out["right_step_success"] is True or out["right_step_failure_reason"]


def test_g1_payload_first_step_phase_contract():
    mod = _load_module()
    out = mod.run_g1_payload_first_step(headless=True, timeout=18.0, verbose=False)
    assert out["phase_sequence"] == [
        "STAND_NEUTRAL",
        "LOAD_HOLD",
        "TORSO_COMPENSATE",
        "SHIFT_TO_LEFT_SUPPORT",
        "UNLOAD_RIGHT_FOOT",
        "LIFT_RIGHT_FOOT_SMALL",
        "MOVE_RIGHT_FOOT_FORWARD_SMALL",
        "LOWER_RIGHT_FOOT",
        "RIGHT_STEP_DOUBLE_SUPPORT_HOLD",
        "RETURN_CENTER",
        "SHIFT_TO_RIGHT_SUPPORT",
        "UNLOAD_LEFT_FOOT",
        "LIFT_LEFT_FOOT_SMALL",
        "MOVE_LEFT_FOOT_FORWARD_SMALL",
        "LOWER_LEFT_FOOT",
        "LEFT_STEP_DOUBLE_SUPPORT_HOLD",
        "RETURN_CENTER_FINAL",
        "DONE",
    ]
    assert not any("WALK" in phase for phase in out["phase_sequence"])
    assert not any("CONTINUOUS" in phase for phase in out["phase_sequence"])
    assert "RL" not in " ".join(out["phase_sequence"])
    assert out["right_foot_landed"] is True
    assert float(out["min_support_foot_force_n"]) > 180.0
    assert float(out["min_com_margin_single_support_m"]) > 0.005
    assert float(out["min_com_margin_double_support_after_landing_m"]) > 0.015


def test_g1_payload_first_step_script_output_includes_metrics():
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--headless",
            "--timeout",
            "18",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    stdout = proc.stdout
    for token in (
        "payload_first_step_validation_only:",
        "no_continuous_walking_attempted:",
        "no_rl_used:",
        "right_step_success:",
        "left_step_attempted:",
        "left_step_skipped_reason:",
        "right_step_length_m:",
        "max_right_foot_clearance_m:",
        "max_swing_foot_lateral_drift_m:",
        "right_foot_landed:",
        "min_com_margin_single_support_m:",
        "min_com_margin_double_support_after_landing_m:",
        "min_support_foot_force_n:",
        "success:",
    ):
        assert token in stdout
