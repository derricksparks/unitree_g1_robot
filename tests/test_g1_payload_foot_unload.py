"""Smoke tests for G1 payload-aware single-foot unload."""

from __future__ import annotations

import importlib.util
import math
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_payload_foot_unload.py"
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
    spec = importlib.util.spec_from_file_location("run_g1_payload_foot_unload", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_payload_foot_unload")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_payload_foot_unload_headless_metrics():
    mod = _load_module()
    out = mod.run_g1_payload_foot_unload(headless=True, timeout=12.0, verbose=False)
    required = {
        "payload_foot_unload_validation_only",
        "no_locomotion_attempted",
        "phases_completed",
        "com_inside_support_all_phases",
        "min_com_margin_m",
        "min_right_foot_force_n",
        "min_left_foot_force_n",
        "min_right_foot_force_ratio",
        "min_left_foot_force_ratio",
        "right_foot_unload_success",
        "left_foot_unload_success",
        "max_support_foot_force_ratio",
        "max_hip_roll_rad",
        "max_ankle_roll_rad",
        "payload_mass_kg",
        "payload_moment_nm",
        "torso_pitch_compensation_rad",
        "success",
    }
    assert required.issubset(out)
    assert isinstance(out["success"], bool)
    assert math.isfinite(float(out["min_com_margin_m"]))
    assert out["no_locomotion_attempted"] is True
    assert out["success"] is True
    assert float(out["min_left_foot_force_n"]) > 20.0
    assert float(out["min_right_foot_force_n"]) > 20.0


def test_g1_payload_foot_unload_phase_contract():
    mod = _load_module()
    out = mod.run_g1_payload_foot_unload(headless=True, timeout=12.0, verbose=False)
    assert out["phase_sequence"] == [
        "STAND_NEUTRAL",
        "LOAD_HOLD",
        "TORSO_COMPENSATE",
        "SHIFT_TO_LEFT_SUPPORT",
        "UNLOAD_RIGHT_FOOT",
        "HOLD_RIGHT_UNLOADED",
        "RETURN_CENTER",
        "SHIFT_TO_RIGHT_SUPPORT",
        "UNLOAD_LEFT_FOOT",
        "HOLD_LEFT_UNLOADED",
        "RETURN_CENTER_FINAL",
        "DONE",
    ]
    assert out["right_foot_unload_success"] is True
    assert out["left_foot_unload_success"] is True
    assert float(out["max_hip_roll_rad"]) <= 0.10
    assert float(out["max_ankle_roll_rad"]) <= 0.08
    assert not any("STEP" in phase for phase in out["phase_sequence"])
    assert not any("WALK" in phase for phase in out["phase_sequence"])
    assert not any("LIFT" in phase for phase in out["phase_sequence"])


def test_g1_payload_foot_unload_script_output_includes_metrics():
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--headless",
            "--timeout",
            "12",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    stdout = proc.stdout
    for token in (
        "payload_foot_unload_validation_only:",
        "phases_completed:",
        "com_inside_support_all_phases:",
        "min_com_margin_m:",
        "min_right_foot_force_n:",
        "min_left_foot_force_n:",
        "min_right_foot_force_ratio:",
        "min_left_foot_force_ratio:",
        "right_foot_unload_success:",
        "left_foot_unload_success:",
        "max_support_foot_force_ratio:",
        "max_hip_roll_rad:",
        "max_ankle_roll_rad:",
        "payload_mass_kg:",
        "payload_moment_nm:",
        "torso_pitch_compensation_rad:",
        "no_locomotion_attempted:",
        "success:",
    ):
        assert token in stdout
