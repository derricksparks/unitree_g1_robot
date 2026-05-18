"""Smoke tests for G1 payload standing-balance validation."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_payload_balance.py"
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
    spec = importlib.util.spec_from_file_location("run_g1_payload_balance", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_payload_balance")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_payload_balance_headless_metrics():
    mod = _load_module()
    out = mod.run_g1_payload_balance(headless=True, timeout=8.0, verbose=False)
    required = {
        "robot_com",
        "box_com",
        "combined_robot_box_com",
        "left_foot_force_n",
        "right_foot_force_n",
        "support_polygon_xy",
        "com_inside_support",
        "min_com_margin_m",
        "forward_pitch_moment_from_payload_nm",
        "suggested_torso_pitch_compensation_rad",
        "max_torso_pitch_rad",
        "foot_force_balance_ratio",
        "payload_mass_kg",
        "payload_moment_nm",
    }
    assert required.issubset(out)
    assert isinstance(out["com_inside_support"], bool)
    assert out["payload_balance_validation_only"] is True
    assert out["physical_grasp_validation"] is False


def test_g1_payload_balance_attempts_no_locomotion():
    mod = _load_module()
    out = mod.run_g1_payload_balance(headless=True, timeout=8.0, verbose=False)
    assert out["no_locomotion_attempted"] is True
    assert float(out["leg_joint_max_delta_from_neutral_rad"]) < 1e-9
    assert out["phase_sequence"] == [
        "STAND_NEUTRAL",
        "LOAD_HOLD",
        "TORSO_COMPENSATE",
        "HOLD_COMPENSATED",
        "DONE",
    ]


def test_g1_payload_balance_script_output_includes_metrics():
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(SCRIPT),
            "--headless",
            "--timeout",
            "8",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    stdout = proc.stdout
    for token in (
        "robot_com:",
        "box_com:",
        "combined_robot_box_com:",
        "left_foot_force_n:",
        "right_foot_force_n:",
        "support_polygon_xy:",
        "com_inside_support:",
        "min_com_margin_m:",
        "payload_mass_kg:",
        "payload_moment_nm:",
        "no_locomotion_attempted:",
    ):
        assert token in stdout
