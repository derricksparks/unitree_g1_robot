"""Smoke test: G1 locked-base right-arm task-space IK demo."""

from __future__ import annotations

import math
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
IK_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_right_arm_ik_demo.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def _load_ik_demo_module():
    spec = importlib.util.spec_from_file_location("run_g1_right_arm_ik_demo", IK_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_right_arm_ik_demo")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_right_arm_ik_demo_headless_smoke():
    mod = _load_ik_demo_module()
    out = mod.run_right_arm_ik_demo(headless=True, timeout=8.0, verbose=False)
    assert isinstance(out, dict)
    assert out["model_path"].endswith("g1_position_actuated.xml")
    assert abs(out["sim_time"] - 8.0) < 0.08
    assert out["fix_base_in_world"] is True

    max_err = out["max_position_error_norm"]
    assert math.isfinite(max_err)

    assert out["max_wrist_delta_x_from_stow"] >= 0.03
    assert out["max_ik_joint_command_magnitude"] <= 0.95
