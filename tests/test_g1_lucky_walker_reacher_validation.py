from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "simulation" / "mujoco" / "g1"
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))


def test_lucky_reacher_policy_loads_and_dims() -> None:
    from lucky_bridge.lucky_paths import LUCKY_RIGHT_REACHER_ONNX, LUCKY_WALKER_ONNX
    from lucky_bridge.lucky_policy_loader import load_right_reacher_policy, load_walker_policy

    walker = load_walker_policy(LUCKY_WALKER_ONNX)
    reacher = load_right_reacher_policy(LUCKY_RIGHT_REACHER_ONNX)

    assert walker.input_dim == 99
    assert walker.output_dim == 29
    assert reacher.input_dim == 36
    assert reacher.output_dim == 7


def test_lucky_walker_reacher_validation_smoke() -> None:
    from run_g1_lucky_walker_reacher_validation import run_g1_lucky_walker_reacher_validation

    result = run_g1_lucky_walker_reacher_validation(
        headless=True,
        timeout=0.08,
        cmd_x=0.0,
        verbose=False,
    )

    assert result["walker_policy_loaded"] is True
    assert result["right_reacher_policy_loaded"] is True
    assert result["walker_controls_legs_waist_only"] is True
    assert result["reacher_controls_right_arm_only"] is True
    assert result["walker_arm_outputs_masked"] is True
    assert result["right_arm_reacher_active"] is True
    assert result["hand_control_separate"] is True
    assert "robot_fell" in result
    assert "walking_preserved" in result
    assert "right_arm_motion_detected" in result
    assert "success" in result
    if result["success"] is False:
        assert result["failure_reason"]
