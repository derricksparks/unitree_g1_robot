from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "simulation" / "mujoco" / "g1"
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))


def test_lucky_bridge_paths_and_policy_files() -> None:
    from lucky_bridge.lucky_paths import (
        LUCKY_MODEL_CONFIG,
        LUCKY_REPO_ROOT,
        LUCKY_RIGHT_REACHER_ONNX,
        LUCKY_SCENE_XML,
        LUCKY_WALKER_ONNX,
        missing_lucky_assets,
    )
    from lucky_bridge.lucky_policy_loader import load_lucky_config

    assert LUCKY_REPO_ROOT.exists(), (
        "Lucky repo missing. Clone it with: "
        "git clone https://github.com/luckyrobots/g1-manipulation-challenge "
        "external/g1-manipulation-challenge"
    )
    assert not missing_lucky_assets()
    assert LUCKY_MODEL_CONFIG.is_file()
    assert LUCKY_SCENE_XML.is_file()
    assert LUCKY_WALKER_ONNX.is_file()
    assert LUCKY_RIGHT_REACHER_ONNX.is_file()

    cfg = load_lucky_config(LUCKY_MODEL_CONFIG)
    assert cfg["walker"]["input_dim"] == 99
    assert cfg["walker"]["output_dim"] == 29
    assert len(cfg["joint_names"]) == 29


def test_lucky_joint_map_masks_arms() -> None:
    import mujoco

    from lucky_bridge.lucky_joint_map import build_lucky_joint_map
    from lucky_bridge.lucky_paths import LUCKY_MODEL_CONFIG, LUCKY_SCENE_XML
    from lucky_bridge.lucky_policy_loader import load_lucky_config

    cfg = load_lucky_config(LUCKY_MODEL_CONFIG)
    model = mujoco.MjModel.from_xml_path(str(LUCKY_SCENE_XML))
    joint_map = build_lucky_joint_map(model, cfg)

    assert joint_map.action_dim == 29
    assert joint_map.mapping_ok
    assert len(joint_map.leg_indices) == 12
    assert len(joint_map.waist_indices) == 3
    assert len(joint_map.arm_indices) == 14
    assert len(joint_map.controlled_indices) == 15
    assert all("shoulder" not in joint_map.joint_names[i] for i in joint_map.controlled_indices)
    assert all("elbow" not in joint_map.joint_names[i] for i in joint_map.controlled_indices)
    assert all("wrist" not in joint_map.joint_names[i] for i in joint_map.controlled_indices)


def test_lucky_locomotion_validation_smoke() -> None:
    from run_g1_lucky_locomotion_validation import run_g1_lucky_locomotion_validation

    result = run_g1_lucky_locomotion_validation(
        headless=True,
        timeout=0.05,
        cmd_x=0.0,
        verbose=False,
    )

    assert result["lucky_locomotion_enabled"] is True
    assert result["policy_loaded"] is True
    assert result["walker_policy_path"]
    assert result["observation_dim"] == 99
    assert result["action_dim"] == 29
    assert "joint_mapping_ok" in result
    assert "success" in result
    if result["success"] is False:
        assert result["failure_reason"]
