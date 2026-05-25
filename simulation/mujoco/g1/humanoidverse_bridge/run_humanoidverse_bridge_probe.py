#!/usr/bin/env python3
"""Probe HumanoidVerse G1 support and ONNX locomotion compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_G1 = Path(__file__).resolve().parents[1]
if str(_G1) not in sys.path:
    sys.path.insert(0, str(_G1))

from humanoidverse_bridge.humanoidverse_g1_locomotion import (
    HUMANOIDVERSE_G1_12DOF_JOINTS,
    HUMANOIDVERSE_G1_29DOF_JOINTS,
    humanoidverse_12dof_obs_dim,
)
from humanoidverse_bridge.humanoidverse_paths import (
    HUMANOIDVERSE_G1_12DOF_CONFIG,
    HUMANOIDVERSE_G1_29DOF_CONFIG,
    HUMANOIDVERSE_REPO_ROOT,
    humanoidverse_assets_available,
)
from lucky_bridge.lucky_policy_loader import LuckyPolicyLoadError, load_walker_policy


MANIPULATION_JOINTS: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _checkpoint_probe(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "checkpoint_provided": False,
            "checkpoint_valid": False,
            "message": "No ONNX path provided. Pass --policy-onnx to validate a checkpoint.",
        }
    try:
        policy = load_walker_policy(path)
    except (LuckyPolicyLoadError, Exception) as exc:
        return {
            "checkpoint_provided": True,
            "checkpoint_path": path,
            "checkpoint_valid": False,
            "error": str(exc),
        }
    return {
        "checkpoint_provided": True,
        "checkpoint_path": path,
        "checkpoint_valid": bool(policy.output_dim in (12, 29)),
        "onnx_input_dim": int(policy.input_dim),
        "onnx_output_dim": int(policy.output_dim),
        "expected_input_dim": int(humanoidverse_12dof_obs_dim()),
        "expected_output_dims": [12, 29],
        "expected_input_dim_for_output_12": 12 + 3 * 12,
        "expected_input_dim_for_output_29": 12 + 3 * 29,
        "compatible_with_bridge": bool(
            (policy.output_dim == 12 and policy.input_dim == (12 + 3 * 12))
            or (policy.output_dim == 29 and policy.input_dim == (12 + 3 * 29))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect HumanoidVerse G1 assets and checkpoint compatibility.")
    ap.add_argument("--policy-onnx", type=str, default=None, help="Path to exported HumanoidVerse ONNX locomotion policy.")
    args = ap.parse_args()

    locomotion_joints = list(HUMANOIDVERSE_G1_12DOF_JOINTS)
    locomotion_joints_with_waist = [jn for jn in HUMANOIDVERSE_G1_29DOF_JOINTS if "waist_" in jn or "hip_" in jn or "knee_" in jn or "ankle_" in jn]
    manipulation_joints = list(MANIPULATION_JOINTS)
    overlap = sorted(set(locomotion_joints).intersection(manipulation_joints))
    report: dict[str, Any] = {
        "humanoidverse_repo_root": str(HUMANOIDVERSE_REPO_ROOT),
        "humanoidverse_assets_available": bool(humanoidverse_assets_available()),
        "g1_12dof_config_exists": bool(HUMANOIDVERSE_G1_12DOF_CONFIG.is_file()),
        "g1_29dof_config_exists": bool(HUMANOIDVERSE_G1_29DOF_CONFIG.is_file()),
        "locomotion_joint_ownership": locomotion_joints,
        "locomotion_joint_ownership_with_waist": locomotion_joints_with_waist,
        "manipulation_joint_ownership": manipulation_joints,
        "joint_ownership_overlap": overlap,
        "arms_maskable_for_locomotion": bool(len(overlap) == 0),
    }
    report.update(_checkpoint_probe(args.policy_onnx))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

