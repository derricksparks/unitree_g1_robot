"""Centralized paths for local HumanoidVerse assets."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
HUMANOIDVERSE_REPO_ROOT = REPO_ROOT / "external" / "HumanoidVerse"
HUMANOIDVERSE_G1_12DOF_CONFIG = HUMANOIDVERSE_REPO_ROOT / "humanoidverse" / "config" / "robot" / "g1" / "g1_12dof.yaml"
HUMANOIDVERSE_G1_29DOF_CONFIG = HUMANOIDVERSE_REPO_ROOT / "humanoidverse" / "config" / "robot" / "g1" / "g1_29dof.yaml"


def humanoidverse_assets_available() -> bool:
    return bool(
        HUMANOIDVERSE_REPO_ROOT.is_dir()
        and HUMANOIDVERSE_G1_12DOF_CONFIG.is_file()
        and HUMANOIDVERSE_G1_29DOF_CONFIG.is_file()
    )

