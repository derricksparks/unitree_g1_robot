"""Centralized paths for the external Lucky Robots G1 challenge repo."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LUCKY_REPO_ROOT = REPO_ROOT / "external" / "g1-manipulation-challenge"

LUCKY_MODEL_CONFIG = LUCKY_REPO_ROOT / "model_config.json"
LUCKY_SCENE_XML = LUCKY_REPO_ROOT / "scene.xml"
LUCKY_G1_XML = LUCKY_REPO_ROOT / "g1.xml"
LUCKY_WALKER_ONNX = LUCKY_REPO_ROOT / "walker.onnx"
LUCKY_RIGHT_REACHER_ONNX = LUCKY_REPO_ROOT / "right_reacher.onnx"
LUCKY_CROUCHER_ONNX = LUCKY_REPO_ROOT / "croucher.onnx"


def lucky_assets_available() -> bool:
    return bool(
        LUCKY_REPO_ROOT.is_dir()
        and LUCKY_MODEL_CONFIG.is_file()
        and LUCKY_SCENE_XML.is_file()
        and LUCKY_WALKER_ONNX.is_file()
    )


def missing_lucky_assets() -> list[str]:
    required = (
        LUCKY_REPO_ROOT,
        LUCKY_MODEL_CONFIG,
        LUCKY_SCENE_XML,
        LUCKY_WALKER_ONNX,
        LUCKY_RIGHT_REACHER_ONNX,
    )
    return [str(path) for path in required if not path.exists()]
