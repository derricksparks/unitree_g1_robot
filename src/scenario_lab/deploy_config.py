from __future__ import annotations

"""Helpers for swapping ``deploy/robots/g1`` FSM presets before launching ``g1_ctrl``."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

DUAL_PRESET_FILENAME = "config.dual_velocity_box_transport.yaml"


def default_g1_ctrl_binary(repo_root: Path) -> Path:
  return repo_root / "deploy" / "robots" / "g1" / "build" / "g1_ctrl"


def g1_deploy_robot_root(repo_root: Path) -> Path:
  return repo_root / "deploy" / "robots" / "g1"


def g1_active_config(repo_root: Path) -> Path:
  return g1_deploy_robot_root(repo_root) / "config" / "config.yaml"


def g1_dual_preset_path(repo_root: Path) -> Path:
  return g1_deploy_robot_root(repo_root) / "config" / DUAL_PRESET_FILENAME


@dataclass(frozen=True)
class BackupResult:
  backup_path: Path | None
  applied_from: Path


def backup_then_apply_preset(repo_root: Path, preset_yaml: Path) -> BackupResult:
  """Copies preset over ``config/config.yaml``, backing up any existing file."""
  robot = g1_deploy_robot_root(repo_root)
  config_dir = robot / "config"
  target = config_dir / "config.yaml"
  if not preset_yaml.is_file():
    raise FileNotFoundError(f"Preset not found: {preset_yaml}")

  bak: Path | None = None
  if target.is_file():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = config_dir / f"config.yaml.bak.{ts}"
    shutil.copy2(target, bak)

  shutil.copy2(preset_yaml, target)
  return BackupResult(backup_path=bak, applied_from=preset_yaml)
