from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AgentMode = Literal["trained", "zero", "random"]


@dataclass(frozen=True)
class ScenarioRolloutConfig:
  """Serializable parameters for batch rollouts inside mjlab only (simulation)."""

  task_id: str = "Unitree-G1-Box-Transport"
  checkpoint_file: Path | None = None
  motion_file: Path | None = None

  agent: AgentMode = "trained"
  num_envs: int = 1
  device: str | None = None
  max_steps: int = 250_000
  """Safety cap across all environments (globally aggregated steps)."""

  episode_target: int = 10
  """Stop after recording this many completed episodes on env index 0."""

  log_csv_path: Path | None = None
  emit_status_every_steps: int = 500


def validate_for_run(cfg: ScenarioRolloutConfig) -> None:
  if cfg.num_envs < 1:
    raise ValueError("num_envs must be >= 1.")
  if cfg.episode_target < 1:
    raise ValueError("episode_target must be >= 1.")
  if cfg.max_steps < 1:
    raise ValueError("max_steps must be >= 1.")
  if cfg.agent == "trained":
    if cfg.checkpoint_file is None:
      raise ValueError("checkpoint_file is required when agent mode is trained.")
    if not cfg.checkpoint_file.exists():
      raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_file}")
