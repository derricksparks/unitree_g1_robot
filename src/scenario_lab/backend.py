from __future__ import annotations

"""Factory-layer around mjlab play-time env + policy (headless).

Mirrors ``scripts/play.py`` without spawning a viewer; suitable for threaded rollouts.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import mjlab.tasks  # noqa: F401 — register mjlab builtins
import src.tasks  # noqa: F401 — register Unitree/custom tasks

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from mjlab.tasks.tracking.mdp import MotionCommandCfg

from src.scenario_lab.run_config import ScenarioRolloutConfig


RolloutHandles = tuple[RslRlVecEnvWrapper, Callable[..., torch.Tensor]]


def _policy_zero_factory(num_envs: int, num_actions: int, device: torch.device):
  class PolicyZero:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros((num_envs, num_actions), device=device)

  return PolicyZero()


def _policy_random_factory(num_envs: int, num_actions: int, device: torch.device):
  class PolicyRandom:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return 2 * torch.rand((num_envs, num_actions), device=device) - 1

  return PolicyRandom()


def build_vec_env_policy(cfg: ScenarioRolloutConfig) -> RolloutHandles:
  """Construct wrapped vec-env and a ``policy(obs) -> actions`` callable."""
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch_device = torch.device(device)

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)

  env_cfg.scene.num_envs = cfg.num_envs

  # Tracking tasks optional motion overrides (parity with scripts/play.py).
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  dummy_mode = cfg.agent != "trained"
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if cfg.motion_file is not None and cfg.motion_file.exists():
      motion_cmd.motion_file = str(cfg.motion_file)
    elif dummy_mode:
      raise ValueError(
        "Mjlab tracking tasks require --motion-file (dummy agents) or omit for trained."
      )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if dummy_mode:
    num_envs = env_wrapped.num_envs
    num_actions = env_wrapped.num_actions
    if cfg.agent == "zero":
      policy_callable = _policy_zero_factory(num_envs, num_actions, torch_device)
    else:
      policy_callable = _policy_random_factory(num_envs, num_actions, torch_device)
    return env_wrapped, policy_callable

  assert cfg.checkpoint_file is not None
  resume_path = cfg.checkpoint_file
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env_wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy_callable = runner.get_inference_policy(device=device)
  return env_wrapped, policy_callable
