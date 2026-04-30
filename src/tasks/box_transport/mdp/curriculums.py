from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .commands import BoxTransportCommand, BoxTransportCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StageSchedule(TypedDict):
  step: int
  stage: int


def box_transport_stage(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[StageSchedule],
) -> torch.Tensor:
  del env_ids
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, BoxTransportCommand)
  cfg = cast(BoxTransportCommandCfg, command.cfg)
  selected_stage = cfg.initial_stage
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      selected_stage = stage["stage"]
  command.set_stage(selected_stage)
  return torch.tensor([selected_stage], device=env.device, dtype=torch.float32)
