from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import BoxTransportCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def box_dropped(
  env: ManagerBasedRlEnv,
  command_name: str,
  margin: float = 0.08,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, BoxTransportCommand)
  box = env.scene["box"]
  active = command.stage >= 1
  return active & (box.data.root_link_pos_w[:, 2] < command.box_start_pos_w[:, 2] - margin)


def box_too_far(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_distance: float = 2.0,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, BoxTransportCommand)
  box = env.scene["box"]
  return torch.norm(box.data.root_link_pos_w[:, :2] - command.box_start_pos_w[:, :2], dim=-1) > max_distance


def bad_torso_orientation(
  env: ManagerBasedRlEnv,
  threshold: float = 0.75,
) -> torch.Tensor:
  robot = env.scene["robot"]
  xy_gravity = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=-1)
  return xy_gravity > threshold
