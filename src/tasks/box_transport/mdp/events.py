from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import BoxTransportCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_box_to_command(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
) -> None:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, BoxTransportCommand)
  box = env.scene["box"]

  root_state = torch.zeros((len(env_ids), 13), device=env.device)
  root_state[:, :3] = command.box_start_pos_w[env_ids]
  root_state[:, 3] = 1.0
  box.write_root_state_to_sim(root_state, env_ids)
