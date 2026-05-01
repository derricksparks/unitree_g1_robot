from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import BoxTransportCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg


class box_dropped_after_lift:
  """Terminate on box drop only after a clear lift in the same episode.

  Avoids resetting on small table bounces or pre-lift bumps while stage≥1 shaping
  is active—common failure mode when the box is unconstrained floating.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg  # Passed by mjlab resolver; unused here.
    self._lifted_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(
    self,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if env_ids is None:
      self._lifted_episode.zero_()
    else:
      self._lifted_episode[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    margin: float = 0.14,
    lift_above_spawn: float = 0.07,
    min_stage: int = 1,
  ) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert isinstance(command, BoxTransportCommand)
    box = env.scene["box"]
    spawn_z = command.box_start_pos_w[:, 2]
    box_z = box.data.root_link_pos_w[:, 2]

    stage_ok = command.stage >= min_stage
    lifted_now = box_z > (spawn_z + lift_above_spawn)

    self._lifted_episode |= stage_ok & lifted_now

    fell_below = box_z < (spawn_z - margin)
    return stage_ok & self._lifted_episode & fell_below


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
