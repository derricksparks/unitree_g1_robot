from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import BoxTransportCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _box(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["box"]


def _command(env: ManagerBasedRlEnv, command_name: str) -> BoxTransportCommand:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, BoxTransportCommand)
  return command


def _to_base_frame(env: ManagerBasedRlEnv, vec_w: torch.Tensor) -> torch.Tensor:
  robot = _robot(env)
  return quat_apply_inverse(robot.data.root_link_quat_w, vec_w)


def _palm_positions_w(env: ManagerBasedRlEnv, left_site: str, right_site: str) -> torch.Tensor:
  """Palm-frame positions (G1 rigid hand pads; no articulated fingers).

  Sites ``left_palm`` / ``right_palm`` sit on the distal collision geometry (~x
  axis on wrist_yaw_link), closer to pinch contacts than the link origin.
  """
  robot = _robot(env)
  site_ids, _ = robot.find_sites((left_site, right_site), preserve_order=True)
  palms = robot.data.site_pos_w[:, site_ids, :]
  assert palms.shape[1] == 2
  return palms


def box_position_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box position relative to the robot base frame."""
  rel_w = _box(env).data.root_link_pos_w[:, :3] - _robot(env).data.root_link_pos_w[:, :3]
  return _to_base_frame(env, rel_w)


def shelf_position_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Shelf target position relative to the robot base frame."""
  cmd = _command(env, command_name)
  rel_w = cmd.shelf_pos_w - _robot(env).data.root_link_pos_w[:, :3]
  return _to_base_frame(env, rel_w)


def box_to_shelf_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = _command(env, command_name)
  return _to_base_frame(env, cmd.shelf_pos_w - _box(env).data.root_link_pos_w[:, :3])


def hand_to_box_b(
  env: ManagerBasedRlEnv,
  left_site: str = "left_palm",
  right_site: str = "right_palm",
) -> torch.Tensor:
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  palms = _palm_positions_w(env, left_site, right_site)
  left_rel = box_pos - palms[:, 0, :]
  right_rel = box_pos - palms[:, 1, :]
  return torch.cat([_to_base_frame(env, left_rel), _to_base_frame(env, right_rel)], dim=-1)


def box_velocity_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  box = _box(env)
  return _to_base_frame(env, box.data.root_link_lin_vel_w[:, :3])


def box_lifted(env: ManagerBasedRlEnv, command_name: str, lift_margin: float = 0.08) -> torch.Tensor:
  cmd = _command(env, command_name)
  lifted = _box(env).data.root_link_pos_w[:, 2] > cmd.box_start_pos_w[:, 2] + lift_margin
  return lifted.float().unsqueeze(-1)


def task_stage(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = _command(env, command_name)
  return (cmd.stage.float() / max(cmd.cfg.max_stage, 1)).unsqueeze(-1)
