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


def _body_pos_w(env: ManagerBasedRlEnv, body_name: str) -> torch.Tensor:
  robot = _robot(env)
  body_ids = robot.find_bodies((body_name,), preserve_order=True)[0]
  return robot.data.body_link_pos_w[:, body_ids[0], :]


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
  left_body_name: str = "left_wrist_yaw_link",
  right_body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  left_rel = box_pos - _body_pos_w(env, left_body_name)
  right_rel = box_pos - _body_pos_w(env, right_body_name)
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
