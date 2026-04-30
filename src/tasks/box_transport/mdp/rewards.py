from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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


def _body_pos_w(env: ManagerBasedRlEnv, body_name: str) -> torch.Tensor:
  robot = _robot(env)
  body_ids = robot.find_bodies((body_name,), preserve_order=True)[0]
  return robot.data.body_link_pos_w[:, body_ids[0], :]


def _stage_mask(env: ManagerBasedRlEnv, command_name: str, min_stage: int) -> torch.Tensor:
  return (_command(env, command_name).stage >= min_stage).float()


def reach_box(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.25,
  left_body_name: str = "left_wrist_yaw_link",
  right_body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
  del command_name
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  left_dist = torch.norm(_body_pos_w(env, left_body_name) - box_pos, dim=-1)
  right_dist = torch.norm(_body_pos_w(env, right_body_name) - box_pos, dim=-1)
  return torch.exp(-0.5 * (left_dist + right_dist) / std)


def lift_box(
  env: ManagerBasedRlEnv,
  command_name: str,
  lift_height: float = 0.12,
  std: float = 0.12,
) -> torch.Tensor:
  cmd = _command(env, command_name)
  target_z = cmd.box_start_pos_w[:, 2] + lift_height
  error = torch.square(_box(env).data.root_link_pos_w[:, 2] - target_z)
  return torch.exp(-error / std**2) * _stage_mask(env, command_name, 1)


def carry_box_to_shelf(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.45,
) -> torch.Tensor:
  cmd = _command(env, command_name)
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  error = torch.sum(torch.square(box_pos - cmd.shelf_pos_w), dim=-1)
  return torch.exp(-error / std**2) * _stage_mask(env, command_name, 2)


def place_box_on_shelf(
  env: ManagerBasedRlEnv,
  command_name: str,
  xy_std: float = 0.18,
  z_std: float = 0.08,
) -> torch.Tensor:
  cmd = _command(env, command_name)
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  xy_error = torch.sum(torch.square(box_pos[:, :2] - cmd.shelf_pos_w[:, :2]), dim=-1)
  z_error = torch.square(box_pos[:, 2] - cmd.shelf_pos_w[:, 2])
  return (
    torch.exp(-xy_error / xy_std**2 - z_error / z_std**2)
    * _stage_mask(env, command_name, 3)
  )


def box_upright(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Reward low angular velocity; for a small cuboid this is a practical carry-stability proxy.
  ang_vel = _box(env).data.root_link_ang_vel_w[:, :3]
  return torch.exp(-torch.sum(torch.square(ang_vel), dim=-1) / 4.0)


def keep_box_in_hands(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_dist: float = 0.28,
  left_body_name: str = "left_wrist_yaw_link",
  right_body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
  box_pos = _box(env).data.root_link_pos_w[:, :3]
  left_dist = torch.norm(_body_pos_w(env, left_body_name) - box_pos, dim=-1)
  right_dist = torch.norm(_body_pos_w(env, right_body_name) - box_pos, dim=-1)
  held = ((left_dist < max_dist) & (right_dist < max_dist)).float()
  return held * _stage_mask(env, command_name, 1)


def foot_contact_balance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = sensor.data.found.squeeze(-1).float()
  # Prefer both feet down during reach/lift/carry; this is conservative for sim-to-real.
  return (contacts.mean(dim=-1) > 0.75).float()


def object_drop_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  drop_margin: float = 0.04,
) -> torch.Tensor:
  cmd = _command(env, command_name)
  dropped = _box(env).data.root_link_pos_w[:, 2] < cmd.box_start_pos_w[:, 2] - drop_margin
  return dropped.float() * _stage_mask(env, command_name, 1)


def base_distance_to_box(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_distance: float = 0.45,
  std: float = 0.35,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  del command_name
  robot: Entity = env.scene[asset_cfg.name]
  box_pos = _box(env).data.root_link_pos_w[:, :2]
  base_pos = robot.data.root_link_pos_w[:, :2]
  error = torch.square(torch.norm(box_pos - base_pos, dim=-1) - target_distance)
  return torch.exp(-error / std**2)
