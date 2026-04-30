from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BoxTransportCommand(CommandTerm):
  """Sample table, box, and shelf targets for the transport task."""

  cfg: BoxTransportCommandCfg

  def __init__(self, cfg: BoxTransportCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.table_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.box_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.shelf_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.metrics["box_to_shelf"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [self.box_start_pos_w, self.shelf_pos_w, self.stage.float().unsqueeze(-1)], dim=-1
    )

  def _update_metrics(self) -> None:
    try:
      box = self._env.scene["box"]
    except KeyError:
      return
    self.metrics["box_to_shelf"] = torch.norm(
      box.data.root_link_pos_w[:, :3] - self.shelf_pos_w, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    origins = self._env.scene.env_origins[env_ids]
    r = torch.empty(len(env_ids), device=self.device)

    self.table_pos_w[env_ids, 0] = origins[:, 0] + r.uniform_(*self.cfg.table_x_range)
    self.table_pos_w[env_ids, 1] = origins[:, 1] + r.uniform_(*self.cfg.table_y_range)
    self.table_pos_w[env_ids, 2] = self.cfg.table_height

    self.box_start_pos_w[env_ids, 0] = self.table_pos_w[env_ids, 0] + r.uniform_(
      *self.cfg.box_xy_jitter
    )
    self.box_start_pos_w[env_ids, 1] = self.table_pos_w[env_ids, 1] + r.uniform_(
      *self.cfg.box_xy_jitter
    )
    self.box_start_pos_w[env_ids, 2] = self.cfg.table_height + self.cfg.box_half_extents[2]

    self.shelf_pos_w[env_ids, 0] = origins[:, 0] + r.uniform_(*self.cfg.shelf_x_range)
    self.shelf_pos_w[env_ids, 1] = origins[:, 1] + r.uniform_(*self.cfg.shelf_y_range)
    self.shelf_pos_w[env_ids, 2] = self.cfg.shelf_height

    self.stage[env_ids] = min(self.cfg.initial_stage, self.cfg.max_stage)

    # Box pose must be written here: env reset events run *before* command_manager.reset,
    # so a separate reset_box event would use stale box_start_pos_w (often zeros).
    box = self._env.scene["box"]
    root_state = torch.zeros((len(env_ids), 13), device=self.device)
    root_state[:, :3] = self.box_start_pos_w[env_ids]
    root_state[:, 3] = 1.0  # quat w = 1, body upright
    box.write_root_state_to_sim(root_state, env_ids)

  def _update_command(self) -> None:
    pass

  def set_stage(self, stage: int) -> None:
    self.stage[:] = min(max(stage, 0), self.cfg.max_stage)

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx,
  ) -> None:
    del get_env_idx
    with server.gui.add_folder(name.capitalize()):
      stage = server.gui.add_slider(
        "Curriculum stage",
        min=0,
        max=self.cfg.max_stage,
        step=1,
        initial_value=self.cfg.initial_stage,
      )

      @stage.on_update
      def _(_ev) -> None:
        self.set_stage(int(stage.value))

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    # DebugVisualizer API differs between mjlab versions; command values are still
    # available through the GUI and logs if marker helpers are unavailable.
    del visualizer


@dataclass(kw_only=True)
class BoxTransportCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = True
  initial_stage: int = 0
  max_stage: int = 4

  table_x_range: tuple[float, float] = (0.55, 0.75)
  table_y_range: tuple[float, float] = (-0.10, 0.10)
  table_height: float = 0.72

  shelf_x_range: tuple[float, float] = (1.25, 1.55)
  shelf_y_range: tuple[float, float] = (-0.25, 0.25)
  shelf_height: float = 0.92

  box_half_extents: tuple[float, float, float] = (0.10, 0.075, 0.06)
  box_xy_jitter: tuple[float, float] = (-0.03, 0.03)

  def build(self, env: ManagerBasedRlEnv) -> BoxTransportCommand:
    return BoxTransportCommand(self, env)
