"""G1 flat velocity scene with box-transport furniture (play / sim2sim preview)."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.objects import (
  get_transport_box_cfg,
  get_transport_shelf_cfg,
  get_transport_table_cfg,
)
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def unitree_g1_flat_transport_scene_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat velocity task with table, shelf, and box (same actor obs as Unitree-G1-Flat)."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.scene.entities.update(
    {
      "table": get_transport_table_cfg(),
      "shelf": get_transport_shelf_cfg(),
      "box": get_transport_box_cfg(),
    }
  )
  cfg.scene.extent = 2.5
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 600

  return cfg
