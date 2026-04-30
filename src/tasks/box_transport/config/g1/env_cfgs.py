"""Unitree G1 box transport environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.box_transport.box_transport_env_cfg import make_g1_box_transport_env_cfg


def unitree_g1_box_transport_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 whole-body box transport configuration."""
  return make_g1_box_transport_env_cfg(play=play)
