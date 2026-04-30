"""MJCF assets for the box transport task."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

from src import SRC_PATH

ASSET_DIR = SRC_PATH / "assets" / "objects"

TABLE_XML: Path = ASSET_DIR / "transport_table.xml"
SHELF_XML: Path = ASSET_DIR / "transport_shelf.xml"
BOX_XML: Path = ASSET_DIR / "transport_box.xml"


def _load_spec(path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(path))
  spec.assets = {}
  return spec


def get_table_spec() -> mujoco.MjSpec:
  return _load_spec(TABLE_XML)


def get_shelf_spec() -> mujoco.MjSpec:
  return _load_spec(SHELF_XML)


def get_box_spec() -> mujoco.MjSpec:
  return _load_spec(BOX_XML)


def get_transport_table_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_table_spec)


def get_transport_shelf_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_shelf_spec)


def get_transport_box_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.65, 0.0, 0.78),
    ),
    spec_fn=get_box_spec,
  )
