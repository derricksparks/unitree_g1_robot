"""Smoke test: locked-base grasp prototype (assist + lift)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_POS_XML = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "g1"
    / "assets"
    / "g1_position_actuated.xml"
)
G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
G1_REACH_SCENE = G1_DIR / "g1_reach_box_scene.xml"
GRASP_BOX_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "run_g1_grasp_box.py"

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def test_g1_grasp_proxy_geoms_and_bodies_in_scene():
    import os

    mujoco = pytest.importorskip("mujoco")

    if not G1_REACH_SCENE.is_file():
        pytest.skip(f"missing {G1_REACH_SCENE}")

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene.xml")
    finally:
        os.chdir(prev)

    for gname in (
        "right_proxy_left_finger_geom",
        "right_proxy_right_finger_geom",
        "right_palm_contact_geom",
    ):
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname) >= 0
        ), f"missing geom {gname}"

    for jn in ("right_proxy_left_finger_slide", "right_proxy_right_finger_slide"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0

    for aname in ("right_proxy_left_finger_motor", "right_proxy_right_finger_motor"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) >= 0
    for bname in ("right_proxy_left_finger", "right_proxy_right_finger"):
        assert (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname) >= 0
        ), f"missing body {bname}"


def _load_grasp_box_module():
    spec = importlib.util.spec_from_file_location("run_g1_grasp_box", GRASP_BOX_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_grasp_box")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_grasp_box_headless():
    mod = _load_grasp_box_module()
    out = mod.run_g1_grasp_box(headless=True, timeout=10.0, verbose=False)
    assert isinstance(out, dict)
    assert out["initial_contact_before_approach"] is False
    assert out["max_penetration_depth"] <= out["max_penetration_depth_ok_m"]
    assert out["pregrasp_success"] is True
    assert out["grasp_assist_used"] is True
    assert out["box_height_delta"] >= 0.04
    assert out["lift_success"] is True
    assert out["grasp_contact_ready"] is True
    assert out["proxy_finger_actuators_present"] is True
    assert out["proxy_finger_closed"] is True
    assert out["palm_contact_count_max"] >= 0
    assert out["left_finger_contact_count_max"] >= 0
    assert out["right_finger_contact_count_max"] >= 0
