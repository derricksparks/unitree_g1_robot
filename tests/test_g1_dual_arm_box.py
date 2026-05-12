"""Smoke test: locked-base dual-arm box coordination (IK + assist + lift)."""

from __future__ import annotations

import os
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
DUAL_ARM_SCRIPT = G1_DIR / "run_g1_dual_arm_box.py"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.g1_box_perception import G1BoxPerception

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)
def test_g1_dual_arm_left_palm_and_proxy_geoms():
    mujoco = pytest.importorskip("mujoco")

    if not G1_REACH_SCENE.is_file():
        pytest.skip(f"missing {G1_REACH_SCENE}")

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene.xml")
    finally:
        os.chdir(prev)

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm_site") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site") >= 0
    for sname in (
        "box_near_face_center_site",
        "box_right_contact_site",
        "box_left_contact_site",
        "box_right_lower_edge_site",
        "box_left_lower_edge_site",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname) >= 0, sname
    for gname in (
        "left_palm_contact_geom",
        "left_proxy_left_finger_geom",
        "left_proxy_right_finger_geom",
    ):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname) >= 0, gname


def _load_dual_arm_module():
    spec = importlib.util.spec_from_file_location("run_g1_dual_arm_box", DUAL_ARM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_dual_arm_box")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_g1_dual_arm_box_headless():
    """
    Full dual-arm demo: strict palm–target squeezes, gated post-contact timeline, lift.

    Proximity fields are diagnostic only; milestone success requires true dual pose + lift height.
    """
    mod = _load_dual_arm_module()
    mujoco = pytest.importorskip("mujoco")
    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene.xml")
    finally:
        os.chdir(prev)
    ddata = mujoco.MjData(model)
    mujoco.mj_forward(model, ddata)
    det = G1BoxPerception.detect_box(model, ddata, box_geom_name="box_geom")
    ly = float(det["left_edge_y"])
    ry = float(det["right_edge_y"])
    assert float(det["right_dual_pregrasp_target"][1]) < ly
    assert float(det["left_dual_pregrasp_target"][1]) > ry

    out = mod.run_g1_dual_arm_box(headless=True, timeout=18.0, verbose=False)
    assert isinstance(out, dict)
    assert abs(float(out["box_center_y"])) < 0.03
    assert out["lift_success"] is True, out
    assert out["left_palm_distance_to_target"] <= 0.10
    assert out["right_palm_distance_to_target"] < 0.07
    assert out["true_dual_contact_ready"] is True
    assert out["dual_contact_ready"] is True
    assert isinstance(out.get("proximity_fallback_used"), bool)
    assert isinstance(out.get("dual_contact_via_proximity_fallback"), bool)
    assert out["box_height_delta"] >= 0.04
    assert out["grasp_assist_used"] is True
    assert out["dual_pregrasp_reach_metric_ok"] is True
    assert out["initial_contact_any"] is False
    assert out["initial_contact_before_approach"] is False
    assert out["max_box_penetration_any_geom"] <= 0.005
    assert out["proximity_fallback_used_for_success"] is False
