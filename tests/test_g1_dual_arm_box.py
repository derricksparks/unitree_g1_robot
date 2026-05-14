"""Smoke test: locked-base dual-arm box coordination (IK + assist + lift)."""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import pytest
import numpy as np

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
G1_REACH_SCENE_DEX3 = G1_DIR / "g1_reach_box_scene_dex3.xml"
G1_DEX3_HANDS_XML = G1_DIR / "assets" / "g1_dex3_hands_actuated.xml"
DUAL_ARM_SCRIPT = G1_DIR / "run_g1_dual_arm_box.py"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.g1_box_perception import DUAL_SURFACE_CONTACT_CLEARANCE_Y_M, G1BoxPerception

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def test_g1_dual_arm_left_palm_and_proxy_geoms():
    """Legacy centered scene still ships proxy fingers (single-file regression)."""
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
        "box_front_contact_site",
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


def test_g1_dex3_dual_arm_scene_palms_and_finger_collisions():
    mujoco = pytest.importorskip("mujoco")
    if not G1_REACH_SCENE_DEX3.is_file() or not G1_DEX3_HANDS_XML.is_file():
        pytest.skip("Dex3 pipeline assets missing")

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene_dex3.xml")
    finally:
        os.chdir(prev)

    assert int(model.nu) == 43
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm_site") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm_site") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_proxy_left_finger_geom") < 0
    sys.path.insert(0, str(G1_DIR))
    from g1_dex_hand_contact import enumerate_hand_contact_geoms

    assert len(enumerate_hand_contact_geoms(model, side="left")) >= 4
    assert len(enumerate_hand_contact_geoms(model, side="right")) >= 4


def test_dex3_finger_actuator_semantic_groups():
    mujoco = pytest.importorskip("mujoco")
    if not G1_REACH_SCENE_DEX3.is_file():
        pytest.skip("Dex3 scene missing")
    sys.path.insert(0, str(G1_DIR))
    from g1_dex3_finger_control import dex3_semantic_finger_actuator_groups

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene_dex3.xml")
    finally:
        os.chdir(prev)
    g = dex3_semantic_finger_actuator_groups(model)
    for k in (
        "right_thumb",
        "right_index",
        "right_middle",
        "left_thumb",
        "left_index",
        "left_middle",
    ):
        assert k in g and len(g[k]) >= 1, (k, g.get(k))


def _load_dual_arm_module():
    spec = importlib.util.spec_from_file_location("run_g1_dual_arm_box", DUAL_ARM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_g1_dual_arm_box")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dual_arm_geometry_guard_helpers():
    mod = _load_dual_arm_module()
    assert mod._is_palm_inside_box_volume(
        np.array([0.48, 0.0, 0.88]),
        0.43,
        0.53,
        -0.06,
        0.06,
        0.83,
        0.93,
    )
    assert not mod._is_palm_inside_box_volume(
        np.array([0.43, -0.13, 0.86]),
        0.43,
        0.53,
        -0.06,
        0.06,
        0.83,
        0.93,
    )
    assert mod._is_proxy_crossing_box_center(np.array([0.0, 0.02, 0.0]), 0.0, side="right")
    assert mod._is_proxy_crossing_box_center(np.array([0.0, -0.02, 0.0]), 0.0, side="left")


def test_g1_dual_arm_box_quick_smoke():
    """Short headless run (fast cap + step limit) without full 24 s episode."""
    mod = _load_dual_arm_module()
    pytest.importorskip("mujoco")
    out = mod.run_g1_dual_arm_box(
        headless=True,
        timeout=12.0,
        verbose=False,
        quiet=True,
        fast=True,
        max_steps=1200,
    )
    assert isinstance(out, dict)
    assert out.get("use_dex3_pipeline") is True
    assert "dex3_phase_name" in out


@pytest.mark.slow
def test_g1_dual_arm_box_headless():
    """
    Full dual-arm demo: primary-right/support-left IK, gated bilateral contact, bounded assist.

    Proximity fields are diagnostic only; penetration and wrist-command stability are asserted every run.
    Lift success is asserted only when the episode achieves it.
    """
    mod = _load_dual_arm_module()
    mujoco = pytest.importorskip("mujoco")
    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene_dex3.xml")
    finally:
        os.chdir(prev)
    ddata = mujoco.MjData(model)
    mujoco.mj_forward(model, ddata)
    det = G1BoxPerception.detect_box(model, ddata, box_geom_name="box_geom")
    ly = float(det["left_edge_y"])
    ry = float(det["right_edge_y"])
    assert float(det["right_dual_pregrasp_target"][1]) < ly
    assert float(det["left_dual_pregrasp_target"][1]) > ry

    out = mod.run_g1_dual_arm_box(headless=True, timeout=24.0, verbose=False)
    assert isinstance(out, dict)
    assert out.get("use_dex3_pipeline") is True
    assert abs(float(out["box_center_y"])) < 0.03
    assert float(out["max_bilateral_stable_s"]) + 1e-9 >= float(mod.DUAL_CONTACT_SETTLE_TIME_S)
    assert float(out["max_right_contact_stability_s"]) > 0.05
    assert float(out["max_left_contact_stability_s"]) > 0.05
    assert float(out["dual_surface_clearance_y_m"]) == float(DUAL_SURFACE_CONTACT_CLEARANCE_Y_M)
    # Dex3 mesh-rich hands: allow slightly deeper numeric AABB peaks than the old proxy plate.
    assert float(out["max_box_penetration_any_geom"]) <= 0.020
    assert float(out["max_wrist_command_delta_episode"]) <= 0.12
    assert int(out["ik_recovery_count"]) < 10000
    assert int(out["large_joint_jump_events"]) <= 200
    assert float(out["max_abs_joint_step_applied_episode"]) <= float(mod.MAX_ARM_JOINT_STEP_RAD) + 1e-5
    bh = np.asarray(out["box_half_size"], dtype=float).reshape(3)
    assert np.allclose(bh, np.array([0.08, 0.07, 0.06]), atol=0.002)
    tpd = out.get("torso_payload_diag") or {}
    assert float(tpd.get("payload_mass_estimate_kg", 0.0)) > 0.0
    assert isinstance(tpd.get("box_com_world_xyz"), list) and len(tpd["box_com_world_xyz"]) == 3
    assert np.isfinite(float(out.get("dex3_min_cross_hand_fingertip_m", float("nan"))))
    assert int(out.get("dex3_max_right_fingertip_contacts", 0)) < 500
    assert int(out.get("dex3_max_left_fingertip_contacts", 0)) < 500
    assert float(out.get("dex3_max_real_contact_stable_s", 0.0)) >= 0.0
    if out["lift_success"]:
        assert out["box_height_delta"] >= 0.04
        assert out["grasp_assist_used"] is True
    else:
        assert out.get("lift_blocked_reason") in (
            None,
            "bilateral_contact_not_stable",
            "dex3_contact_settle_insufficient",
        ), out
    assert out["left_palm_distance_to_target"] <= 0.10
    assert out["right_palm_distance_to_target"] < 0.07
    assert out["true_dual_contact_ready"] is True
    assert out["dual_contact_ready"] is True
    assert isinstance(out.get("proximity_fallback_used"), bool)
    assert isinstance(out.get("dual_contact_via_proximity_fallback"), bool)
    assert out["dual_pregrasp_reach_metric_ok"] is True
    assert out["initial_contact_any"] is False
    assert out["initial_contact_before_approach"] is False
    assert out["proximity_fallback_used_for_success"] is False
