"""Tests for simulated G1 box perception (MuJoCo geom truth)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
G1_POS_XML = G1_DIR / "assets" / "g1_position_actuated.xml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))

from perception.g1_box_perception import G1BoxPerception

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def test_detect_box_reach_scene():
    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene.xml")
    finally:
        os.chdir(prev)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    d = G1BoxPerception.detect_box(model, data, box_geom_name="box_geom")
    cx = float(d["box_center"][0])

    # Dual palm IK targets sit outside −y / +y box faces (not inside the obstacle volume).
    assert abs(float(d["box_center"][1])) < 0.03
    ly = float(d["left_edge_y"])
    ry = float(d["right_edge_y"])
    assert float(d["right_dual_pregrasp_target"][1]) < ly - 1e-6
    assert float(d["left_dual_pregrasp_target"][1]) > ry + 1e-6

    assert d["near_face_x"] < cx
    assert d["bottom_z"] < d["grasp_height_z"] < d["top_z"]

    assert np.allclose(
        d["viz_front_contact_world"],
        np.array([d["near_face_x"], float(d["box_center"][1]), d["grasp_height_z"]]),
        atol=1e-7,
    )

    # Palm pre-grasp sits on the −x side of the mathematical near face (toward the robot).
    assert d["palm_pregrasp_target"][0] < d["near_face_x"]


def test_detect_box_dex3_scene_half_size_and_semantic_targets():
    """Dex3-only enlarged box: geom half-extents drive faces and Dex3 hint sites."""
    dex3_scene = G1_DIR / "g1_reach_box_scene_dex3.xml"
    if not dex3_scene.is_file():
        pytest.skip("Dex3 reach scene missing")

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene_dex3.xml")
    finally:
        os.chdir(prev)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    d = G1BoxPerception.detect_box(model, data, box_geom_name="box_geom")

    h = np.asarray(d["box_half_size"], dtype=float).reshape(3)
    assert np.allclose(h, np.array([0.08, 0.07, 0.06]), atol=0.002)
    ly = float(d["left_edge_y"])
    ry = float(d["right_edge_y"])
    # ``left_edge_y`` / ``right_edge_y`` are ymin / ymax in world frame (robot −y is the right-hand face).
    assert ly < ry
    assert float(d["dex3_right_index_target_world"][1]) < ly
    assert float(d["dex3_left_index_target_world"][1]) > ry
    for k in (
        "dex3_right_index_target_world",
        "dex3_right_middle_target_world",
        "dex3_right_thumb_target_world",
        "dex3_right_lower_support_target_world",
        "dex3_left_index_target_world",
        "dex3_left_middle_target_world",
        "dex3_left_thumb_target_world",
        "dex3_left_lower_support_target_world",
    ):
        assert k in d
        assert np.asarray(d[k], dtype=float).shape == (3,)
