"""Tests for Dex3 hand–box contact helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1"
G1_POS_XML = G1_DIR / "assets" / "g1_position_actuated.xml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))

from g1_dex_hand_contact import (  # noqa: E402
    dex3_finger_clearance_metrics,
    dex3_per_hand_grasp_contact_ready,
)

pytestmark = pytest.mark.skipif(
    not G1_POS_XML.is_file(),
    reason="g1_position_actuated.xml not present",
)


def test_dex3_per_hand_grasp_contact_ready_logic():
    assert dex3_per_hand_grasp_contact_ready(palm_contacts=0, fingertip_contacts=0) is False
    assert dex3_per_hand_grasp_contact_ready(palm_contacts=0, fingertip_contacts=1) is False
    assert dex3_per_hand_grasp_contact_ready(palm_contacts=1, fingertip_contacts=1) is True
    assert dex3_per_hand_grasp_contact_ready(palm_contacts=0, fingertip_contacts=2) is True


def test_dex3_finger_clearance_metrics_on_dex3_scene():
    mujoco = pytest.importorskip("mujoco")
    scene = G1_DIR / "g1_reach_box_scene_dex3.xml"
    if not scene.is_file():
        pytest.skip("Dex3 scene missing")

    prev = os.getcwd()
    try:
        os.chdir(G1_DIR)
        model = mujoco.MjModel.from_xml_path("g1_reach_box_scene_dex3.xml")
    finally:
        os.chdir(prev)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    m = dex3_finger_clearance_metrics(model, data)
    for k in ("cross_hand_fingertip_min_m", "right_hand_fingertip_min_m", "left_hand_fingertip_min_m"):
        assert k in m
        assert float(m[k]) < float("inf")
