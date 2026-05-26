from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "simulation" / "mujoco" / "g1"
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))


def test_compact_carry_posture_validation_smoke() -> None:
    from run_g1_lucky_compact_carry_posture_validation import run_g1_lucky_compact_carry_posture_validation

    result = run_g1_lucky_compact_carry_posture_validation(
        headless=True,
        timeout=1.0,
        cmd_x=0.8,
        verbose=False,
    )

    assert result["compact_carry_posture_validation"] is True
    assert result["policy_loaded"] is True
    assert result["arms_locked_during_walking"] is True
    assert result["payload_present"] is False
    assert result["finger_hold_commanded"] is True
    assert result["finger_hold_mode"] == "box_stable_support_grasp"
    assert result["finger_hold_actuator_count"] > 0
    assert result["compact_carry_candidates_tested"] >= 3
    assert result["selected_compact_carry_posture"]
    assert result["compact_carry_posture_valid"] is True
    assert result["selected_compact_carry_posture"] == "front_center_chest_carry"
    assert result["left_elbow_bent_rad"] == -1.0472
    assert result["right_elbow_bent_rad"] == -1.0472
    assert result["avg_palm_torso_distance_m"] < 0.24
    assert 0.12 <= result["avg_palm_forward_offset_m"] <= 0.16
    assert result["palms_at_chest_height"] is True
    assert result["palms_front_centered"] is True
    assert result["palms_parallel"] is True
    assert result["palms_face_front"] is True
    assert result["hands_not_folded_inward"] is True
    assert result["palms_parallel_score"] >= 0.95
    assert result["palms_front_facing_score"] >= 0.89
    assert abs(result["palm_lateral_center_offset_m"]) <= 0.06
    assert 0.23 <= result["avg_palm_height_above_pelvis_m"] <= 0.25
    assert "success" in result
    if result["success"] is False:
        assert result["failure_reason"]


def test_compact_carry_posture_walks_stably() -> None:
    from run_g1_lucky_compact_carry_posture_validation import run_g1_lucky_compact_carry_posture_validation

    result = run_g1_lucky_compact_carry_posture_validation(
        headless=True,
        timeout=10.0,
        cmd_x=0.8,
        verbose=False,
    )

    assert result["success"] is True
    assert result["robot_fell"] is False
    assert result["actual_visible_step_count"] >= 2
    assert result["real_foot_placement_count"] >= 2
    assert result["pelvis_forward_progress_m"] >= 0.5
    assert result["left_actual_foot_lift_m"] >= 0.008
    assert result["right_actual_foot_lift_m"] >= 0.008
    assert result["max_stance_foot_slip_m"] < 0.025
    assert result["actual_floor_contact_detected"] is True


def test_compact_carry_posture_walks_with_minimum_weight_attached_box() -> None:
    from run_g1_lucky_compact_carry_posture_validation import run_g1_lucky_compact_carry_posture_validation

    result = run_g1_lucky_compact_carry_posture_validation(
        headless=True,
        timeout=10.0,
        cmd_x=0.8,
        attach_box=True,
        box_mass_kg=0.01,
        box_size_x=0.12,
        box_size_y=0.338,
        box_size_z=0.10,
        attached_box_finger_mode="light_side_prepare",
        verbose=False,
    )

    assert result["success"] is True
    assert result["payload_present"] is True
    assert result["payload_attach_mode"] == "fixed_to_hands_collision_enabled"
    assert result["payload_mass_kg"] == 0.01
    assert result["payload_minimum_weight_trial"] is True
    assert result["attached_box_size_xyz_m"] == [0.12, 0.338, 0.10]
    assert result["attached_box_collision_enabled"] is True
    assert result["attached_box_fixed_to_hands"] is True
    assert result["attached_box_fits_between_palms"] is True
    assert result["attached_box_side_clearance_m"] < 0.002
    assert result["max_hand_box_penetration_m"] <= 0.003
    assert result["hand_box_impenetrable_visual"] is True
    assert result["left_elbow_bent_rad"] == -1.0472
    assert result["right_elbow_bent_rad"] == -1.0472
    assert result["avg_palm_torso_distance_m"] < 0.24
    assert 0.23 <= result["avg_palm_height_above_pelvis_m"] <= 0.25
    assert result["actual_visible_step_count"] >= 2
    assert result["robot_fell"] is False
