from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "simulation" / "mujoco" / "g1"
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))


def test_table_pick_lifts_box_to_locked_carry_posture() -> None:
    from run_g1_lucky_table_pick_locked_carry import run_g1_lucky_table_pick_locked_carry

    result = run_g1_lucky_table_pick_locked_carry(
        headless=True,
        timeout=30.0,
        box_mass_kg=0.01,
        box_size_x=0.12,
        box_size_y=0.33867925908248114,
        box_size_z=0.10,
        lift_duration_s=1.6,
        verbose=False,
    )

    assert result["success"] is True
    assert result["box_initially_on_table"] is True
    assert result["walk_to_pick_table_success"] is True
    assert result["robot_stood_back_from_table"] is True
    assert result["robot_table_standoff_m"] >= 0.10
    assert result["approach_start_distance_from_table_m"] == 0.5
    assert result["distance_to_arm_raise_standoff_m"] <= 0.20
    assert result["second_table_configured"] is True
    assert result["second_table_distance_m"] == 2.0
    assert result["reach_to_box_success"] is True
    assert result["approach_with_hands_in_carry_position"] is False
    assert result["target_upper_body_metrics_preserved"] is True
    assert result["box_lift_height_m"] > 0.15
    assert result["box_motion_before_grip_m"] == 0.0
    assert result["box_slid_into_hands_before_grip"] is False
    assert result["hands_touch_box_before_lift"] is True
    assert result["hand_box_side_gap_at_grip_m"] <= 1e-4
    assert result["all_three_fingers_commanded_to_touch"] is True
    assert result["all_three_fingers_touch_box_before_lift"] is True
    assert result["box_fixed_to_hands_before_lift"] is True
    assert result["box_lift_driven_by_hands"] is True
    assert result["hand_box_lift_synchronized"] is True
    assert result["lift_duration_s"] == 1.6
    assert result["chest_hold_duration_s"] == 3.0
    assert result["arm_raise_duration_s"] == 1.4
    assert result["reach_duration_s"] == 1.6
    assert result["grip_duration_s"] == 1.0
    assert result["box_lifted_vertically_from_table"] is True
    assert result["box_horizontal_shift_during_table_lift_m"] <= 0.005
    assert result["backward_carry_removed"] is True
    assert result["pick_table_turn_clearance_m"] >= 0.24
    assert result["right_turn_success"] is True
    assert result["right_turn_duration_s"] == 1.2
    assert result["right_turn_stable"] is True
    assert result["right_turn_assisted_in_place"] is False
    assert result["right_turn_assisted_finish"] is True
    assert result["right_turn_xy_drift_m"] <= 0.7
    assert result["right_turn_yaw_change_deg"] < -25.0
    assert result["travel_to_second_table_success"] is True
    assert result["transport_final_alignment_assisted"] is True
    assert result["box_placed_on_second_table"] is True
    assert result["box_place_error_m"] <= 0.03
    assert max(result["place_left_error_m"], result["place_right_error_m"]) <= 0.18
    assert result["payload_attach_mode"] == "fixed_to_hands_collision_enabled_after_table_pick"
    assert result["max_hand_box_penetration_m"] == 0.0
    assert result["hand_box_impenetrable_visual"] is True
    assert result["hand_table_collision_free"] is True
    assert result["max_hand_table_contact_count"] == 0
    assert all(count == 0 for count in result["hand_table_contact_count_by_phase"].values())
    assert result["max_hand_table_penetration_m"] == 0.0
    assert result["robot_table_collision_free"] is True
    assert result["max_robot_table_contact_count"] == 0
    assert all(count == 0 for count in result["robot_table_contact_count_by_phase"].values())
    assert result["max_robot_table_penetration_m"] == 0.0
    assert result["left_elbow_bent_rad"] == -1.0472
    assert result["right_elbow_bent_rad"] == -1.0472
    assert result["avg_palm_height_m"] == 1.0005638911825643
    assert result["robot_fell"] is False
