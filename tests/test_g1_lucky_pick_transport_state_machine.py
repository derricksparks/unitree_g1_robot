from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
G1_DIR = REPO_ROOT / "simulation" / "mujoco" / "g1"
if str(G1_DIR) not in sys.path:
    sys.path.insert(0, str(G1_DIR))


def test_lucky_pick_transport_state_machine_imports() -> None:
    from lucky_bridge.lucky_scene_setup import detect_box_ground_truth, setup_lucky_pick_scene
    from run_g1_lucky_pick_lift_demo import run_g1_lucky_pick_lift_demo
    from run_g1_lucky_pick_transport_state_machine import run_g1_lucky_pick_transport_state_machine

    assert callable(detect_box_ground_truth)
    assert callable(setup_lucky_pick_scene)
    assert callable(run_g1_lucky_pick_lift_demo)
    assert callable(run_g1_lucky_pick_transport_state_machine)


def test_lucky_pick_scene_setup_and_detection() -> None:
    import mujoco

    from lucky_bridge.lucky_paths import LUCKY_SCENE_XML
    from lucky_bridge.lucky_scene_setup import compute_pick_point, detect_box_ground_truth, setup_lucky_pick_scene

    model = mujoco.MjModel.from_xml_path(str(LUCKY_SCENE_XML))
    data = mujoco.MjData(model)
    metrics = setup_lucky_pick_scene(
        model,
        data,
        box_size_xyz=(0.18, 0.12, 0.10),
        box_edge_offset=0.08,
    )
    detection = detect_box_ground_truth(model, data)
    pick = compute_pick_point(model, pick_stand_off=0.35, box_y=0.026)

    assert metrics["box_initial_pose_valid"] is True
    assert metrics["box_on_table"] is True
    assert metrics["box_fell_through_table"] is False
    assert metrics["table_collision_detected"] is True
    assert metrics["box_size_xyz"] == [0.18, 0.12, 0.10]
    assert metrics["box_graspable_size"] is True
    assert metrics["box_near_table_edge"] is True
    assert abs(metrics["box_distance_from_table_front_edge_m"] - 0.08) < 0.03
    assert detection.box_detected is True
    assert detection.detection_mode == "mujoco_ground_truth"
    assert detection.grasp_targets_above_table is True
    assert detection.grasp_targets_clear_table_edge is True
    assert isinstance(pick[0], float)
    assert isinstance(pick[1], float)


def test_lucky_pick_transport_state_machine_smoke() -> None:
    from run_g1_lucky_pick_transport_state_machine import run_g1_lucky_pick_transport_state_machine

    result = run_g1_lucky_pick_transport_state_machine(
        headless=True,
        timeout=8.0,
        pick_stand_off=0.35,
        auto_select_manipulation_stance=True,
        stance_foot="left",
        stance_forward_offset=0.08,
        stance_width_offset=0.03,
        manipulation_knee_bend=0.10,
        stance_duration=0.25,
        box_size_x=0.18,
        box_size_y=0.12,
        box_size_z=0.10,
        box_edge_offset=0.08,
        verbose=False,
    )

    assert isinstance(result, dict)
    assert result["box_initial_pose_valid"] is True
    assert result["box_on_table"] is True
    assert result["box_graspable_size"] is True
    assert result["box_near_table_edge"] is True
    assert result["box_detected"] is True
    assert "computed_pick_point" in result
    assert "grasp_target_height_above_table_m" in result
    assert "grasp_targets_above_table" in result
    assert "grasp_targets_clear_table_edge" in result
    assert "table_collision_blocking_reach" in result
    assert "WALK_TO_PICK_POINT" in result["phases_completed"]
    assert "STOP_AND_STABILIZE" in result["phases_completed"]
    assert "ENTER_MANIPULATION_STANCE" in result["phases_completed"]
    assert "DETECT_BOX" in result["phases_completed"]
    assert result["robot_fell_during_walk"] is False
    assert result["manipulation_stance_enabled"] is True
    assert "auto_select_manipulation_stance" in result
    assert "manipulation_stance_candidates_tested" in result
    assert "selected_stance_forward_offset_m" in result
    assert "selected_stance_width_offset_m" in result
    assert "selected_knee_bend_rad" in result
    assert "selected_torso_pitch_bias_rad" in result
    assert "best_stance_com_margin_m" in result
    assert "stance_selection_success" in result
    assert "manipulation_stance_fallback" in result
    assert "com_margin_before_stance_m" in result
    assert "com_margin_after_stance_m" in result
    assert "com_margin_improved_by_stance" in result
    assert "robot_fell_backward" in result
    assert result["com_margin_after_stance_m"] >= result["com_margin_before_stance_m"] - 0.006
    if result["stance_selection_success"] is False:
        assert result["manipulation_stance_fallback"] == "neutral_stand"
    assert "enter_manipulation_stance_success" in result
    assert "support_polygon_length_before_m" in result
    assert "support_polygon_length_after_m" in result
    assert "support_polygon_area_before_m2" in result
    assert "support_polygon_area_after_m2" in result
    assert "com_margin_before_reach_m" in result
    assert "com_margin_after_stance_m" in result
    assert "robot_fell_during_reach" in result
    assert "min_com_margin_during_reach_m" in result
    assert "left_hand_to_target_error_m" in result
    assert "right_hand_to_target_error_m" in result
    assert "expected_payload_forward_moment_nm" in result
    if result["enter_manipulation_stance_success"] is False:
        assert result["failure_reason"]
        assert result["reach_paused_for_stability"] is True
        assert result["robot_fell_during_reach"] is False
    if result["two_arm_reach_success"] is False:
        assert result["reach_failure_reason"]
    if result["success"] is False:
        assert result["failure_reason"]


def test_lucky_pick_transport_reach_only_tunes_standoff() -> None:
    from run_g1_lucky_pick_transport_state_machine import run_g1_lucky_pick_transport_state_machine

    result = run_g1_lucky_pick_transport_state_machine(
        headless=True,
        timeout=5.0,
        reach_only=True,
        auto_tune_pick_stand_off=True,
        reach_mode="staged",
        reach_duration=5.0,
        verbose=False,
    )

    assert result["reach_only_mode"] is True
    assert result["reach_animation_played"] is True
    assert result["sim_time_s"] >= result["reach_duration_s"]
    assert result["pick_stand_off_candidates_tested"] == 5
    assert result["best_pick_stand_off_m"] in (0.20, 0.25, 0.30, 0.35, 0.40)
    assert "best_reach_error_m" in result
    assert result["hand_measurement_frame_valid"] is True
    assert result["left_hand_measurement_name"] == "left_palm"
    assert result["right_hand_measurement_name"] == "right_palm"
    assert "left_hand_to_target_error_m" in result
    assert "right_hand_to_target_error_m" in result
    assert "final_left_hand_to_target_error_m" in result
    assert "final_right_hand_to_target_error_m" in result
    assert "selected_reach_preset" in result
    assert "best_preset_error_m" in result
    assert "arm_joint_mapping_valid" in result
    assert result["table_collision_blocking_reach"] is False
    if result["reach_near_success"] is True:
        assert result["reach_only_success"] is True
    if result["two_arm_reach_success"] is False:
        assert result["reach_failure_reason"]


def test_lucky_pick_lift_demo_assisted_headless() -> None:
    from run_g1_lucky_pick_lift_demo import run_g1_lucky_pick_lift_demo

    result = run_g1_lucky_pick_lift_demo(
        headless=True,
        timeout=20.0,
        assisted_grasp=True,
        reach_duration=0.75,
        post_done_hold=0.0,
        verbose=False,
    )

    assert isinstance(result, dict)
    assert result["box_on_table"] is True
    assert result["walk_to_pick_success"] is True
    assert result["stop_and_stabilize_success"] is True
    assert result["box_detected"] is True
    assert result["hand_box_collision_enabled"] is True
    assert result["hand_table_collision_enabled"] is True
    assert result["box_table_collision_enabled"] is True
    assert result["box_mass_kg"] >= 0.5
    assert result["box_friction"] >= 3.0
    assert result["table_friction"] >= 1.0
    assert result["pick_table_full_top_dimension"] is True
    assert result["pick_table_leg_collision_enabled"] is True
    assert "contact_pairs_detected" in result
    assert result["selected_reach_mode"] == "two_arm_box_ik"
    assert result["ik_candidates_tested"] > 0
    assert result["two_arm_ik_reach_success"] is True
    assert result["max_hand_to_target_error_m"] < 0.08
    assert result["assisted_grasp_used"] is True
    assert result["grasp_mode"] in {"assisted_weld", "physical_contact"}
    assert result["stable_finger_grip_mode"] == "box_stable_support_grasp"
    assert result["index_finger_under_box_commanded"] is True
    assert result["thumb_front_face_hold_commanded"] is True
    assert result["middle_finger_side_hold_commanded"] is True
    assert result["middle_finger_kept_mostly_extended"] is True
    assert result["palm_edge_hold_commanded"] is True
    assert result["grasp_attempt_success"] is True
    assert result["lift_attempted"] is True
    assert result["final_left_lift_target_error_m"] < 0.08
    assert result["final_right_lift_target_error_m"] < 0.08
    assert result["box_lift_success"] is True
    assert result["box_lift_height_m"] >= 0.05
    assert result["box_not_slid_into_fingers"] is True
    assert result["box_xy_shift_during_grasp_m"] <= 0.01
    assert result["robot_stable_after_lift"] is True
    assert result["min_com_margin_during_lift_m"] > 0.01
    assert "WALK_TO_PLACE_POINT" in result["phases_completed"]
    assert "PLACE_BOX" in result["phases_completed"]
    assert "RELEASE_BOX" in result["phases_completed"]
    assert result["box_firm_hold_mode"] == "assisted_carry_frame"
    assert result["max_box_slip_during_transport_m"] <= 0.025
    assert result["box_remained_firmly_held"] is True
    assert result["walk_to_place_success"] is True
    assert abs(result["table_center_distance_m"] - 3.0) < 1e-6
    assert abs(result["place_table_x_m"] - 0.351) < 1e-6
    assert result["place_table_y_m"] < -2.9
    assert result["place_stand_off_m"] == 0.2
    assert result["distance_to_place_point_m"] <= result["place_stop_tolerance_m"]
    assert result["box_transport_distance_m"] >= 2.50
    assert result["transport_base_assist_used"] is True
    assert result["step_gated_assisted_transport"] is True
    assert result["transport_assist_mode"] == "step_gated_base_assist"
    assert result["walker_driven_transport"] is False
    assert result["payload_mpc_wbc_enabled"] is True
    assert result["payload_mpc_controller_mode"] == "payload_mpc_wbc_stabilized"
    assert result["payload_mpc_balance_plan_valid"] is True
    assert result["payload_mpc_torso_pitch_bias_rad"] < 0.0
    assert result["payload_mpc_max_step_assist_m"] >= 0.0
    assert result["post_pick_backup_completed"] is True
    assert result["right_turn_after_pick_completed"] is True
    assert result["transport_path_line_drawn"] is True
    assert len(result["transport_path_waypoints"]) >= 4
    assert result["torso_backward_compensation_enabled"] is True
    assert result["carry_torso_backward_bias_rad"] < 0.0
    assert result["loaded_arm_step_shift_enabled"] is True
    assert result["transport_visible_step_count"] >= 4
    assert result["left_transport_step_count"] >= 2
    assert result["right_transport_step_count"] >= 2
    assert result["max_left_transport_foot_lift_m"] >= 0.008
    assert result["max_right_transport_foot_lift_m"] >= 0.008
    assert result["box_on_place_table"] is True
    assert result["finger_box_penetration_detected"] is False
    assert result["finger_box_max_penetration_m"] <= 0.001
    assert result["robot_table_collision_free"] is True
    assert result["robot_box_collision_free_except_grasp"] is True
    assert result["box_nonpenetrable_collision_model"] is True
    assert result["robot_stable_after_place"] is True
    assert result["min_com_margin_during_place_m"] > -0.02
    assert result["robot_fell_during_reach"] is False
    assert result["robot_fell_during_lift"] is False
    assert result["robot_fell_during_transport"] is False
    assert result["robot_fell_during_place"] is False
    assert result["success"] is True
    if result["success"] is False:
        assert result["failure_reason"]
