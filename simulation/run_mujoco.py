import argparse
import importlib
import sys
import time
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.locomotion_mpc import LocomotionMPC
from control.arm_ik_controller import ArmIKController
from evaluation.metrics_logger import MetricsLogger
from perception.object_detector import ObjectDetector


MODEL_PATH = "simulation/mujoco/world.xml"

USE_IK_ARM_CONTROL = True

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.04
GRIPPER_CLOSE_TIME = 0.45
PRINT_INTERVAL = 0.10
HEADLESS_TIMEOUT = 30.0
SLIP_DISTANCE_THRESHOLD = 0.05
GRASP_ASSIST_STIFFNESS = 1200.0
GRASP_ASSIST_DAMPING = 120.0
GRASP_ASSIST_MAX_FORCE = 700.0

ARM_STOW_JOINTS = (0.0, 0.0)
ARM_ABOVE_BOX_JOINTS = (0.47, -0.50)
ARM_PICK_JOINTS = (0.35, 0.45)
ARM_LIFT_JOINTS = (0.47, -0.45)
ARM_CARRY_JOINTS = (0.20, 0.20)
ARM_SHELF_APPROACH_JOINTS = (0.25, -0.40)
ARM_PLACE_JOINTS = (0.25, -0.40)

# Task-space targets for the placeholder 2-link arm. Stow, carry, and shelf
# approach are base-relative because the hand should keep the same posture while
# the sliding base navigates. Pick/lift targets are box-relative so randomized
# object positions remain reachable. Place is shelf-target-relative so randomized
# placement goals move the hand consistently with the target shelf point.
STOW_TARGET = np.array([0.75, -0.12, 0.22])
ABOVE_BOX_TARGET = np.array([-0.05, 0.0, 0.27])
PICK_TARGET = np.array([-0.118, 0.0, 0.075])
LIFT_TARGET = np.array([-0.05, 0.0, 0.255])
CARRY_TARGET = np.array([0.68664369, -0.12, 0.0473531])
SHELF_APPROACH_TARGET = np.array([0.70168884, -0.12, 0.19538568])
PLACE_TARGET = np.array([-0.08, 0.0, 0.035])

TABLE_WALK_TARGET = np.array([0.55, 0.0, 1.0])
BACK_AWAY_TARGET = np.array([0.35, 0.0, 1.0])
SIDE_LANE_Y = -0.65
TABLE_X_RANGE = (0.75, 1.65)
TABLE_Y_RANGE = (-0.45, 0.45)
TABLE_CLEAR_X = 1.75
SHELF_X_RANGE = (2.55, 3.45)
SHELF_Y_RANGE = (-0.45, 0.45)
SHELF_Z_RANGE = (0.90, 1.10)
SHELF_TOP_Z = 1.10
SHELF_APPROACH_CLEARANCE_Z = 1.14
SHELF_HAND_CLEARANCE_Z = SHELF_APPROACH_CLEARANCE_Z
SHELF_WALK_TARGET = np.array([2.15, 0.0, 1.0])
SHELF_PLACE_POS = np.array([2.90, -0.12, 1.16])
CARRY_OFFSET_FROM_HAND = np.array([0.08, 0.0, -0.04])
BOX_OFFSET_FROM_FINGER_MIDPOINT = np.array([0.035, 0.0, 0.0])
SHELF_CLEARANCE_OFFSET_FROM_HAND = np.array([0.08, 0.0, 0.01])
BOX_DEFAULT_POS = np.array([1.2, -0.12, 0.84])
BOX_DEFAULT_QUAT = (1.0, 0.0, 0.0, 0.0)
PICK_HAND_OFFSET_FROM_BASE = np.array([0.65, -0.12])
GRASP_STABLE_TIME = 0.12
LIFT_DURATION = 0.9
PLACE_DURATION = 0.8


class TaskPhase(Enum):
    WALK_TO_TABLE = auto()
    REACH_ABOVE_BOX = auto()
    LOWER_TO_BOX = auto()
    GRASP_BOX = auto()
    LIFT_BOX = auto()
    BACK_AWAY_FROM_TABLE = auto()
    WALK_AROUND_TABLE = auto()
    WALK_TO_SHELF = auto()
    REACH_SHELF = auto()
    PLACE_BOX = auto()
    RELEASE_BOX = auto()
    DONE = auto()


def get_body_pos(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


def get_geom_pos(model, data, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[geom_id].copy()


def get_geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def get_actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def get_joint_qpos_addr(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[joint_id]


def get_joint_qvel_addr(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_dofadr[joint_id]


def zero_base_velocity(data, joint_addrs):
    data.qvel[joint_addrs["base_x_qvel"]] = 0.0
    data.qvel[joint_addrs["base_y_qvel"]] = 0.0


def hold_base(data, joint_addrs, base_xy):
    data.qpos[joint_addrs["base_x_qpos"]] = base_xy[0]
    data.qpos[joint_addrs["base_y_qpos"]] = base_xy[1]
    zero_base_velocity(data, joint_addrs)


def table_collision_risk(base_pos):
    return (
        TABLE_X_RANGE[0] <= base_pos[0] <= TABLE_X_RANGE[1]
        and TABLE_Y_RANGE[0] <= base_pos[1] <= TABLE_Y_RANGE[1]
    )


def shelf_collision_risk(base_pos):
    return (
        SHELF_X_RANGE[0] <= base_pos[0] <= SHELF_X_RANGE[1]
        and SHELF_Y_RANGE[0] <= base_pos[1] <= SHELF_Y_RANGE[1]
    )


def shelf_hand_collision_risk(hand_pos):
    return (
        SHELF_X_RANGE[0] <= hand_pos[0] <= SHELF_X_RANGE[1]
        and SHELF_Y_RANGE[0] <= hand_pos[1] <= SHELF_Y_RANGE[1]
        and SHELF_Z_RANGE[0] <= hand_pos[2] <= SHELF_HAND_CLEARANCE_Z
    )


def box_shelf_collision_risk(box_pos):
    return (
        SHELF_X_RANGE[0] <= box_pos[0] <= SHELF_X_RANGE[1]
        and SHELF_Y_RANGE[0] <= box_pos[1] <= SHELF_Y_RANGE[1]
        and SHELF_Z_RANGE[0] <= box_pos[2] <= SHELF_TOP_Z
    )


def enforce_table_clearance(data, joint_addrs):
    x = data.qpos[joint_addrs["base_x_qpos"]]
    y = data.qpos[joint_addrs["base_y_qpos"]]
    if TABLE_X_RANGE[0] <= x <= TABLE_X_RANGE[1] and TABLE_Y_RANGE[0] <= y <= TABLE_Y_RANGE[1]:
        data.qpos[joint_addrs["base_y_qpos"]] = SIDE_LANE_Y
        zero_base_velocity(data, joint_addrs)


def enforce_shelf_clearance(data, joint_addrs):
    x = data.qpos[joint_addrs["base_x_qpos"]]
    y = data.qpos[joint_addrs["base_y_qpos"]]
    if SHELF_X_RANGE[0] <= x <= SHELF_X_RANGE[1] and SHELF_Y_RANGE[0] <= y <= SHELF_Y_RANGE[1]:
        data.qpos[joint_addrs["base_x_qpos"]] = SHELF_X_RANGE[0] - 0.05
        zero_base_velocity(data, joint_addrs)


def move_base_toward(data, joint_addrs, mpc, base_pos, target_pos, dt):
    distance = np.linalg.norm(base_pos[:2] - target_pos[:2])
    if distance < 0.05:
        zero_base_velocity(data, joint_addrs)
        return 0.0

    cmd = mpc.compute_command({"base_pos": base_pos}, target_pos)
    data.qpos[joint_addrs["base_x_qpos"]] += cmd["vx"] * dt
    data.qpos[joint_addrs["base_y_qpos"]] += cmd["vy"] * dt
    zero_base_velocity(data, joint_addrs)
    enforce_table_clearance(data, joint_addrs)
    enforce_shelf_clearance(data, joint_addrs)
    return float(np.hypot(cmd["vx"], cmd["vy"]))


def command_arm_joints(data, actuator_ids, joint_addrs, shoulder, elbow):
    data.ctrl[actuator_ids["shoulder"]] = shoulder
    data.ctrl[actuator_ids["elbow"]] = elbow
    data.qpos[joint_addrs["shoulder_qpos"]] = shoulder
    data.qpos[joint_addrs["elbow_qpos"]] = elbow
    data.qvel[joint_addrs["shoulder_qvel"]] = 0.0
    data.qvel[joint_addrs["elbow_qvel"]] = 0.0


def command_arm_to_target(
    data,
    actuator_ids,
    joint_addrs,
    arm_ik,
    shoulder_pos,
    target_pos,
):
    current_joints = (
        data.qpos[joint_addrs["shoulder_qpos"]],
        data.qpos[joint_addrs["elbow_qpos"]],
    )
    targets = arm_ik.compute_joint_targets(
        shoulder_pos,
        target_pos,
        current_joints=current_joints,
    )
    command_arm_joints(
        data,
        actuator_ids,
        joint_addrs,
        targets["shoulder"],
        targets["elbow"],
    )


def command_arm(
    data,
    actuator_ids,
    joint_addrs,
    arm_ik,
    shoulder_pos,
    target_pos,
    fallback_joints,
):
    if USE_IK_ARM_CONTROL:
        command_arm_to_target(
            data, actuator_ids, joint_addrs, arm_ik, shoulder_pos, target_pos
        )
        return np.asarray(target_pos, dtype=float)

    command_arm_joints(data, actuator_ids, joint_addrs, *fallback_joints)
    return None


def interpolate_joints(start_joints, end_joints, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start = np.asarray(start_joints, dtype=float)
    end = np.asarray(end_joints, dtype=float)
    return tuple(start + (end - start) * alpha)


def interpolate_target(start_target, end_target, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(start_target, dtype=float) + (
        np.asarray(end_target, dtype=float) - np.asarray(start_target, dtype=float)
    ) * alpha


def command_gripper(data, actuator_ids, joint_addrs, opening):
    data.ctrl[actuator_ids["left_finger"]] = opening
    data.ctrl[actuator_ids["right_finger"]] = opening
    data.qpos[joint_addrs["left_finger_qpos"]] = opening
    data.qpos[joint_addrs["right_finger_qpos"]] = opening
    data.qvel[joint_addrs["left_finger_qvel"]] = 0.0
    data.qvel[joint_addrs["right_finger_qvel"]] = 0.0


def set_freejoint_pose(model, data, joint_name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    qpos_addr = get_joint_qpos_addr(model, joint_name)
    qvel_addr = get_joint_qvel_addr(model, joint_name)
    data.qpos[qpos_addr:qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3:qpos_addr + 7] = quat
    data.qvel[qvel_addr:qvel_addr + 6] = 0.0


def contact_between(model, data, geom_a_name, geom_b_name):
    geom_a_id = get_geom_id(model, geom_a_name)
    geom_b_id = get_geom_id(model, geom_b_name)
    count = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if {contact.geom1, contact.geom2} == {geom_a_id, geom_b_id}:
            count += 1
    return count


def box_contacts_shelf(model, data, shelf_geom_name="shelf_top_geom"):
    return contact_between(model, data, "box_geom", shelf_geom_name) > 0


def detect_grasp(model, data):
    # The grasp event is contact-gated: both sliding finger geoms must touch the
    # box before the controller is allowed to use any external grasp assistance.
    box_pos = get_body_pos(model, data, "box")
    left_finger_pos = get_geom_pos(model, data, "right_left_finger_geom")
    right_finger_pos = get_geom_pos(model, data, "right_right_finger_geom")
    left_contact_count = contact_between(
        model, data, "right_left_finger_geom", "box_geom"
    )
    right_contact_count = contact_between(
        model, data, "right_right_finger_geom", "box_geom"
    )
    support_contact_count = contact_between(
        model, data, "right_gripper_support_geom", "box_geom"
    )
    left_distance = np.linalg.norm(left_finger_pos - box_pos)
    right_distance = np.linalg.norm(right_finger_pos - box_pos)
    success = left_contact_count > 0 and right_contact_count > 0
    return {
        "success": bool(success),
        "contact_count": int(
            left_contact_count + right_contact_count + support_contact_count
        ),
        "left_contact": bool(left_contact_count > 0),
        "right_contact": bool(right_contact_count > 0),
        "support_contact": bool(support_contact_count > 0),
        "left_distance": float(left_distance),
        "right_distance": float(right_distance),
    }


def detect_box_slipping(hand_pos, box_pos, grasp_reference_distance):
    if grasp_reference_distance is None:
        return False
    hand_box_distance = np.linalg.norm(hand_pos - box_pos)
    return bool(hand_box_distance > grasp_reference_distance + SLIP_DISTANCE_THRESHOLD)


def apply_box_grasp_assist(model, data, target_pos):
    # This is an assisted research-prototype grasp. After real finger-box
    # contact is detected, MuJoCo's xfrc_applied adds a bounded stabilizing
    # force that keeps the payload near the gripper target. It is not a fully
    # passive force-closure grasp, and it does not teleport the box during carry.
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
    qvel_addr = model.jnt_dofadr[box_joint_id]
    box_pos = data.xpos[box_body_id]
    box_vel = data.qvel[qvel_addr:qvel_addr + 3]
    force = (
        GRASP_ASSIST_STIFFNESS * (target_pos - box_pos)
        - GRASP_ASSIST_DAMPING * box_vel
    )
    force_norm = np.linalg.norm(force)
    if force_norm > GRASP_ASSIST_MAX_FORCE:
        force *= GRASP_ASSIST_MAX_FORCE / force_norm
        force_norm = GRASP_ASSIST_MAX_FORCE
    data.xfrc_applied[box_body_id, :3] = force
    return float(force_norm)


def get_gripper_carry_target(model, data):
    left_finger_pos = get_geom_pos(model, data, "right_left_finger_geom")
    right_finger_pos = get_geom_pos(model, data, "right_right_finger_geom")
    finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
    return finger_midpoint + BOX_OFFSET_FROM_FINGER_MIDPOINT


def build_joint_address_map(model):
    return {
        "base_x_qpos": get_joint_qpos_addr(model, "robot_slide_x"),
        "base_y_qpos": get_joint_qpos_addr(model, "robot_slide_y"),
        "base_x_qvel": get_joint_qvel_addr(model, "robot_slide_x"),
        "base_y_qvel": get_joint_qvel_addr(model, "robot_slide_y"),
        "shoulder_qpos": get_joint_qpos_addr(model, "right_shoulder_pitch"),
        "elbow_qpos": get_joint_qpos_addr(model, "right_elbow_pitch"),
        "shoulder_qvel": get_joint_qvel_addr(model, "right_shoulder_pitch"),
        "elbow_qvel": get_joint_qvel_addr(model, "right_elbow_pitch"),
        "left_finger_qpos": get_joint_qpos_addr(model, "right_left_finger_slide"),
        "right_finger_qpos": get_joint_qpos_addr(model, "right_right_finger_slide"),
        "left_finger_qvel": get_joint_qvel_addr(model, "right_left_finger_slide"),
        "right_finger_qvel": get_joint_qvel_addr(model, "right_right_finger_slide"),
    }


def build_actuator_id_map(model):
    return {
        "shoulder": get_actuator_id(model, "right_shoulder_motor"),
        "elbow": get_actuator_id(model, "right_elbow_motor"),
        "left_finger": get_actuator_id(model, "right_left_finger_motor"),
        "right_finger": get_actuator_id(model, "right_right_finger_motor"),
    }


def get_walk_around_target(base_pos, shelf_walk_target):
    if abs(base_pos[1] - SIDE_LANE_Y) > 0.05:
        return np.array([BACK_AWAY_TARGET[0], SIDE_LANE_Y, 1.0])
    if base_pos[0] < TABLE_CLEAR_X:
        return np.array([TABLE_CLEAR_X, SIDE_LANE_Y, 1.0])
    return shelf_walk_target


def get_height_checks(phase, box_pos, shelf_goal):
    return {
        "lift_box_z_gt_1.0": phase == TaskPhase.LIFT_BOX and box_pos[2] > 1.0,
        "carry_box_z_gt_0.95": phase == TaskPhase.WALK_TO_SHELF and box_pos[2] > 0.95,
        "box_above_shelf_top": phase in (TaskPhase.REACH_SHELF, TaskPhase.PLACE_BOX)
        and box_pos[2] > SHELF_TOP_Z,
        "place_near_target": phase == TaskPhase.PLACE_BOX
        and np.linalg.norm(box_pos - shelf_goal) < 0.08,
    }


def transfer_success(phase, box_pos, shelf_place_pos=SHELF_PLACE_POS):
    return phase == TaskPhase.DONE and np.linalg.norm(box_pos - shelf_place_pos) < 0.10


def get_payload_mass_kg(model):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    return float(model.body_mass[body_id])


def set_payload_mass_kg(model, mass_kg):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    model.body_mass[body_id] = float(mass_kg)


def print_status(
    phase,
    speed,
    base_pos,
    hand_pos,
    box_pos,
    gripper_target,
    grasp_status,
    navigation_target,
    table_risk,
    shelf_risk,
    hand_shelf_risk,
    box_shelf_risk,
    height_checks,
    transfer_ok,
    box_slipping,
    grasp_assist_active,
    assist_force_norm,
    current_hand_target,
    hand_target_error,
):
    target_text = None if current_hand_target is None else np.asarray(current_hand_target)
    error_text = "None" if hand_target_error is None else f"{hand_target_error:.3f}"
    print(
        f"Phase: {phase.name.lower()} | "
        f"Speed: {speed:.3f} | "
        f"Base: {base_pos} | "
        f"NavTarget: {navigation_target} | "
        f"TableRisk: {table_risk} | "
        f"ShelfRisk: {shelf_risk} | "
        f"hand_shelf_collision_risk: {hand_shelf_risk} | "
        f"box_shelf_collision_risk: {box_shelf_risk} | "
        f"HandZ: {hand_pos[2]:.3f} | "
        f"BoxZ: {box_pos[2]:.3f} | "
        f"current_hand_target: {target_text} | "
        f"hand_target_error: {error_text} | "
        f"Gripper: {gripper_target:.3f} | "
        f"Grasp: {grasp_status['success']} "
        f"Contacts: {grasp_status['contact_count']} | "
        f"BoxSlipping: {box_slipping} | "
        f"grasp_assist_active: {grasp_assist_active} | "
        f"assist_force_norm: {assist_force_norm:.3f} | "
        f"(L={grasp_status['left_distance']:.3f}, "
        f"R={grasp_status['right_distance']:.3f}) | "
        f"Checks: {height_checks} | "
        f"TransferSuccess: {transfer_ok}"
    )


def run_simulation(
    headless=False,
    timeout=HEADLESS_TIMEOUT,
    print_progress=True,
    box_pos=None,
    payload_mass_kg=None,
    shelf_place_pos=None,
):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    if payload_mass_kg is not None:
        set_payload_mass_kg(model, payload_mass_kg)
    if box_pos is not None:
        set_freejoint_pose(model, data, "box_freejoint", np.asarray(box_pos), BOX_DEFAULT_QUAT)
    mujoco.mj_forward(model, data)
    shelf_place_pos = np.asarray(
        SHELF_PLACE_POS if shelf_place_pos is None else shelf_place_pos,
        dtype=float,
    )
    detector = ObjectDetector()
    initial_detection = detector.detect(model, data, shelf_position=shelf_place_pos)
    detected_shelf_pos = initial_detection["shelf_position"]
    shelf_walk_target = SHELF_WALK_TARGET.copy()
    shelf_walk_target[:2] += detected_shelf_pos[:2] - SHELF_PLACE_POS[:2]
    initial_box_pos = initial_detection["box_position"]
    table_walk_target = np.array(
        [
            initial_box_pos[0] - PICK_HAND_OFFSET_FROM_BASE[0],
            initial_box_pos[1] - PICK_HAND_OFFSET_FROM_BASE[1],
            1.0,
        ]
    )
    mpc = LocomotionMPC()
    arm_ik = ArmIKController()
    joint_addrs = build_joint_address_map(model)
    actuator_ids = build_actuator_id_map(model)
    metrics = MetricsLogger(
        payload_mass_kg=get_payload_mass_kg(model),
        use_ik_arm_control=USE_IK_ARM_CONTROL,
    )
    metrics.record_frame_processing_time(initial_detection["processing_time_s"])

    state = {
        "phase": TaskPhase.WALK_TO_TABLE,
        "phase_start_time": 0.0,
        "base_hold_xy": None,
        "grasp_reference_distance": None,
        "grasp_contact_start_time": None,
        "grasp_assist_active": False,
        "assist_force_norm": 0.0,
        "current_hand_target": None,
        "last_reported_phase": None,
        "last_print_time": -PRINT_INTERVAL,
        "summary_printed": False,
    }

    if print_progress:
        print("Starting pick, carry, and place simulation")

    def finish_if_done():
        if state["phase"] != TaskPhase.DONE or state["summary_printed"]:
            return
        box_pos = get_body_pos(model, data, "box")
        final_error = float(np.linalg.norm(box_pos - shelf_place_pos))
        metrics.set_final_box_error(final_error)
        metrics.mark_transfer_success(final_error < 0.10)
        state["summary_printed"] = True
        if print_progress:
            metrics.print_summary()

    def step_once(viewer=None):
        step_start = time.time()
        decision_start = time.perf_counter()
        dt = model.opt.timestep

        phase = state["phase"]
        base_pos = get_body_pos(model, data, "robot_base")
        shoulder_pos = get_body_pos(model, data, "right_upper_arm")
        hand_pos = get_body_pos(model, data, "right_hand")
        perception = detector.detect(model, data, shelf_position=shelf_place_pos)
        metrics.record_frame_processing_time(perception["processing_time_s"])
        box_pos = perception["box_position"]
        detected_shelf_pos = perception["shelf_position"]
        true_box_pos = get_body_pos(model, data, "box")
        grasp_status = detect_grasp(model, data)
        gripper_target = GRIPPER_OPEN
        navigation_target = None
        speed = 0.0
        data.xfrc_applied[:] = 0.0
        state["assist_force_norm"] = 0.0
        state["current_hand_target"] = None

        if phase != state["last_reported_phase"]:
            if print_progress:
                print(f"Task phase: {phase.name.lower()}")
            state["last_reported_phase"] = phase
            state["phase_start_time"] = data.time

        if phase == TaskPhase.WALK_TO_TABLE:
            arm_target = base_pos + STOW_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_STOW_JOINTS,
            )
            navigation_target = table_walk_target
            speed = move_base_toward(data, joint_addrs, mpc, base_pos, navigation_target, dt)
            if speed == 0.0:
                state["base_hold_xy"] = base_pos[:2].copy()
                state["phase"] = TaskPhase.REACH_ABOVE_BOX

        elif phase == TaskPhase.REACH_ABOVE_BOX:
            arm_target = box_pos + ABOVE_BOX_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_ABOVE_BOX_JOINTS,
            )
            if data.time - state["phase_start_time"] > 0.5:
                state["phase"] = TaskPhase.LOWER_TO_BOX

        elif phase == TaskPhase.LOWER_TO_BOX:
            arm_target = box_pos + PICK_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_PICK_JOINTS,
            )
            if data.time - state["phase_start_time"] > 0.5:
                state["phase"] = TaskPhase.GRASP_BOX

        elif phase == TaskPhase.GRASP_BOX:
            grasp_elapsed = data.time - state["phase_start_time"]
            gripper_target = GRIPPER_CLOSED * min(
                grasp_elapsed / GRIPPER_CLOSE_TIME, 1.0
            )
            arm_target = box_pos + PICK_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_PICK_JOINTS,
            )
            if grasp_status["success"]:
                if state["grasp_contact_start_time"] is None:
                    state["grasp_contact_start_time"] = data.time
            else:
                state["grasp_contact_start_time"] = None

            stable_grasp = (
                state["grasp_contact_start_time"] is not None
                and data.time - state["grasp_contact_start_time"] >= GRASP_STABLE_TIME
            )
            if grasp_status["success"] and not state["grasp_assist_active"]:
                state["grasp_assist_active"] = True
                metrics.mark_grasp_assist_used()
                metrics.mark_capture_success()
                state["grasp_reference_distance"] = float(
                    np.linalg.norm(hand_pos - true_box_pos)
                )

            if grasp_elapsed >= GRIPPER_CLOSE_TIME and (
                stable_grasp or state["grasp_assist_active"]
            ):
                state["phase"] = TaskPhase.LIFT_BOX

        elif phase == TaskPhase.LIFT_BOX:
            gripper_target = GRIPPER_CLOSED
            elapsed = data.time - state["phase_start_time"]
            arm_target = interpolate_target(
                box_pos + PICK_TARGET,
                box_pos + LIFT_TARGET,
                elapsed / LIFT_DURATION,
            )
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                interpolate_joints(
                    ARM_PICK_JOINTS, ARM_LIFT_JOINTS, elapsed / LIFT_DURATION
                ),
            )
            if elapsed > LIFT_DURATION:
                state["base_hold_xy"] = None
                state["phase"] = TaskPhase.BACK_AWAY_FROM_TABLE

        elif phase == TaskPhase.BACK_AWAY_FROM_TABLE:
            gripper_target = GRIPPER_CLOSED
            arm_target = base_pos + CARRY_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_CARRY_JOINTS,
            )
            navigation_target = BACK_AWAY_TARGET
            speed = move_base_toward(data, joint_addrs, mpc, base_pos, navigation_target, dt)
            if speed == 0.0:
                state["phase"] = TaskPhase.WALK_AROUND_TABLE

        elif phase == TaskPhase.WALK_AROUND_TABLE:
            gripper_target = GRIPPER_CLOSED
            arm_target = base_pos + CARRY_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_CARRY_JOINTS,
            )
            navigation_target = get_walk_around_target(base_pos, shelf_walk_target)
            speed = move_base_toward(data, joint_addrs, mpc, base_pos, navigation_target, dt)
            if speed == 0.0 and np.linalg.norm(base_pos[:2] - shelf_walk_target[:2]) < 0.08:
                state["phase"] = TaskPhase.WALK_TO_SHELF
            elif base_pos[0] > TABLE_X_RANGE[1] and abs(base_pos[1]) > TABLE_Y_RANGE[1]:
                state["phase"] = TaskPhase.WALK_TO_SHELF

        elif phase == TaskPhase.WALK_TO_SHELF:
            gripper_target = GRIPPER_CLOSED
            arm_target = base_pos + SHELF_APPROACH_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_SHELF_APPROACH_JOINTS,
            )
            navigation_target = shelf_walk_target
            speed = move_base_toward(data, joint_addrs, mpc, base_pos, navigation_target, dt)
            if speed == 0.0:
                state["base_hold_xy"] = base_pos[:2].copy()
                state["phase"] = TaskPhase.REACH_SHELF

        elif phase == TaskPhase.REACH_SHELF:
            gripper_target = GRIPPER_CLOSED
            elapsed = data.time - state["phase_start_time"]
            arm_target = base_pos + SHELF_APPROACH_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_SHELF_APPROACH_JOINTS,
            )
            if (
                elapsed > PLACE_DURATION
                and not shelf_hand_collision_risk(hand_pos)
                and not box_shelf_collision_risk(box_pos)
            ):
                state["phase"] = TaskPhase.PLACE_BOX

        elif phase == TaskPhase.PLACE_BOX:
            gripper_target = GRIPPER_CLOSED
            elapsed = data.time - state["phase_start_time"]
            arm_target = interpolate_target(
                base_pos + SHELF_APPROACH_TARGET,
                detected_shelf_pos + PLACE_TARGET,
                elapsed / PLACE_DURATION,
            )
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                interpolate_joints(
                    ARM_SHELF_APPROACH_JOINTS,
                    ARM_PLACE_JOINTS,
                    elapsed / PLACE_DURATION,
                ),
            )
            if elapsed > 0.45 and (box_contacts_shelf(model, data) or elapsed > 1.6):
                state["phase"] = TaskPhase.RELEASE_BOX

        elif phase == TaskPhase.RELEASE_BOX:
            gripper_target = GRIPPER_OPEN
            arm_target = detected_shelf_pos + PLACE_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_PLACE_JOINTS,
            )
            if data.time - state["phase_start_time"] > 0.35:
                state["grasp_assist_active"] = False
                state["phase"] = TaskPhase.DONE

        elif phase == TaskPhase.DONE:
            gripper_target = GRIPPER_OPEN
            state["grasp_assist_active"] = False
            arm_target = base_pos + STOW_TARGET
            state["current_hand_target"] = command_arm(
                data,
                actuator_ids,
                joint_addrs,
                arm_ik,
                shoulder_pos,
                arm_target,
                ARM_STOW_JOINTS,
            )

        command_gripper(data, actuator_ids, joint_addrs, gripper_target)

        if state["base_hold_xy"] is not None:
            hold_base(data, joint_addrs, state["base_hold_xy"])
        else:
            zero_base_velocity(data, joint_addrs)

        if state["grasp_assist_active"]:
            mujoco.mj_forward(model, data)
            if state["phase"] in (TaskPhase.PLACE_BOX, TaskPhase.RELEASE_BOX):
                assist_target = detected_shelf_pos
            else:
                assist_target = get_gripper_carry_target(model, data)
            state["assist_force_norm"] = apply_box_grasp_assist(
                model, data, assist_target
            )
            metrics.record_assist_force(state["assist_force_norm"])

        metrics.record_control_decision_time(time.perf_counter() - decision_start)
        metrics.record_speed(speed)

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

        new_hand_pos = get_body_pos(model, data, "right_hand")
        hand_target_error = None
        if state["current_hand_target"] is not None:
            hand_target_error = float(
                np.linalg.norm(new_hand_pos - state["current_hand_target"])
            )
            metrics.record_hand_target_error(hand_target_error)

        if print_progress and data.time - state["last_print_time"] >= PRINT_INTERVAL:
            new_box_pos = true_box_pos
            print_status(
                state["phase"],
                speed,
                base_pos,
                new_hand_pos,
                new_box_pos,
                gripper_target,
                grasp_status,
                navigation_target,
                table_collision_risk(base_pos),
                shelf_collision_risk(base_pos),
                shelf_hand_collision_risk(hand_pos),
                box_shelf_collision_risk(new_box_pos),
                get_height_checks(state["phase"], new_box_pos, shelf_place_pos),
                transfer_success(state["phase"], new_box_pos, shelf_place_pos),
                detect_box_slipping(
                    hand_pos, new_box_pos, state["grasp_reference_distance"]
                ),
                state["grasp_assist_active"],
                state["assist_force_norm"],
                state["current_hand_target"],
                hand_target_error,
            )
            state["last_print_time"] = data.time

        finish_if_done()

        if not headless:
            time.sleep(max(0.0, dt - (time.time() - step_start)))

        return state["phase"] == TaskPhase.DONE

    if headless:
        while data.time < timeout:
            if step_once():
                break
        if not state["summary_printed"]:
            box_pos = get_body_pos(model, data, "box")
            metrics.set_final_box_error(float(np.linalg.norm(box_pos - shelf_place_pos)))
            metrics.mark_transfer_success(False)
            if print_progress:
                print("Headless simulation timed out before DONE")
                metrics.print_summary()
        return metrics.as_dict()

    viewer_module = importlib.import_module("mujoco.viewer")

    with viewer_module.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_once(viewer)

    return metrics.as_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--timeout", type=float, default=HEADLESS_TIMEOUT)
    args = parser.parse_args()
    run_simulation(headless=args.headless, timeout=args.timeout)
    if args.headless:
        sys.exit(0)


if __name__ == "__main__":
    main()
