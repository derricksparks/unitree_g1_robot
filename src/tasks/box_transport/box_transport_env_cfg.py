"""Whole-body box transport task configuration."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.objects import (
  get_transport_box_cfg,
  get_transport_shelf_cfg,
  get_transport_table_cfg,
)
from src.tasks.box_transport import mdp
from src.tasks.box_transport.mdp import BoxTransportCommandCfg
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def make_g1_box_transport_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 whole-body box transport task."""

  cfg = unitree_g1_flat_env_cfg(play=play)

  cfg.scene.entities.update(
    {
      "table": get_transport_table_cfg(),
      "shelf": get_transport_shelf_cfg(),
      "box": get_transport_box_cfg(),
    }
  )
  cfg.scene.extent = 2.5
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 600

  feet_ground_contact = next(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name == "feet_ground_contact"
  )
  box_table_contact = ContactSensorCfg(
    name="box_table_contact",
    primary=ContactMatch(mode="body", pattern="box", entity="box"),
    secondary=ContactMatch(mode="body", pattern="transport_table", entity="table"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  box_shelf_contact = ContactSensorCfg(
    name="box_shelf_contact",
    primary=ContactMatch(mode="body", pattern="box", entity="box"),
    secondary=ContactMatch(mode="body", pattern="transport_shelf", entity="shelf"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    box_table_contact,
    box_shelf_contact,
  )

  cfg.commands = {
    "box_transport": BoxTransportCommandCfg(initial_stage=0 if not play else 4)
  }

  # Replace velocity-command observations with object-centric task observations.
  actor_terms = cfg.observations["actor"].terms
  actor_terms.pop("command", None)
  actor_terms.pop("phase", None)
  actor_terms["box_pos"] = ObservationTermCfg(func=mdp.box_position_b)
  actor_terms["shelf_pos"] = ObservationTermCfg(
    func=mdp.shelf_position_b, params={"command_name": "box_transport"}
  )
  actor_terms["box_to_shelf"] = ObservationTermCfg(
    func=mdp.box_to_shelf_b, params={"command_name": "box_transport"}
  )
  actor_terms["hand_to_box"] = ObservationTermCfg(func=mdp.hand_to_box_b)
  actor_terms["box_vel"] = ObservationTermCfg(func=mdp.box_velocity_b)
  actor_terms["box_lifted"] = ObservationTermCfg(
    func=mdp.box_lifted, params={"command_name": "box_transport"}
  )
  actor_terms["task_stage"] = ObservationTermCfg(
    func=mdp.task_stage, params={"command_name": "box_transport"}
  )

  critic_terms = cfg.observations["critic"].terms
  critic_terms.pop("command", None)
  critic_terms.pop("phase", None)
  critic_terms["box_pos"] = ObservationTermCfg(func=mdp.box_position_b)
  critic_terms["shelf_pos"] = ObservationTermCfg(
    func=mdp.shelf_position_b, params={"command_name": "box_transport"}
  )
  critic_terms["box_to_shelf"] = ObservationTermCfg(
    func=mdp.box_to_shelf_b, params={"command_name": "box_transport"}
  )
  critic_terms["hand_to_box"] = ObservationTermCfg(func=mdp.hand_to_box_b)
  critic_terms["box_vel"] = ObservationTermCfg(func=mdp.box_velocity_b)
  critic_terms["box_lifted"] = ObservationTermCfg(
    func=mdp.box_lifted, params={"command_name": "box_transport"}
  )
  critic_terms["task_stage"] = ObservationTermCfg(
    func=mdp.task_stage, params={"command_name": "box_transport"}
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  action_scale = dict(joint_pos_action.scale)
  for pattern, multiplier in {
    r".*_shoulder_pitch_joint": 0.78,
    r".*_shoulder_roll_joint": 0.78,
    r".*_shoulder_yaw_joint": 0.78,
    r".*_elbow_joint": 0.78,
    r".*_wrist_roll_joint": 0.62,
    r".*_wrist_pitch_joint": 0.62,
    r".*_wrist_yaw_joint": 0.62,
    "waist_pitch_joint": 0.68,
    "waist_roll_joint": 0.68,
  }.items():
    if pattern in action_scale:
      action_scale[pattern] *= multiplier
  joint_pos_action.scale = action_scale

  cfg.events.pop("push_robot", None)

  # Drop velocity-tracking rewards and keep the stability core.
  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "foot_gait",
    "stand_still",
  ):
    cfg.rewards.pop(name, None)

  if "soft_landing" in cfg.rewards:
    cfg.rewards["soft_landing"].params["command_name"] = None

  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_slip", None)

  cfg.rewards.update(
    {
      "base_distance_to_box": RewardTermCfg(
        func=mdp.base_distance_to_box,
        weight=0.72,
        params={
          "command_name": "box_transport",
          "target_distance": 0.45,
          "std": 0.38,
        },
      ),
      "reach_box": RewardTermCfg(
        func=mdp.reach_box,
        weight=1.55,
        params={
          "command_name": "box_transport",
          "std_pair": 0.38,
          "std_max_hand": 0.20,
        },
      ),
      "lift_progress": RewardTermCfg(
        func=mdp.lift_progress,
        weight=1.35,
        params={"command_name": "box_transport", "lift_cap": 0.12},
      ),
      "lift_box": RewardTermCfg(
        func=mdp.lift_box,
        weight=2.05,
        params={"command_name": "box_transport"},
      ),
      "keep_box_in_hands": RewardTermCfg(
        func=mdp.keep_box_in_hands,
        weight=1.1,
        params={"command_name": "box_transport"},
      ),
      "carry_box_to_shelf": RewardTermCfg(
        func=mdp.carry_box_to_shelf,
        weight=2.35,
        params={"command_name": "box_transport"},
      ),
      "place_box_on_shelf": RewardTermCfg(
        func=mdp.place_box_on_shelf,
        weight=5.25,
        params={"command_name": "box_transport"},
      ),
      "box_stable_on_shelf": RewardTermCfg(
        func=mdp.box_stable_on_shelf,
        weight=4.0,
        params={"command_name": "box_transport", "sensor_name": "box_shelf_contact"},
      ),
      "box_upright": RewardTermCfg(func=mdp.box_upright, weight=0.42),
      "foot_contact_balance": RewardTermCfg(
        func=mdp.foot_contact_balance,
        weight=0.38,
        params={"sensor_name": feet_ground_contact.name},
      ),
      "object_drop": RewardTermCfg(
        func=mdp.object_drop_penalty,
        weight=-0.72,
        params={"command_name": "box_transport", "drop_margin": 0.055},
      ),
    }
  )

  cfg.terminations.update(
    {
      "box_dropped": TerminationTermCfg(
        func=mdp.box_dropped_after_lift,
        params={
          "command_name": "box_transport",
          "margin": 0.2,
          "lift_above_spawn": 0.05,
          "min_stage": 1,
        },
      ),
      "box_too_far": TerminationTermCfg(
        func=mdp.box_too_far,
        params={"command_name": "box_transport", "max_distance": 3.75},
      ),
      "bad_torso_orientation": TerminationTermCfg(
        func=mdp.bad_torso_orientation,
        params={"threshold": 0.97},
      ),
    }
  )

  cfg.curriculum = {
    "box_transport_stage": CurriculumTermCfg(
      func=mdp.box_transport_stage,
      params={
        "command_name": "box_transport",
        "stages": [
          {"step": 0, "stage": 0},
          {"step": 5200 * 24, "stage": 1},
          {"step": 11000 * 24, "stage": 2},
          {"step": 17500 * 24, "stage": 3},
          {"step": 23500 * 24, "stage": 4},
        ],
      },
    )
  }

  cfg.episode_length_s = 18.5
  cfg.viewer.distance = 3.2

  if play:
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("base_com", None)
    cfg.curriculum = {}
    cfg.commands["box_transport"] = replace(cfg.commands["box_transport"], initial_stage=4)

  return cfg
