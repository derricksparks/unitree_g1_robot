# Lucky G1 Integration Notes

## External Repo

- Source: `https://github.com/luckyrobots/g1-manipulation-challenge`
- Local path: `external/g1-manipulation-challenge`
- Main files:
  - `run.py`: interactive MuJoCo runner combining walker, right-arm reacher, and grip.
  - `model_config.json`: joint order, default positions, action scales, policy dimensions, reacher config.
  - `scene.xml`: MuJoCo scene entry point.
  - `g1.xml`: G1 robot model.
  - `assets/`: mesh assets.

## Policy Files Found

- `external/g1-manipulation-challenge/walker.onnx`
- `external/g1-manipulation-challenge/walker.onnx.data`
- `external/g1-manipulation-challenge/right_reacher.onnx`
- `external/g1-manipulation-challenge/right_reacher.onnx.data`
- `external/g1-manipulation-challenge/croucher.onnx`
- `external/g1-manipulation-challenge/croucher.onnx.data`
- `external/g1-manipulation-challenge/rotator.onnx`
- `external/g1-manipulation-challenge/rotator.onnx.data`

## Walker Policy Interface

- Config path: `external/g1-manipulation-challenge/model_config.json`
- Walker path: `external/g1-manipulation-challenge/walker.onnx`
- Observation dimension: `99`
- Action dimension: `29`
- Observation layout used in `run.py`:
  - base linear velocity in body frame, 3 values
  - base angular velocity, 3 values
  - projected gravity, 3 values
  - joint position offsets, 29 values
  - joint velocities, 29 values
  - last action, 29 values
  - command velocity `[lin_x, lin_y, yaw]`, 3 values

## Joint Order

The walker action follows this 29-joint order:

1. `left_hip_pitch_joint`
2. `left_hip_roll_joint`
3. `left_hip_yaw_joint`
4. `left_knee_joint`
5. `left_ankle_pitch_joint`
6. `left_ankle_roll_joint`
7. `right_hip_pitch_joint`
8. `right_hip_roll_joint`
9. `right_hip_yaw_joint`
10. `right_knee_joint`
11. `right_ankle_pitch_joint`
12. `right_ankle_roll_joint`
13. `waist_yaw_joint`
14. `waist_roll_joint`
15. `waist_pitch_joint`
16. `left_shoulder_pitch_joint`
17. `left_shoulder_roll_joint`
18. `left_shoulder_yaw_joint`
19. `left_elbow_joint`
20. `left_wrist_roll_joint`
21. `left_wrist_pitch_joint`
22. `left_wrist_yaw_joint`
23. `right_shoulder_pitch_joint`
24. `right_shoulder_roll_joint`
25. `right_shoulder_yaw_joint`
26. `right_elbow_joint`
27. `right_wrist_roll_joint`
28. `right_wrist_pitch_joint`
29. `right_wrist_yaw_joint`

## Joint Ownership Plan

The Lucky walker emits all 29 body joints, including both arms. We should not let that policy own the arms in our warehouse manipulation stack.

Use from walker:

- Both legs: hip, knee, ankle joints.
- Waist joints initially, because the walker was trained with waist control and may need it for balance.

Mask from walker:

- `left_shoulder_*`, `left_elbow_joint`, `left_wrist_*`
- `right_shoulder_*`, `right_elbow_joint`, `right_wrist_*`

Keep under manipulation controller:

- Both arm chains and wrists.
- Dex3 hand/finger actuators.
- Any grasp/hold/place posture overlays.

This matches Lucky's own `run.py` pattern: it runs the walker every step, then overrides arm targets after walker inference. The left arm is reset to default. The right arm either holds a frozen target or receives an overlay from `right_reacher.onnx`.

## Arm Override and Grip Behavior

In `run.py`, the walker output is first converted to joint targets:

`target_pos = default_joint_pos + action * action_scales`

Then arm indices are overwritten:

- all arm joints are reset to default,
- the right arm can hold a frozen reached pose,
- in reach mode, `right_reacher.onnx` produces right-arm actions and overwrites the corresponding walker targets.

Grip control is independent of the walker. `run.py` caches right-hand finger actuators and writes fixed open/closed control targets, toggled by keyboard. This is the part we can replace with our Dex3 grasp logic later.

## Bridge Files Added

- `simulation/mujoco/g1/lucky_bridge/lucky_paths.py`: central paths to external Lucky assets.
- `simulation/mujoco/g1/lucky_bridge/lucky_policy_loader.py`: ONNX Runtime loader and config loading.
- `simulation/mujoco/g1/lucky_bridge/lucky_joint_map.py`: joint classification, action scaling, and MuJoCo actuator mapping.
- `simulation/mujoco/g1/run_g1_lucky_locomotion_validation.py`: first locomotion-only validation wrapper.

## First Validation Scope

The first wrapper uses Lucky's own `scene.xml` and disables/moves front obstacle contacts at runtime. It applies walker output only to legs and waist, leaving arms masked for later manipulation integration.

Validation metrics:

- `lucky_locomotion_enabled`
- `policy_loaded`
- `walker_policy_path`
- `observation_dim`
- `action_dim`
- `joint_mapping_ok`
- `robot_fell`
- `actual_visible_step_count`
- `pelvis_forward_progress_m`
- `average_forward_speed_mps`
- `actual_floor_contact_detected`
- `max_torso_pitch_rad`
- `max_torso_roll_rad`
- `success`

## Exact Files To Adapt Next

- Reuse Lucky's observation construction and walker ONNX inference.
- Keep our bridge modules as the stable integration boundary.
- After locomotion-only validation is visually acceptable, adapt our manipulation controller to write arm/hand targets after walker inference, following Lucky's arm override pattern.
- Do not merge Lucky's files into `simulation/`; keep the external repo under `external/` and reference it through `lucky_bridge`.
