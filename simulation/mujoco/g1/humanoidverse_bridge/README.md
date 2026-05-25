# HumanoidVerse Locomotion Bridge

This bridge swaps only locomotion, while leaving Lucky manipulation unchanged.

## What it does

- Uses HumanoidVerse-style `G1-12DoF` locomotion ownership:
  - `12DoF` mode: legs only (`hip/knee/ankle`)
  - `29DoF` mode: legs + waist (`hip/knee/ankle/waist`) with arm outputs ignored
  - no shoulder/elbow/wrist/hand ownership
- Keeps the existing Lucky table-pick state machine, grasp flow, scene setup, and carry logic.
- Adds a backend switch to `run_g1_lucky_table_pick_locked_carry.py`:
  - `--locomotion-backend lucky` (default)
  - `--locomotion-backend humanoidverse --humanoidverse-policy-onnx <path>`

## Bridge modules

- `humanoidverse_g1_locomotion.py`
  - defines G1-12DoF locomotion joint ownership
  - builds 48-dim actor observation:
    - `base_lin_vel(3), base_ang_vel(3), projected_gravity(3), command_lin_vel(2), command_ang_vel(1), dof_pos(12), dof_vel(12), actions(12)`
- `humanoidverse_paths.py`
  - validates local HumanoidVerse repo/config paths
- `run_humanoidverse_bridge_probe.py`
  - verifies ownership split and ONNX compatibility

## Quick checks

Ownership/checkpoint probe:

```bash
./.venv/bin/python simulation/mujoco/g1/humanoidverse_bridge/run_humanoidverse_bridge_probe.py \
  --policy-onnx /absolute/path/to/humanoidverse_g1_12dof.onnx
```

Run table pick-to-turn using HumanoidVerse locomotion:

```bash
./.venv/bin/python simulation/mujoco/g1/run_g1_lucky_table_pick_locked_carry.py \
  --headless \
  --stop-after-turning-point \
  --locomotion-backend humanoidverse \
  --humanoidverse-policy-onnx /absolute/path/to/humanoidverse_g1_12dof.onnx
```

## Current blocker

HumanoidVerse upstream repository currently does not publish ready-to-download G1 locomotion checkpoints in Releases/artifacts.  
The bridge is ready for integration once a compatible exported ONNX checkpoint is provided:
- `12DoF`: `input_dim=48`, `output_dim=12`
- `29DoF`: `input_dim=99`, `output_dim=29` (bridge keeps arm ownership masked out)

