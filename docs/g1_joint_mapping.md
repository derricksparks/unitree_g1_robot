# G1 joint mapping

This repository will eventually swap the **sliding-base prototype** for a **full Unitree G1 MJCF**. MuJoCo model files use **opaque integer indices** (`jnt_dofadr`, `body_id`, …); your control stack usually reasons in **logical groups** (“left knee pitch”, “right wrist roll”). **Joint mapping** is the audited table that ties simulator names to those groups so controllers, IK, and whole-body QP do not hard-code brittle indices.

Until mapping matches a **verified** inspection of your real asset **on disk**, downstream control should assume **nothing** about DOF ordering.

## Why mapping is required

- **MJCF authoring variance**: exporters and Unitree revisions rename bodies/joints or split chains differently.
- **Stable control APIs**: MPC / WBC / IK want semantic keys (`leg_left/knee_pitch`) backed by lookups, not magic numbers.
- **Safety**: commanding the wrong actuator id is worse than refusing to command until YAML and inspection agree.

## How to run the inspection script

From the repository root:

```bash
python simulation/mujoco/g1/inspect_g1_model.py --path simulation/mujoco/g1/assets/<your_model.xml>
```

Optional JSON for diffing across model versions:

```bash
python simulation/mujoco/g1/inspect_g1_model.py \
  --path simulation/mujoco/g1/assets/<your_model.xml> \
  --output-json simulation/mujoco/g1/assets/inspect_report.json
```

The script prints **body count**, **joint count**, **actuator count**, every **joint name** with **MuJoCo type** and **`qpos` / `qvel` address spans**, plus all **body** and **actuator** names.

## How to fill `joint_map_template.yaml`

1. Copy `simulation/mujoco/g1/joint_map_template.yaml` to a tracked or local file (e.g. `joint_map_g1.yaml`; keep OEM secrets out of public repos per policy).

2. Use the JSON from `--output-json` (or stdout sections) and **copy literal `name` strings** from `"joints"` and `"actuators"` into:

   - `base.root_joint` — root linkage (floating base, fixed world joint placeholder, etc.).
   - `legs.left` / `legs.right` — ordered chains from proximal to distal.
   - `torso` — waist/spine grouping.
   - `arms.left` / `arms.right` — manipulation chains.
   - `hands.*` — finger / gripper **joints** and, if driven separately, **actuators**.
   - `ignored_joints` — anything in MJCF you deliberately do **not** command.

3. Re-run inspection whenever the MJCF checksum changes and **reconcile** the YAML.

## Expected joint roles (humanoid task context)

Rough expectations for biped pick-and-place; **exact names follow your MJCF**, not this table:

| Subsystem | Role |
|-----------|------|
| **Walking / locomotion** | Leg joints (typically 6×2 DOF plus contact scheduling); interacts with torso for momentum. |
| **Balance / posture** | Leg force distribution + torso + optionally arm nullspace **without violating** friction / CoM constraints. |
| **Reaching** | Arm (+ torso redundancy) IK toward object / shelf poses; wrist alignment for constrained approach. |
| **Grasping** | Hand / gripper joints or pinch actuators; contact stability eventually replaces prototype grasp-assist hacks. |

The **prototype** simulation uses only a planar base and simple arm joints; filling this template for **real G1** is orthogonal to `simulation/run_mujoco.py`.

## Control readiness

**No closed-loop locomotion, balance, IK, or hardware commands should be exercised on the full model until:**

1. `inspect_g1_model.py` has been run on the deployed MJCF/XML, and  
2. `joint_map_template.yaml` (or successor) lists every commanded joint/actuator **by exact name**, with spurious DOFs in `ignored_joints`.

Treat unmapped MJCF joints as **out of scope** for automation until explicitly classified.
