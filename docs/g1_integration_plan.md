# Unitree G1 Integration Plan

This document describes the **parallel integration path** toward a more realistic Unitree G1 model and controller stack. It does **not** replace the current MuJoCo prototype in `simulation/run_mujoco.py`; that pipeline remains the reference implementation for pick-and-place logic, metrics, and trials.

Related code stubs:

- `simulation/mujoco/g1/load_g1_model.py` — optional MJCF inspection / future URDF hook.
- `control/whole_body/whole_body_controller.py` — placeholder whole-body command composition.

---

## Current simplified prototype limitations

The existing scene uses a **planar sliding base** (`robot_slide_x`, `robot_slide_y`) instead of articulated legs, a **reduced 2-DOF arm** with IK in a vertical plane, and **direct actuator / kinematic hacks** suited to task ordering rather than legged dynamics. Clearance logic nudges the base around table/shelf volumes; there is **no CoM**, **no foot placement**, and **no full-body inertia** coupling locomotion and manipulation.

Perception defaults to **MuJoCo ground truth**. Grasping uses **contact-gated external forces** on the payload after fingers touch the box, which stabilizes lift/carry/place under this abstraction but does not certify passive force closure on hardware.

---

## Target Unitree G1 capabilities

The integration target is a **full (or warehouse-relevant) DOF specification** suitable for simultaneous:

- **Bipedal/locomotion** — stepping, disturbance rejection, torso regulation.
- **Dual-arm manipulation** — reaching, constrained motion near the environment.
- **Hand / end-effector control** — firm contact without relying on simulator-only assist (long term).

Simulation should load **Authoritative kinematic/dynamic parameters** from **MJCF or URDF** aligned with Unitree releases, plus calibrated collision geometry for shelving and payloads.

Hardware-facing goals include the same task phases as the prototype FSM (`WALK_*`, grasp, lift, place) but backed by **feasibility checks** (reachability under balance, wrench limits, MPC/whole-body QP constraints).

---

## Expected degrees of freedom (DOF)

Published G1 specs vary by sub-version; plan for roughly:

- **Legs**: 6 DOF per leg (×2) → **12** locomotion joints.
- **Waist / torso**: typically **1–3** joints depending on variant.
- **Arms**: order of **7 per arm** → **14** shoulder–wrist articulations.
- **Hands**: tens of joints for dexterous hands, or fewer for pragmatic grippers—**budget explicitly** once the MJCF asset is pinned.

Integration planning should assume **≥ 29** actuated axes for torso + legs + dual arms before hands, then add hand DOF on top. Exact counts must match the **chosen MJCF/URDF**, not this estimate alone.

---

## Whole-body balance requirements

A legged robot must maintain **support polygon / centroidal dynamics** feasibility while manipulating:

- Track or regulate **linear/angular momentum / CoM trajectory** compatible with footholds.
- Respect **contact friction cones** at feet (and intermittently hands if allowed).
- Avoid **singular or collapsed leg postures** during reaches.
- Separate **slow manipulation** torque budgets from **fast balance** corrections (typically via hierarchical QP, MPC + WBC, or similar).

The placeholder `WholeBodyController` reserves APIs for **`compute_balance_command`** vs **`compute_manipulation_command`** and **`combine_commands`** so the codebase can absorb a hierarchal stack without rewriting the task FSM first.

---

## Locomotion challenges

- **Nonholonomic-ish stepping** coupled to torso motion and arm swings.
- **Terrain and collision** near table legs and shelves; foot placement preview.
- **Sim-to-real** on contact timing, actuator delays, and state estimation drift.
- **Speed vs stability** trading against the prototype’s nominal **0.5 m/s**-class commands (walking will not map 1:1 from sliding-base velocity).

---

## Manipulation challenges

- **Kinematic redundancy** across arms and torso; task-space IK must respect joint limits and self-collision.
- **Dual-hand** or single-hand payloads with inertia reflected at the EE.
- **Shelf alignment** requires precise approach normals and guarded motion to avoid chatter.
- **Sensor noise** replaces ground-truth poses; IK and grasp timing must degrade gracefully.

---

## Contact stability issues

- **Friction, compliance, and impact** differ from rigid MuJoCo defaults.
- **Grasp slips** manifest as wrench errors at hands; mitigation is **real contact control**, not prolonged `xfrc_applied` assist.
- **Foot slip** during reaching must be detected and recovered through stepping or torso adjustment.
- **Whole-body juggling** — hand contact forces perturb CoM and must appear in balance layer models.

---

## Why grasp assist was used in the prototype stage

Grasp assist in `run_mujoco.py` does three practical things **for simulation-only task validation**:

1. **Decouples** “does the planner sequence work?” from “do we yet have realistic finger friction and compliance?” — enabling end-to-end FSM tuning.
2. **Reduces brittle failures** (payload drift during base motion) caused by simplified grippers and no tactile/force loop.
3. **Keeps benchmarking stable** — success rates and timing metrics reflect sequencing and IK/MPC interplay rather than stochastic slip on every trial.

Physical G1 integration should **phase out assist** and replace it with measurable contact objectives (squeeze force profiles, slip detection, re-grasp behaviors).

---

## Next implementation steps (non-breaking)

1. Acquire and version-pin **official or community G1 MJCF/URDF**; place under `simulation/mujoco/g1/assets/` once licensing permits.
2. Extend `load_g1_model.py` URDF branch when a repeatable compile path exists (MuJoCo `compile` toolchain or Pinocchio + export).
3. Implement `WholeBodyController` internals (stub → QP/MPC stubs → hardware).
4. Add a **parallel** simulator entry script (future) that reuses perception/metrics interfaces but swaps the scene and controllers—**leave** `run_mujoco.py` as regression baseline until parity is demonstrated.
