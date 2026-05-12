# Methodology and Results — MuJoCo Pick-and-Place Prototype

This note describes the current research-oriented MuJoCo pick-and-place prototype: a finite-state task planner coordinating simplified base navigation, arm inverse kinematics, and contact-gated grasp assistance inside a planar sliding-base manipulator scene.

See [README](../README.md) for environment setup and how to run single simulations and randomized trial batches.

## 1. System Overview

The prototype loads a MuJoCo scene (`simulation/mujoco/world.xml`) with a simplified robot body: a planar **sliding base** (`robot_slide_x`, `robot_slide_y`), a **two-joint arm** used for manipulation, and parallel-jaw gripper sliders. A **payload box** (free joint) rests on a table near a **shelf** target region.

Each simulation step executes:

1. **Perception**: `ObjectDetector` reports box and shelf poses (currently MuJoCo ground truth via the simulator state).
2. **Task planning**: a discrete finite-state machine (`TaskPhase`) selects arm targets, gripper commands, navigation goals, and when to activate grasp assist.
3. **Control**: locomotion produces horizontal base velocities; an IK solver converts Cartesian hand targets to shoulder and elbow pitches; gripper openness is commanded directly.
4. **Physics**: MuJoCo integrates contact dynamics; optional external forces stabilize the grasp after real finger-box contact.

Metrics (speed, timings, successes, placement error) are recorded by `evaluation.metrics_logger.MetricsLogger` for benchmarking and CSV/JSON aggregation in trial scripts.

## 2. Control Architecture

### FSM task planner

The high-level sequencer is implemented as an explicit `TaskPhase` enum and transition logic in `simulation/run_mujoco.py`. Phases impose timed segments (e.g. reach, lift, dwell), waypoint arrival (base within tolerance of navigation targets), and contact or collision-aware conditions before advancing.

### MPC-based base navigation

`control.locomotion_mpc.LocomotionMPC` implements a **simplified MPC-style interface** for base motion: given the current base pose and a 3D navigation target, it outputs constant horizontal velocity toward the goal at a **commanded 0.5 m/s** (capped by `max_speed`), with zero yaw rate in this prototype. The integrator applies `vx`, `vy` scaled by the simulation timestep to update sliding joint positions, with **table and shelf clearance heuristics** that nudge the base into a side lane when it would otherwise overlap restricted regions.

### IK-based arm target control

`control.arm_ik_controller.ArmIKController` performs **2-link planar IK** in the arm’s vertical plane (shoulder and elbow pitch) with damped least-squares steps and joint limits. Phases pass **world-frame hand targets** that are either base-relative (stow, carry, shelf approach) or **object/shelf-relative** (pre-grasp, pick, lift, place) so randomized box and shelf poses remain consistent with the kinematic targets.

### Contact-triggered grasp assist

After **both gripper fingers** register MuJoCo contact with the box, the controller may enable **bounded external forces** on the box (`xfrc_applied`): a spring-damper toward a carry or placement anchor, clipped by maximum force. This stabilizes payloads during lift, carry, approach, and place; it **does not teleport** the object and **only activates after real geometric contact**.

### Simulated perception interface

`perception.object_detector.ObjectDetector` exposes `detect(model, data, shelf_position)` returning box position, shelf position, and measured processing time. The default backend reads **exact body/geom poses** from MuJoCo—the same schema intended for eventual camera-backed estimates. Optional Gaussian noise on positions is supported for robustness experiments (default noise is zero).

## 3. Task Phases

| Phase | Role |
|--------|------|
| **WALK_TO_TABLE** | Navigate base toward pick stance in front of the detected box while arm holds stow posture. |
| **REACH_ABOVE_BOX** | IK to a pre-contact pose above the box. |
| **LOWER_TO_BOX** | IK to the grasp-ready pose aligned with pick offset. |
| **GRASP_BOX** | Ramp gripper closure; require sustained two-finger contact before proceeding; enable capture success and grasp assist when contact criteria are met. |
| **LIFT_BOX** | Interpolate hand target from pick height to lift height while gripper stays closed. |
| **BACK_AWAY_FROM_TABLE** | Retract base from the table while carrying (arm in carry frame). |
| **WALK_AROUND_TABLE** | Follow a side-lane / clear-x path to avoid table volume, then align toward shelf approach waypoint. |
| **WALK_TO_SHELF** | Navigate to shelf front while arm moves to shelf-approach configuration. |
| **REACH_SHELF** | Hold approach near shelf until timing and collision-risk checks allow lowering. |
| **PLACE_BOX** | Interpolate hand target toward detected shelf placement frame; transition on shelf contact or timeout. |
| **RELEASE_BOX** | Open gripper at place pose; disable grasp assist after a short dwell. |
| **DONE** | Task complete; stow arm; final metrics (e.g. transfer success by box-to-goal distance) are recorded. |

## 4. Development Requirements and Validation Table

Configuration lives in `config/task_config.yaml`. Automated trial scripts (`scripts/run_trials.py`) evaluate aggregate statistics against similar thresholds (e.g. success rates, speed, and latency caps).

| Requirement | Source / check | Target |
|-------------|----------------|--------|
| Minimum articulated DOF (design proxy) | `robot.min_dof` | ≥ 10 |
| Nominal payload (config) | `robot.payload_kg` | ≤ 2.0 kg |
| Locomotion command speed | `task.target_speed_mps` | ≥ 0.5 m/s |
| Capture success (batch) | `task.min_capture_success_rate` | ≥ 75% |
| Transfer success (batch) | `task.min_transfer_success_rate` | ≥ 80% |
| Max control decision time | `task.max_control_decision_time_s` | ≤ 0.5 s |
| Max perception frame time | `task.max_frame_processing_time_s` | ≤ 1.0 s |

Unit tests under `tests/` include a minimal config load check (`test_task_config_loads`).

## 5. Experimental Setup

- **50 randomized trials** — run with `python scripts/run_trials.py --trials 50 --randomize --seed 42` (see README).
- **Payload randomization** — uniform mass in **0.5–2.0 kg** per trial when `--randomize` is set (`PAYLOAD_MIN_KG`, `PAYLOAD_MAX_KG` in `run_trials.py`).
- **Randomized object / shelf perturbations** — box position perturbed in x/y around the default table pose; shelf placement goal perturbed in x/y around the nominal shelf point (see `BOX_*_PERTURB` and `SHELF_*_PERTURB` in `run_trials.py`).
- **Reproducibility** — pseudo-random stream from **`numpy.random.default_rng` with seed 42** when `--seed 42` is passed.

## 6. Results Summary

Reported prototype performance on the above randomized suite (seed **42**, 50 trials):

| Metric | Value |
|--------|--------|
| Capture success | **100%** |
| Transfer success | **100%** |
| Commanded navigation speed | **0.5 m/s** (nominal from locomotion module) |
| Max control decision time (approx.) | **0.00425 s** |
| Max frame processing time (approx.) | **0.000165 s** |
| Final placement error (approx.) | **0.01–0.04 m** (Euclidean box position vs. shelf goal) |

These timing numbers reflect Python control and synthetic perception on the development machine used for the study; they are useful for relative comparison (IK on/off, noise on/off) rather than as hard real-time guarantees on hardware.

## 7. Limitations

- **Simplified robot body** — not a full Unitree G1 MJCF/URDF; a sliding base and reduced arm/gripper abstraction.
- **Assisted grasp** — post-contact `xfrc_applied` stabilization, not a demonstration of pure passive force-closure alone.
- **Simulated perception** — detector reads MuJoCo **ground truth** by default; no real sensor pipeline.
- **No full G1 dynamics** — no legged whole-body model or full robot equation-of-motion fidelity in this scene.

## 8. Next Work

- Integrate full **Unitree G1 MJCF/URDF** and align joint/actuator naming with production models.
- Improve **whole-body balance** and locomotion (replace or extend the sliding-base abstraction).
- Replace assisted grasp with **physically robust gripper / contact control** (force limits, slip detection, passive-friendly policies).
- Add **vision-based detection** (replace or augment ground-truth `ObjectDetector`).
- Compare **MPC-only**, **IK + MPC**, and **RL-assisted grasping** baselines under shared metrics and randomization protocols.
