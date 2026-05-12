# Unitree G1 Pick-and-Place (MuJoCo Prototype)

MuJoCo simulation prototype for a table pick, carry, and shelf place task: finite-state planning, simplified base navigation, 2-DOF arm IK, and contact-gated grasp assist.

For methodology, phase descriptions, experimental protocol, and reported results, see **[docs/methodology_and_results.md](docs/methodology_and_results.md)**.

## Setup

1. **Python** — 3.10+ recommended (project tested with 3.12).

2. **Create a virtual environment** (recommended):

   ```bash
   cd /path/to/unitree_g1_pick_place
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Optional sanity check** (imports NumPy, MuJoCo, Gymnasium, PyTorch, Pinocchio):

   ```bash
   python scripts/check_install.py
   ```

5. **Run tests** (optional):

   ```bash
   pytest -q
   ```

   Some shells source ROS workspaces and export `PYTHONPATH`; that can pull extra pytest plugins (e.g. `launch_testing`) that are not bundled with this venv and may fail during collection. Either **unset ROS `PYTHONPATH` for this session** or run, for example, `env -u PYTHONPATH pytest -q`.

Task and validation thresholds are in `config/task_config.yaml`.

## Run the simulation

From the **repository root** (so paths like `simulation/mujoco/world.xml` resolve correctly):

### Interactive (passive viewer)

```bash
python simulation/run_mujoco.py
```

Uses `mujoco.viewer.launch_passive` and steps in real time (sleep aligned to `model.opt.timestep`).

### Headless

```bash
python simulation/run_mujoco.py --headless
```

Optional timeout (seconds, default 30):

```bash
python simulation/run_mujoco.py --headless --timeout 60
```

Headless mode runs stepping in a tight loop with no viewer; progress prints are on by default unless you call `run_simulation(..., print_progress=False)` from code.

## Run 50 randomized trials

Batch evaluation uses `scripts/run_trials.py`:

```bash
python scripts/run_trials.py --trials 50 --randomize --seed 42
```

- **`--trials`** — number of episodes (default in script is 20; use **`50`** for the documented experiment).
- **`--randomize`** — random box position, shelf goal, and payload mass (0.5–2.0 kg); omit for a fixed nominal configuration.
- **`--seed`** — `numpy.random` seed for reproducibility (e.g. **42**).

Without `--randomize`, trials use fixed defaults (`payload_mass_kg=2.0`, default box and shelf poses from `simulation/run_mujoco.py`).

## Where results are saved

All batch outputs go under **`results/`** at the repository root (`RESULTS_DIR` in `scripts/run_trials.py`):

| File pattern | Contents |
|--------------|----------|
| `results/randomized_trials_seed<seed>.csv` | Per-trial metrics: payload, successes, speeds, timings, final box error, hand-target errors (`none` if seed omitted). |
| `results/summary_seed<seed>.json` | Aggregated statistics and boolean flags (`speed_ok`, `capture_ok`, etc.) versus config-style thresholds. |

Example for seed 42:

- `results/randomized_trials_seed42.csv`
- `results/summary_seed42.json`

The script prints the same paths when it finishes (`saved_csv`, `saved_summary_json`).

## Roadmap Toward Full Unitree G1 Integration

The sliding-base prototype in `simulation/run_mujoco.py` remains the **validated** pick-and-place pipeline (FSM, metrics, trials). Parallel work prepares a **full humanoid geometry and control stack**:

| Artifact | Purpose |
|----------|---------|
| [docs/g1_integration_plan.md](docs/g1_integration_plan.md) | Limitations of the prototype, expected G1 DOF, balance/locomotion/manipulation risks, why grasp assist was used initially, phased next steps. |
| `simulation/mujoco/g1/load_g1_model.py` | **Optional** CLI to load `.xml`/`.mjcf` and print bodies, joints, and actuators (URDF placeholder). Not wired into the main sim. |
| `control/whole_body/whole_body_controller.py` | Placeholder **`WholeBodyController`** with `compute_balance_command()`, `compute_manipulation_command()`, and `combine_commands()` for future WBC/QP layering. |

When an official or approved G1 MJCF/URDF is available, add assets under something like `simulation/mujoco/g1/assets/` and extend the loader URDF branch; introduce a **separate** runner script only after regression against the baseline prototype metrics.

## Repository layout (short)

| Path | Role |
|------|------|
| `simulation/run_mujoco.py` | Main loop, `TaskPhase` FSM, grasp assist, metrics. |
| `simulation/mujoco/world.xml` | MuJoCo scene (linked assets). |
| `control/` | `LocomotionMPC`, `ArmIKController`, `whole_body/` (G1 placeholders). |
| `simulation/mujoco/g1/` | Future G1 MJCF/URDF + demos: **`run_g1_touch_box.py`** (authoritative pre-grasp contact), **`run_g1_reach_box.py`** (legacy IK/motion viz only); see `simulation/mujoco/g1/README.md`. |
| `perception/object_detector.py` | Ground-truth-style detector interface. |
| `evaluation/metrics_logger.py` | Timing and success metrics. |
| `scripts/run_trials.py` | CSV/JSON batch runs. |
