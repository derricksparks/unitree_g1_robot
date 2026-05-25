# Unitree G1 MuJoCo assets (parallel integration)

This directory holds **optional** model files for the future full G1 scene. The main pick-and-place prototype continues to use `simulation/mujoco/world.xml` and is **unchanged** by files here.

## Where to put models

Place G1 **MJCF** (or, once supported, **URDF**) and any small includes under:

`simulation/mujoco/g1/assets/`

Use a clear root filename, e.g. `g1.xml` or `scene_with_g1.xml`, and keep mesh paths relative to that file or this folder as MuJoCo expects.

## Supported formats

| Extension | Status |
|-----------|--------|
| `.xml`, `.mjcf` | **Supported now** via `mujoco.MjModel.from_xml_path` in `load_g1_model.py` |
| `.urdf` | **Planned** — loader raises `NotImplementedError` until a URDF→MJCF pipeline is wired |

## Inspecting a model

From the **repository root**:

```bash
python simulation/mujoco/g1/load_g1_model.py --path simulation/mujoco/g1/assets/<model_file>
```

Example using the simplified prototype scene (for a quick sanity check):

```bash
python simulation/mujoco/g1/load_g1_model.py --path simulation/mujoco/world.xml
```

## Git and large assets

Do **not** commit large binary meshes or heavy scan data unless your project policy requires it. Typical mesh interchange formats are ignored in the repo-root `.gitignore` under `simulation/mujoco/g1/assets/` (e.g. `.stl`, `.dae`, `.obj`, `.glb`, `.bin`). Prefer **Git LFS**, an internal artifact server, or documented download scripts for OEM assets.

## G1 palm touch-box demo

Locked-base scene with a red `box_geom` and palm IK toward the near face (see `g1_reach_box_scene.xml`).

- **`run_g1_touch_box.py`** is the **authoritative** pre-grasp contact demo (contact detection, penetration checks, richer summary).
- **`run_g1_reach_box.py`** is retained only as an earlier **motion / IK visualization** demo (cyclical reach), not for validating solid pre-grasp contact.

From the **repository root**, use the project venv so MuJoCo resolves reliably:

```bash
./.venv/bin/python simulation/mujoco/g1/run_g1_touch_box.py --headless --timeout 8
```

Or activate the venv first:

```bash
source .venv/bin/activate
python simulation/mujoco/g1/run_g1_touch_box.py --headless --timeout 8
```
