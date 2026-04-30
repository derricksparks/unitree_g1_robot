#!/usr/bin/env bash
# Train Unitree G1 whole-body box transport (RSL-RL / PPO).
# Usage:
#   ./scripts/train_box_transport.sh
#   ./scripts/train_box_transport.sh --gpu-ids None --env.scene.num-envs=64
# Override WANDB_MODE (default disabled) if you want Weights & Biases logging.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${WANDB_MODE:=disabled}"
export WANDB_MODE
export MUJOCO_GL="${MUJOCO_GL:-egl}"

VENV_PY="${ROOT}/.venv/bin/python"
if [[ -x "${VENV_PY}" ]]; then
  PYTHON="${VENV_PY}"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "${PYTHON}" scripts/train.py Unitree-G1-Box-Transport "$@"
