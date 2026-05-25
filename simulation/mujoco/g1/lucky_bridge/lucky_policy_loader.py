"""Policy/config loading for Lucky Robots ONNX assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LuckyPolicyLoadError(RuntimeError):
    """Raised when a Lucky policy/config cannot be loaded."""


class ONNXPolicy:
    """Small CPU-only ONNX policy wrapper."""

    def __init__(self, model_path: str | Path):
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - environment dependent
            raise LuckyPolicyLoadError("onnxruntime is required to load Lucky policies") from exc

        path = Path(model_path)
        if not path.is_file():
            raise LuckyPolicyLoadError(f"Policy file not found: {path}")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_dim = int(self.session.get_inputs()[0].shape[-1])
        self.output_dim = int(self.session.get_outputs()[0].shape[-1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = self.session.run([self.output_name], {self.input_name: arr})[0]
        return np.asarray(out[0], dtype=np.float32)


def load_lucky_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise LuckyPolicyLoadError(f"Lucky config not found: {path}")
    return json.loads(path.read_text())


def load_walker_policy(policy_path: str | Path) -> ONNXPolicy:
    return ONNXPolicy(policy_path)


def load_right_reacher_policy(policy_path: str | Path) -> ONNXPolicy:
    return ONNXPolicy(policy_path)
