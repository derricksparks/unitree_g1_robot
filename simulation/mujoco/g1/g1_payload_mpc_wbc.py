"""Lightweight payload-aware MPC/WBC bridge for G1 carry demos.

This module is intentionally small and dependency-free.  It is not a full
torque-level whole-body controller; it provides the stabilizing layer we can
run today around the Lucky walker: payload-aware torso compensation, COM margin
targets, and explicit diagnostics for how much transport assistance remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PayloadMPCCommand:
    torso_pitch_bias_rad: float
    torso_roll_bias_rad: float
    max_step_assist_m: float
    target_com_margin_m: float
    predicted_margin_m: float
    payload_offset_xy_m: tuple[float, float]
    payload_distance_m: float
    controller_mode: str


class PayloadAwareMPCWBC:
    """MPC-style stabilizer for a payload carried in front of the torso."""

    def __init__(
        self,
        *,
        horizon_steps: int = 12,
        dt: float = 0.05,
        target_margin_m: float = 0.02,
        max_pitch_rad: float = 0.12,
        max_roll_rad: float = 0.08,
        max_assist_m: float = 0.006,
    ) -> None:
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.target_margin_m = float(target_margin_m)
        self.max_pitch_rad = float(max_pitch_rad)
        self.max_roll_rad = float(max_roll_rad)
        self.max_assist_m = float(max_assist_m)

    def solve(
        self,
        *,
        pelvis_xy: np.ndarray,
        payload_xyz: np.ndarray,
        com_margin_m: float,
        nominal_pitch_bias_rad: float = -0.08,
    ) -> PayloadMPCCommand:
        pelvis_xy = np.asarray(pelvis_xy, dtype=float)
        payload_xyz = np.asarray(payload_xyz, dtype=float)
        offset = payload_xyz[:2] - pelvis_xy
        payload_distance = float(np.linalg.norm(offset))

        # A forward payload moves the combined COM forward. Bias the torso back
        # smoothly, and add a small roll bias if the payload is laterally offset.
        forward_load = float(np.clip(offset[0], -0.35, 0.35))
        lateral_load = float(np.clip(offset[1], -0.25, 0.25))
        pitch_bias = nominal_pitch_bias_rad - 0.12 * max(0.0, forward_load)
        roll_bias = -0.08 * lateral_load

        # Predict a conservative one-step margin change from load distance.
        predicted_margin = float(com_margin_m - 0.35 * max(0.0, payload_distance - 0.12))
        margin_deficit = max(0.0, self.target_margin_m - predicted_margin)
        assist = min(self.max_assist_m, 0.04 * margin_deficit)

        return PayloadMPCCommand(
            torso_pitch_bias_rad=float(np.clip(pitch_bias, -self.max_pitch_rad, self.max_pitch_rad)),
            torso_roll_bias_rad=float(np.clip(roll_bias, -self.max_roll_rad, self.max_roll_rad)),
            max_step_assist_m=float(max(assist, self.max_assist_m if predicted_margin < -0.02 else 0.0)),
            target_com_margin_m=float(self.target_margin_m),
            predicted_margin_m=predicted_margin,
            payload_offset_xy_m=(float(offset[0]), float(offset[1])),
            payload_distance_m=payload_distance,
            controller_mode="payload_mpc_wbc_stabilized",
        )


def command_to_metrics(command: PayloadMPCCommand) -> dict[str, float | str | list[float] | bool]:
    return {
        "payload_mpc_wbc_enabled": True,
        "payload_mpc_controller_mode": command.controller_mode,
        "payload_mpc_torso_pitch_bias_rad": float(command.torso_pitch_bias_rad),
        "payload_mpc_torso_roll_bias_rad": float(command.torso_roll_bias_rad),
        "payload_mpc_max_step_assist_m": float(command.max_step_assist_m),
        "payload_mpc_target_com_margin_m": float(command.target_com_margin_m),
        "payload_mpc_predicted_margin_m": float(command.predicted_margin_m),
        "payload_mpc_payload_offset_xy_m": [float(command.payload_offset_xy_m[0]), float(command.payload_offset_xy_m[1])],
        "payload_mpc_payload_distance_m": float(command.payload_distance_m),
        "payload_mpc_balance_plan_valid": bool(command.predicted_margin_m > -0.20),
    }
