"""
Whole-body controller placeholder for Unitree G1 integration.

Separate from ``control.locomotion_mpc.LocomotionMPC`` and the sliding-base FSM:
fill in QP/MPC/WBC layers here without touching ``simulation/run_mujoco.py``
until parity testing is ready.
"""

from __future__ import annotations

from typing import Any


class WholeBodyController:
    """
    Hierarchical command composition (balance + manipulation).

    All methods return plain dicts so downstream code can evolve from
    simulation torques to hardware-specific message types.
    """

    def __init__(self) -> None:
        self._manipulation_weight = 1.0
        self._balance_weight = 1.0

    def compute_balance_command(
        self,
        robot_state: dict[str, Any],
        reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Stance / CoM / momentum regulation (placeholder).

        Args:
            robot_state: Estimated joint q/v, IMU, foot contacts, etc.
            reference: Optional CoM, swing foot, or nominal posture targets.
        """
        _ = (robot_state, reference)
        return {"mode": "balance", "tau": {}}

    def compute_manipulation_command(
        self,
        robot_state: dict[str, Any],
        task: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Task-space arm / EE commands mapped to joint torques (placeholder).

        Args:
            robot_state: Same as balance branch; may omit fields not needed.
            task: e.g. pose targets, impedance gains, grasp mode.
        """
        _ = (robot_state, task)
        return {"mode": "manipulation", "tau": {}}

    def combine_commands(
        self,
        balance_cmd: dict[str, Any],
        manipulation_cmd: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge balance and manipulation torques / priorities.

        Placeholder concatenates keys; a real implementation would respect
        nullspace projection, QP constraints, or motor torque limits.
        """
        tau_b = balance_cmd.get("tau") or {}
        tau_m = manipulation_cmd.get("tau") or {}
        keys = sorted(set(tau_b) | set(tau_m))
        combined_tau = {
            k: float(tau_b.get(k, 0.0)) * self._balance_weight
            + float(tau_m.get(k, 0.0)) * self._manipulation_weight
            for k in keys
        }
        return {
            "balance": balance_cmd.get("mode", "balance"),
            "manipulation": manipulation_cmd.get("mode", "manipulation"),
            "tau": combined_tau,
        }
