import numpy as np
import time


class LocomotionMPC:
    """
    Simplified MPC interface for G1 walking.
    Target: >= 0.5 m/s walking speed.
    """

    def __init__(self, max_speed=0.6):
        self.max_speed = max_speed
        self.target_speed = 0.5

    def compute_command(self, robot_state, target_position):
        base_pos = robot_state["base_pos"]

        error = target_position - base_pos
        distance = np.linalg.norm(error[:2])

        if distance < 0.05:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.0
            }

        direction = error[:2] / distance
        velocity = direction * self.target_speed

        return {
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "yaw_rate": 0.0
        }
