import numpy as np


class ArmIKController:
    """
    Simple 2-link planar IK for the placeholder G1 arm.
    This controls shoulder + elbow pitch in the x/z plane.
    """

    def __init__(self):
        self.upper_link = np.array([0.3, -0.15], dtype=float)
        self.forearm_link = np.array([0.3, -0.01], dtype=float)
        self.max_iterations = 24
        self.damping = 1e-3

    def forward_kinematics(self, joints):
        shoulder, elbow = joints
        return (
            self._rotate_y_planar(self.upper_link, shoulder)
            + self._rotate_y_planar(self.forearm_link, shoulder + elbow)
        )

    def compute_joint_targets(self, shoulder_pos, target_pos, current_joints=None):
        rel = target_pos - shoulder_pos
        target = np.array([rel[0], rel[2]], dtype=float)

        if current_joints is None:
            joints = np.zeros(2, dtype=float)
        else:
            joints = np.asarray(current_joints, dtype=float).copy()

        for _ in range(self.max_iterations):
            error = target - self.forward_kinematics(joints)
            if np.linalg.norm(error) < 1e-4:
                break

            jacobian = self._numerical_jacobian(joints)
            lhs = jacobian @ jacobian.T + self.damping * np.eye(2)
            step = jacobian.T @ np.linalg.solve(lhs, error)
            joints += np.clip(step, -0.15, 0.15)
            joints = np.clip(joints, -2.0, 2.0)

        return {
            "shoulder": float(joints[0]),
            "elbow": float(joints[1])
        }

    def _numerical_jacobian(self, joints):
        eps = 1e-5
        base = self.forward_kinematics(joints)
        jacobian = np.zeros((2, 2), dtype=float)
        for idx in range(2):
            perturbed = joints.copy()
            perturbed[idx] += eps
            jacobian[:, idx] = (self.forward_kinematics(perturbed) - base) / eps
        return jacobian

    def _rotate_y_planar(self, vector, angle):
        x, z = vector
        return np.array(
            [
                np.cos(angle) * x + np.sin(angle) * z,
                -np.sin(angle) * x + np.cos(angle) * z,
            ],
            dtype=float,
        )
