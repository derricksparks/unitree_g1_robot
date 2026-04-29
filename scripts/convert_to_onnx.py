"""
G1 Walking Controller - Using YOUR Trained Policy

This module provides a clean interface for G1 walking using your
pre-trained locomotion policy (model_2999.pt).
"""

import numpy as np
import torch
import mujoco
import os
from pathlib import Path


class G1Walker:
    """
    G1 Humanoid Walking Controller using YOUR trained RL policy.

    Usage:
        walker = G1Walker()
        walker.reset()

        while running:
            walker.set_command(forward=0.5, lateral=0.0, yaw=0.0)
            walker.step()
            pos, vel = walker.get_state()
    """

    def __init__(self, model=None, data=None):
        """
        Initialize the walker.

        Args:
            model: Optional MuJoCo model (will load default if None)
            data: Optional MuJoCo data (will create if None)
        """
        # Paths
        self.base_path = Path(__file__).parent.parent.parent
        self.your_policy_path = Path("/home/drake/unitree_rl_mjlab/scripts/wandb/model_2999.pt")
        self.robot_xml = self.base_path / "unitree_rl_gym/resources/robots/g1_description/scene.xml"

        # Fallback to mujoco_menagerie if unitree model not found
        if not self.robot_xml.exists():
            self.robot_xml = self.base_path / "mujoco_menagerie/unitree_g1/scene.xml"

        # Load model if not provided
        if model is None:
            self.model = mujoco.MjModel.from_xml_path(str(self.robot_xml))
            self.data = mujoco.MjData(self.model)
            self._owns_model = True
        else:
            self.model = model
            self.data = data
            self._owns_model = False

        # Set simulation timestep
        self.model.opt.timestep = 0.002

        # Control parameters (from Unitree config)
        self.control_decimation = 10  # Policy runs at 50Hz
        self.num_actions = 12        # We only use 12 leg joints
        self.num_obs = 47            # Walker builds 47-dim observation

        # PD gains (tuned by Unitree)
        self.kps = np.array([100, 100, 100, 150, 40, 40,   # Left leg
                            100, 100, 100, 150, 40, 40],   # Right leg
                            dtype=np.float32)
        self.kds = np.array([2, 2, 2, 4, 2, 2,
                            2, 2, 2, 4, 2, 2], dtype=np.float32)

        # Default standing pose
        self.default_angles = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # Left leg
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0   # Right leg
        ], dtype=np.float32)

        # Scaling factors
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)

        # Command velocity [forward, lateral, yaw_rate]
        self.cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        # State variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        self.start_pos = np.zeros(3)

        # ================================================================
        # LOAD YOUR TRAINED POLICY
        # ================================================================
        if not self.your_policy_path.exists():
            raise FileNotFoundError(f"Your policy not found at {self.your_policy_path}")

        print(f"[INFO] Loading YOUR trained policy from {self.your_policy_path}...")
        ckpt = torch.load(str(self.your_policy_path), map_location='cpu', weights_only=False)

        # Extract actor state dict
        actor_state = ckpt['actor_state_dict']

        # Rebuild the MLP from state dict
        # Architecture: 98 → 512 → 256 → 128 → 29
        layers = []

        for i in [0, 2, 4, 6]:
            w_key = f'mlp.{i}.weight'
            b_key = f'mlp.{i}.bias'
            layer = torch.nn.Linear(actor_state[w_key].shape[1], actor_state[w_key].shape[0])
            layer.weight.data = actor_state[w_key]
            layer.bias.data = actor_state[b_key]
            layers.append(layer)

        # Build the network with ELU activations
        self.your_policy = torch.nn.Sequential(
            layers[0], torch.nn.ELU(alpha=1.0),
            layers[1], torch.nn.ELU(alpha=1.0),
            layers[2], torch.nn.ELU(alpha=1.0),
            layers[3]
        )
        self.your_policy.eval()

        print(f"  ✓ Policy loaded: 98-dim input → 29-dim output")
        print(f"  ✓ Using first 12 output dimensions for leg control")

    def reset(self):
        """Reset the robot to standing position."""
        mujoco.mj_resetData(self.model, self.data)

        # Set initial joint positions to default standing pose
        if self.model.nq > 7 + self.num_actions:
            self.data.qpos[7:7 + self.num_actions] = self.default_angles

        mujoco.mj_forward(self.model, self.data)

        # Let it settle
        for _ in range(100):
            self._apply_pd_control()
            mujoco.mj_step(self.model, self.data)

        self.start_pos = self.data.qpos[:3].copy()
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()

        return self.get_state()

    def set_command(self, forward=1.0, lateral=0.0, yaw=0.0):
        """
        Set velocity command.

        Args:
            forward: Forward velocity (m/s), positive = forward
            lateral: Lateral velocity (m/s), positive = left
            yaw: Yaw rate (rad/s), positive = counter-clockwise
        """
        self.cmd = np.array([forward, lateral, yaw], dtype=np.float32)

    def step(self):
        """
        Execute one control step (multiple simulation steps).

        Returns:
            dict with state info
        """
        for _ in range(self.control_decimation):
            self._apply_pd_control()
            mujoco.mj_step(self.model, self.data)
            self.counter += 1

        # Update policy at control frequency
        self._update_policy()

        return self.get_state()

    def get_state(self):
        """
        Get current robot state.

        Returns:
            dict with position, velocity, orientation, etc.
        """
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        vel = self.data.qvel[:3].copy()

        # Compute pitch and roll
        w, x, y, z = quat
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

        return {
            'position': pos,
            'quaternion': quat,
            'velocity': vel,
            'pitch': pitch,
            'roll': roll,
            'distance': pos[0] - self.start_pos[0],
            'height': pos[2],
            'is_fallen': pos[2] < 0.5 or abs(pitch) > 1.0 or abs(roll) > 1.0
        }

    def _get_gravity_orientation(self, quat):
        """Convert quaternion to gravity vector in body frame."""
        qw, qx, qy, qz = quat
        grav = np.zeros(3)
        grav[0] = 2 * (-qz * qx + qw * qy)
        grav[1] = -2 * (qz * qy + qw * qx)
        grav[2] = 1 - 2 * (qw * qw + qz * qz)
        return grav

    def _apply_pd_control(self):
        """Apply PD control to track target joint positions."""
        q = self.data.qpos[7:7 + self.num_actions]
        dq = self.data.qvel[6:6 + self.num_actions]

        tau = self.kps * (self.target_dof_pos - q) + self.kds * (0 - dq)
        self.data.ctrl[:self.num_actions] = tau

    def _update_policy(self):
        """Run YOUR policy inference to get new action."""
        # Get sensor data
        qj = self.data.qpos[7:7 + self.num_actions]
        dqj = self.data.qvel[6:6 + self.num_actions]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        # Scale observations
        qj_scaled = (qj - self.default_angles) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        gravity_orientation = self._get_gravity_orientation(quat)
        omega_scaled = omega * self.ang_vel_scale

        # Gait phase
        period = 0.8
        count = self.counter * self.model.opt.timestep
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # Build the 47-dim observation that the walker constructs
        self.obs[:3] = omega_scaled
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:9 + self.num_actions] = qj_scaled
        self.obs[9 + self.num_actions:9 + 2 * self.num_actions] = dqj_scaled
        self.obs[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = self.action
        self.obs[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = [sin_phase, cos_phase]

        # ================================================================
        # USE YOUR TRAINED POLICY
        # Your policy expects 98-dim input, we have 47-dim
        # Pad with zeros for the remaining dimensions
        # ================================================================
        with torch.no_grad():
            # Build 98-dim observation by padding our 47-dim with zeros
            obs_98 = np.zeros(98, dtype=np.float32)
            obs_98[:47] = self.obs  # Copy first 47 dimensions

            obs_tensor = torch.from_numpy(obs_98).unsqueeze(0).float()

            # Your policy outputs 29-dim (all joints)
            full_action = self.your_policy(obs_tensor).numpy().squeeze()

            # Extract only the 12 leg joint actions (first 12 of 29)
            self.action = full_action[:12].astype(np.float32)

        # Convert action to joint targets
        self.target_dof_pos = self.action * self.action_scale + self.default_angles


# Demo script
def demo():
    """Run a simple walking demo."""
    import mujoco.viewer
    import time

    print("=" * 60)
    print("G1 Walking Demo - Using YOUR Trained Policy")
    print("=" * 60)

    walker = G1Walker()
    walker.reset()
    walker.set_command(forward=1.0, lateral=0.0, yaw=0.0)

    print(f"Command: forward=1.0 m/s")
    input("Press ENTER to start...")

    with mujoco.viewer.launch_passive(walker.model, walker.data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        start_time = time.time()
        last_print = 0

        while viewer.is_running() and time.time() - start_time < 60:
            state = walker.step()

            # Update camera
            viewer.cam.lookat[:] = [state['position'][0], state['position'][1], 0.8]

            # Print status
            elapsed = time.time() - start_time
            if elapsed - last_print > 1.0:
                speed = state['distance'] / elapsed if elapsed > 0.5 else 0
                print(f"t={elapsed:5.1f}s | X={state['position'][0]:+6.2f} Y={state['position'][1]:+6.2f} | "
                      f"d={state['distance']:+5.2f}m | v={speed:.2f}m/s")
                last_print = elapsed

                if state['is_fallen']:
                    print("Robot fell!")
                    break

            viewer.sync()
            time.sleep(0.001)

        elapsed = time.time() - start_time
        print("-" * 60)
        print(
            f"Final: {
                state['distance']:.2f}m in {
                elapsed:.1f}s = {
                state['distance'] /
                elapsed:.2f} m/s")


if __name__ == "__main__":
    demo()
