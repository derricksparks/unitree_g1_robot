#!/usr/bin/env python3
"""
Simplified G1 Mixer - Direct DDS Passthrough with Arm Override
"""

import mujoco
import numpy as np
import time
import sys
import select
import termios
import tty
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

# Joint indices
L_ARM_INDICES = [15, 16, 17, 18, 19, 20, 21]
R_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]
LEG_INDICES = list(range(0, 12))
WAIST_INDICES = list(range(12, 15))


class SimpleMixer:
    def __init__(self, xml_path):
        print("[INFO] Loading MuJoCo model...")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Setup IK
        self.l_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
        self.r_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")

        l_names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                   "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        r_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                   "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

        self.l_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in l_names]
        self.r_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in r_names]

        # DDS Setup - CRITICAL: Listen and publish to SAME topic
        ChannelFactoryInitialize(0, "lo")

        print("[INFO] Subscribing to rt/lowcmd...")
        self.sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub.Init()

        print("[INFO] Publishing to rt/lowcmd...")
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        # State
        self.mode = "WALK"  # "WALK" or "MANIPULATE"
        self.gripper_closed = False
        self.running = True
        self.step = 0.01

        # IK targets
        mujoco.mj_forward(self.model, self.data)
        self.l_target = self.data.site_xpos[self.l_ee_id].copy()
        self.r_target = self.data.site_xpos[self.r_ee_id].copy()

        # Standing leg pose
        self.stand_legs = np.array([
            0.0, 0.1, 0.0, -0.3, 0.0, -0.3,
            0.0, -0.1, 0.0, -0.3, 0.0, -0.3
        ])
        self.stand_waist = np.array([0.0, 0.0, 0.0])

        # Home arms
        self.home_left = np.array([0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0])
        self.home_right = np.array([0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0])

        # Last received RL command
        self.last_rl_cmd = None
        self.frame_count = 0

        print("\n" + "=" * 50)
        print("  SIMPLE G1 MIXER")
        print("=" * 50)
        print("\nCONTROLS (press in terminal):")
        print("  m - Toggle WALK / MANIPULATE mode")
        print("  g - Toggle gripper (MANIPULATE mode)")
        print("  r - Reset arm targets")
        print("  q - Quit")
        print("\nARM MOVEMENT (MANIPULATE mode):")
        print("  w/s - Left arm forward/back")
        print("  a/d - Left arm left/right")
        print("  z/x - Left arm up/down")
        print("  i/k - Right arm forward/back")
        print("  j/l - Right arm left/right")
        print("  n/m - Right arm up/down")
        print("\n[READY] Current mode: WALK")
        print("=" * 50 + "\n")

    def get_key(self):
        """Non-blocking keyboard input"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def solve_ik(self):
        """Solve IK for arms"""
        mujoco.mj_forward(self.model, self.data)

        jac_l = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_l, None, self.l_ee_id)
        mujoco.mj_jacSite(self.model, self.data, jac_r, None, self.r_ee_id)

        mask_l = np.zeros(self.model.nv)
        mask_r = np.zeros(self.model.nv)
        for idx in self.l_adr:
            mask_l[idx - 1] = 1.0
        for idx in self.r_adr:
            mask_r[idx - 1] = 1.0

        jac_combined = np.vstack((jac_l * mask_l, jac_r * mask_r))
        err_l = (self.l_target - self.data.site_xpos[self.l_ee_id]) * 0.3
        err_r = (self.r_target - self.data.site_xpos[self.r_ee_id]) * 0.3
        err_combined = np.concatenate((err_l, err_r))

        dq = jac_combined.T @ np.linalg.solve(
            jac_combined @ jac_combined.T + 0.1**2 * np.eye(6),
            err_combined
        )

        mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)

        # Enforce limits
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                l, h = self.model.jnt_range[i]
                addr = self.model.jnt_qposadr[i]
                self.data.qpos[addr] = np.clip(self.data.qpos[addr], l, h)

        l_q = [self.data.qpos[idx] for idx in self.l_adr]
        r_q = [self.data.qpos[idx] for idx in self.r_adr]

        # Gripper
        gripper_val = 1.2 if self.gripper_closed else -1.2
        l_q[4] = gripper_val
        r_q[4] = gripper_val

        return l_q, r_q

    def handle_keyboard(self):
        """Process keyboard input"""
        k = self.get_key()
        if k is None:
            return

        if k == 'm':
            self.mode = "MANIPULATE" if self.mode == "WALK" else "WALK"
            print(f"\n>>> MODE SWITCHED TO: {self.mode} <<<")
            if self.mode == "WALK":
                print("    Legs: RL control | Arms: Home position")
            else:
                print("    Legs: LOCKED standing | Arms: IK control")

        elif k == 'g' and self.mode == "MANIPULATE":
            self.gripper_closed = not self.gripper_closed
            print(f"\n>>> GRIPPER: {'CLOSED' if self.gripper_closed else 'OPEN'} <<<")

        elif k == 'r' and self.mode == "MANIPULATE":
            mujoco.mj_forward(self.model, self.data)
            self.l_target = self.data.site_xpos[self.l_ee_id].copy()
            self.r_target = self.data.site_xpos[self.r_ee_id].copy()
            print("\n>>> ARM TARGETS RESET <<<")

        elif k == 'q':
            self.running = False
            print("\n>>> QUITTING <<<")

        elif self.mode == "MANIPULATE":
            # Left arm
            if k == 'w':
                self.l_target[0] += self.step
            elif k == 's':
                self.l_target[0] -= self.step
            elif k == 'a':
                self.l_target[1] += self.step
            elif k == 'd':
                self.l_target[1] -= self.step
            elif k == 'z':
                self.l_target[2] += self.step
            elif k == 'x':
                self.l_target[2] -= self.step
            # Right arm
            elif k == 'i':
                self.r_target[0] += self.step
            elif k == 'k':
                self.r_target[0] -= self.step
            elif k == 'j':
                self.r_target[1] += self.step
            elif k == 'l':
                self.r_target[1] -= self.step
            elif k == 'n':
                self.r_target[2] += self.step
            elif k == 'm':
                self.r_target[2] -= self.step

    def run(self):
        """Main loop"""
        # Setup terminal for raw input
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while self.running:
                # Handle keyboard
                self.handle_keyboard()

                # Read RL command
                res = self.sub.Read()
                if res is not None:
                    rl_msg = res[0] if isinstance(res, tuple) else res
                    self.last_rl_cmd = rl_msg
                    self.frame_count += 1

                # Create command
                cmd = unitree_hg_msg_dds__LowCmd_()

                if self.mode == "WALK":
                    # Pass through RL commands for everything
                    if self.last_rl_cmd is not None:
                        for i in range(35):
                            cmd.motor_cmd[i].q = self.last_rl_cmd.motor_cmd[i].q
                            cmd.motor_cmd[i].kp = self.last_rl_cmd.motor_cmd[i].kp
                            cmd.motor_cmd[i].kd = self.last_rl_cmd.motor_cmd[i].kd
                    else:
                        # Default standing if no RL yet
                        for i, idx in enumerate(LEG_INDICES):
                            cmd.motor_cmd[idx].q = float(self.stand_legs[i])
                            cmd.motor_cmd[idx].kp = 60.0
                            cmd.motor_cmd[idx].kd = 2.0
                        for i, idx in enumerate(WAIST_INDICES):
                            cmd.motor_cmd[idx].q = float(self.stand_waist[i])
                            cmd.motor_cmd[idx].kp = 60.0
                            cmd.motor_cmd[idx].kd = 2.0

                    # Arms always at home in WALK mode
                    for i, idx in enumerate(L_ARM_INDICES):
                        cmd.motor_cmd[idx].q = float(self.home_left[i])
                        cmd.motor_cmd[idx].kp = 30.0
                        cmd.motor_cmd[idx].kd = 1.5
                    for i, idx in enumerate(R_ARM_INDICES):
                        cmd.motor_cmd[idx].q = float(self.home_right[i])
                        cmd.motor_cmd[idx].kp = 30.0
                        cmd.motor_cmd[idx].kd = 1.5

                else:  # MANIPULATE mode
                    # Legs locked in standing
                    for i, idx in enumerate(LEG_INDICES):
                        cmd.motor_cmd[idx].q = float(self.stand_legs[i])
                        cmd.motor_cmd[idx].kp = 80.0  # HIGH stiffness
                        cmd.motor_cmd[idx].kd = 3.0

                    # Waist locked
                    for i, idx in enumerate(WAIST_INDICES):
                        cmd.motor_cmd[idx].q = float(self.stand_waist[i])
                        cmd.motor_cmd[idx].kp = 60.0
                        cmd.motor_cmd[idx].kd = 2.0

                    # Arms via IK
                    l_q, r_q = self.solve_ik()
                    for i, idx in enumerate(L_ARM_INDICES):
                        cmd.motor_cmd[idx].q = float(l_q[i])
                        cmd.motor_cmd[idx].kp = 50.0
                        cmd.motor_cmd[idx].kd = 2.0
                    for i, idx in enumerate(R_ARM_INDICES):
                        cmd.motor_cmd[idx].q = float(r_q[i])
                        cmd.motor_cmd[idx].kp = 50.0
                        cmd.motor_cmd[idx].kd = 2.0

                # Publish
                self.pub.Write(cmd)

                # Status
                if self.frame_count % 30 == 0:
                    if self.mode == "MANIPULATE":
                        g = "CLOSED" if self.gripper_closed else "OPEN"
                        sys.stdout.write(
                            f"\r[MANIPULATE] Gripper: ")
                    else:
                        sys.stdout.write(
                            f"\r[WALK] RL active | Arms at home | Frame: ",
                            f"{self.frame_count}    ")
                    sys.stdout.flush()

                time.sleep(0.01)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\n")


if __name__ == "__main__":
    mixer = SimpleMixer(
        "/home/drake/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml"
    )
    mixer.run()
