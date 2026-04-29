#!/usr/bin/env python3
"""
G1 Whole-Body Mixer - With Proper Mode Isolation
"""

import mujoco
import numpy as np
import time
import threading
import sys
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Joint indices
LEG_JOINTS = list(range(0, 12))      # 0-11: Both legs
WAIST_JOINTS = list(range(12, 15))   # 12-14: Waist
L_ARM_JOINTS = [15, 16, 17, 18, 19, 20, 21]
R_ARM_JOINTS = [22, 23, 24, 25, 26, 27, 28]
ALL_JOINTS = LEG_JOINTS + WAIST_JOINTS + L_ARM_JOINTS + R_ARM_JOINTS


class G1WholeBodyMixer:
    def __init__(self, xml_path):
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

        # SDK Setup
        ChannelFactoryInitialize(0, "lo")

        self.rl_sub = ChannelSubscriber("rt/rl_balance_onlys", LowCmd_)
        self.rl_sub.Init()

        self.real_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.real_pub.Init()

        # Default poses
        mujoco.mj_forward(self.model, self.data)
        
        # Standing leg pose (stable)
        self.standing_leg_joints = np.array([
            0.0, 0.1, 0.0, -0.3, 0.0, -0.3,   # Left leg
            0.0, -0.1, 0.0, -0.3, 0.0, -0.3   # Right leg
        ])
        self.standing_waist_joints = np.array([0.0, 0.0, 0.0])
        
        # Home arm pose (relaxed at sides)
        self.home_left_arm = np.array([0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0])
        self.home_right_arm = np.array([0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0])
        
        # IK targets (start at home position)
        self.l_target = self.data.site_xpos[self.l_ee_id].copy()
        self.r_target = self.data.site_xpos[self.r_ee_id].copy()
        
        # Mode state
        self.mode = "WALK"  # "WALK" or "MANIPULATE"
        self.gripper_closed = False
        self.running = True
        self.step = 0.008
        
        self.frame_count = 0
        self.last_rl_msg = None
        self.keys_pressed = set()
        
        # Smoothing for mode transitions
        self.transition_progress = 0.0
        self.transition_speed = 0.05
        
        print("\n" + "=" * 60)
        print("  G1 WHOLE-BODY MIXER - MODE ISOLATION")
        print("=" * 60)
        print("\nCONTROLS:")
        print("  [TAB]   - Toggle WALK / MANIPULATE mode")
        print("  [SPACE] - Toggle gripper (in MANIPULATE mode)")
        print("  [R]     - Reset arm targets to home")
        print("  [Q]     - Quit")
        print("\nARM CONTROLS (MANIPULATE mode):")
        print("  LEFT:  W/S (fwd/back), A/D (left/right), Z/X (up/down)")
        print("  RIGHT: I/K (fwd/back), J/L (left/right), N/M (up/down)")
        print("\nCURRENT MODE: WALK")
        print("  → Legs: RL Policy   | Arms: Home position")
        print("=" * 60 + "\n")

    def on_key_press(self, key):
        try:
            if hasattr(key, 'char'):
                k = key.char.lower() if key.char else None
            else:
                k = str(key).lower()
            
            if k:
                self.keys_pressed.add(k)
                self.process_key(k)
        except Exception:
            pass
    
    def on_key_release(self, key):
        try:
            if hasattr(key, 'char'):
                k = key.char.lower() if key.char else None
            else:
                k = str(key).lower()
            
            if k and k in self.keys_pressed:
                self.keys_pressed.remove(k)
        except Exception:
            pass
    
    def process_key(self, k):
        # Mode toggle with TAB
        if k == '\t':
            old_mode = self.mode
            self.mode = "MANIPULATE" if self.mode == "WALK" else "WALK"
            self.transition_progress = 0.0
            print(f"\n{'=' * 60}")
            print(f"  MODE SWITCH: {old_mode} → {self.mode}")
            if self.mode == "WALK":
                print("  → Legs: RL Policy   | Arms: Returning to home")
            else:
                print("  → Legs: LOCKED (standing) | Arms: IK Control")
            print(f"{'=' * 60}\n")
        
        # Gripper toggle
        elif k == ' ' and self.mode == "MANIPULATE":
            self.gripper_closed = not self.gripper_closed
            print(f">>> GRIPPER: {'CLOSED (grasping)' if self.gripper_closed else 'OPEN (released)'}")
        
        # Reset arms
        elif k == 'r' and self.mode == "MANIPULATE":
            mujoco.mj_forward(self.model, self.data)
            self.l_target = self.data.site_xpos[self.l_ee_id].copy()
            self.r_target = self.data.site_xpos[self.r_ee_id].copy()
            print(">>> ARMS RESET to current position")
        
        # Quit
        elif k == 'q':
            self.running = False
            print("\n>>> QUITTING <<<")
    
    def update_targets_from_keys(self):
        """Update arm targets based on held keys (only in MANIPULATE mode)"""
        if self.mode != "MANIPULATE":
            return
        
        # Left arm
        if 'w' in self.keys_pressed:
            self.l_target[0] += self.step
        if 's' in self.keys_pressed:
            self.l_target[0] -= self.step
        if 'a' in self.keys_pressed:
            self.l_target[1] += self.step
        if 'd' in self.keys_pressed:
            self.l_target[1] -= self.step
        if 'z' in self.keys_pressed:
            self.l_target[2] += self.step
        if 'x' in self.keys_pressed:
            self.l_target[2] -= self.step
        
        # Right arm
        if 'i' in self.keys_pressed:
            self.r_target[0] += self.step
        if 'k' in self.keys_pressed:
            self.r_target[0] -= self.step
        if 'j' in self.keys_pressed:
            self.r_target[1] += self.step
        if 'l' in self.keys_pressed:
            self.r_target[1] -= self.step
        if 'n' in self.keys_pressed:
            self.r_target[2] += self.step
        if 'm' in self.keys_pressed:
            self.r_target[2] -= self.step

    def solve_ik_for_arms(self):
	    mujoco.mj_forward(self.model, self.data)
	    
	    # Combined Jacobian for both hands
	    jac_l = np.zeros((3, self.model.nv))
	    jac_r = np.zeros((3, self.model.nv))
	    mujoco.mj_jacSite(self.model, self.data, jac_l, None, self.l_ee_id)
	    mujoco.mj_jacSite(self.model, self.data, jac_r, None, self.r_ee_id)
	    
	    # Masking: Ensure Left command only moves Left joints
	    mask_l, mask_r = np.zeros(self.model.nv), np.zeros(self.model.nv)
	    for idx in self.l_adr: mask_l[idx-1] = 1.0
	    for idx in self.r_adr: mask_r[idx-1] = 1.0
	    
	    jac_combined = np.vstack((jac_l * mask_l, jac_r * mask_r))
	    
	    # Error with a 'Soft' gain (0.2 instead of 0.4 or 1.0)
	    err_combined = np.concatenate((
		(self.l_target - self.data.site_xpos[self.l_ee_id]) * 0.2,
		(self.r_target - self.data.site_xpos[self.r_ee_id]) * 0.2
	    ))
	    
	    # Increase Damping (0.05) to stop high-frequency jitter
	    damping = 0.05**2 * np.eye(6)
	    dq = jac_combined.T @ np.linalg.solve(jac_combined @ jac_combined.T + damping, err_combined)
	    
	    mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
	    return [self.data.qpos[idx] for idx in self.l_adr], [self.data.qpos[idx] for idx in self.r_adr]


    def run_with_pynput(self):
        listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        listener.start()
        
        print("Waiting for RL commands...")
        
        while self.running:
            # Update targets based on held keys
            self.update_targets_from_keys()
            
            # Update transition smoothing
            if self.transition_progress < 1.0:
                self.transition_progress = min(1.0, self.transition_progress + self.transition_speed)
            
            # Read RL command
            res = self.rl_sub.Read()
            if res is not None:
                rl_msg = res[0] if isinstance(res, tuple) else res
                self.last_rl_msg = rl_msg
                self.frame_count += 1
            
            # Create command message
            final_cmd = unitree_hg_msg_dds__LowCmd_()
            
            # Default: initialize all motors to zero
            for i in range(35):
                final_cmd.motor_cmd[i].q = 0.0
                final_cmd.motor_cmd[i].kp = 0.0
                final_cmd.motor_cmd[i].kd = 0.0
                final_cmd.motor_cmd[i].dq = 0.0
                final_cmd.motor_cmd[i].tau = 0.0
            
            if self.mode == "WALK":
                # ============================================
                # WALK MODE: RL controls legs + waist
                #            Arms go to home position
                # ============================================
                
                if self.last_rl_msg is not None:
                    # Copy leg and waist commands from RL
                    for i in LEG_JOINTS + WAIST_JOINTS:
                        final_cmd.motor_cmd[i].q = self.last_rl_msg.motor_cmd[i].q
                        final_cmd.motor_cmd[i].kp = self.last_rl_msg.motor_cmd[i].kp
                        final_cmd.motor_cmd[i].kd = self.last_rl_msg.motor_cmd[i].kd
                
                # Arms go to home position (relaxed at sides)
                for i, idx in enumerate(L_ARM_JOINTS):
                    final_cmd.motor_cmd[idx].mode = 0x01 
                    final_cmd.motor_cmd[idx].q = float(self.home_left_arm[i])
                    final_cmd.motor_cmd[idx].kp = 30.0
                    final_cmd.motor_cmd[idx].kd = 2.5
                
                for i, idx in enumerate(R_ARM_JOINTS):
                    final_cmd.motor_cmd[idx].mode = 0x01 
                    final_cmd.motor_cmd[idx].q = float(self.home_right_arm[i])
                    final_cmd.motor_cmd[idx].kp = 30.0
                    final_cmd.motor_cmd[idx].kd = 2.5
                
                if self.frame_count % 50 == 0:
                    print(f"\r[WALK] RL controlling legs | Arms at home                    ", end="")
            
            else:
                # ============================================
                # MANIPULATE MODE: Legs locked in standing
                #                 IK controls arms
                # ============================================
                
                # Legs locked in standing pose
                for i, idx in enumerate(LEG_JOINTS):
                    final_cmd.motor_cmd[idx].q = float(self.standing_leg_joints[i])
                    final_cmd.motor_cmd[idx].kp = 80.0  # Higher stiffness to stay put
                    final_cmd.motor_cmd[idx].kd = 3.0
                
                # Waist neutral
                for i, idx in enumerate(WAIST_JOINTS):
                    final_cmd.motor_cmd[idx].q = float(self.standing_waist_joints[i])
                    final_cmd.motor_cmd[idx].kp = 60.0
                    final_cmd.motor_cmd[idx].kd = 2.0
                
                # Arms via IK
                l_q, r_q = self.solve_ik_for_arms()
                
                for i, idx in enumerate(L_ARM_JOINTS):
                    final_cmd.motor_cmd[idx].q = float(l_q[i])
                    final_cmd.motor_cmd[idx].kp = 50.0
                    final_cmd.motor_cmd[idx].kd = 2.0
                
                for i, idx in enumerate(R_ARM_JOINTS):
                    final_cmd.motor_cmd[idx].q = float(r_q[i])
                    final_cmd.motor_cmd[idx].kp = 50.0
                    final_cmd.motor_cmd[idx].kd = 2.0
                
                if self.frame_count % 25 == 0:
                    gripper = "CLOSED" if self.gripper_closed else "OPEN"
                    print(f"\r[MANIPULATE] Legs LOCKED | Gripper: {gripper} | L:[{self.l_target[0]:.2f},{self.l_target[1]:.2f},{self.l_target[2]:.2f}] R:[{self.r_target[0]:.2f},{self.r_target[1]:.2f},{self.r_target[2]:.2f}]  ", end="")
            
            # Publish command
            self.real_pub.Write(final_cmd)
            time.sleep(0.012)  # ~80 Hz
        
        listener.stop()

    def run(self):
        if PYNPUT_AVAILABLE:
            self.run_with_pynput()
        else:
            print("Please install pynput: pip install pynput")


if __name__ == "__main__":
    mixer = G1WholeBodyMixer(
        "/home/drake/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml")
    mixer.run()
