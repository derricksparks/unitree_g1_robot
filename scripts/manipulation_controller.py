#!/usr/bin/env python3
"""
FULL TAKEOVER ARM CONTROLLER
Completely bypasses g1_ctrl - no conflicts
"""
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from stable_baselines3 import PPO
import sys
import os
import time
import select
import termios
import tty
import numpy as np

SENTDEX_PATH = '/home/drake/unitree_rl_mjlab/perception/unitree_g1_vibes/RL-shenanigans'
sys.path.insert(0, SENTDEX_PATH)


# Load model
MODEL_PATH = os.path.join(SENTDEX_PATH, 'models', 'ppo_g1_left_53178k.zip')
print(f"Loading {MODEL_PATH}...")
model = PPO.load(MODEL_PATH, device="cpu")
model.policy.set_training_mode(False)
print("✓ Model loaded")

# DDS
ChannelFactoryInitialize(0, "lo")
pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
print("✓ DDS connected")

# ============================================================
# FULL ROBOT CONTROL - No g1_ctrl needed
# ============================================================
LEGS = [0.0, 0.1, 0.0, -0.3, 0.0, -0.3, 0.0, -0.1, 0.0, -0.3, 0.0, -0.3]
WAIST = [0.0, 0.0, 0.0]

# Home arm positions (relaxed)
HOME_LEFT = [0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0]
HOME_RIGHT = [0.3, 0.0, 0.0, -0.5, -1.2, 0.0, 0.0]

goal = np.array([0.3, 0.2, 0.5])
gripper = False
step = 0.02


def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


def send_full_command(left_arm_q, grip_val):
    """Send COMPLETE robot command - all 29 joints"""
    cmd = unitree_hg_msg_dds__LowCmd_()

    # Initialize all to zero
    for i in range(35):
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0

    # === LEGS (0-11) - Standing pose ===
    for i in range(12):
        cmd.motor_cmd[i].q = float(LEGS[i])
        cmd.motor_cmd[i].kp = 80.0
        cmd.motor_cmd[i].kd = 3.0

    # === WAIST (12-14) - Neutral ===
    for i in range(12, 15):
        cmd.motor_cmd[i].q = float(WAIST[i - 12])
        cmd.motor_cmd[i].kp = 60.0
        cmd.motor_cmd[i].kd = 2.0

    # === LEFT ARM (15-21) - From policy ===
    for i, idx in enumerate(range(15, 22)):
        cmd.motor_cmd[idx].q = float(left_arm_q[i])
        cmd.motor_cmd[idx].kp = 40.0
        cmd.motor_cmd[idx].kd = 1.0

    # === RIGHT ARM (22-28) - Home position ===
    for i, idx in enumerate(range(22, 29)):
        cmd.motor_cmd[idx].q = float(HOME_RIGHT[i])
        cmd.motor_cmd[idx].kp = 20.0
        cmd.motor_cmd[idx].kd = 1.0

    # === GRIPPER (left wrist roll = index 19) ===
    cmd.motor_cmd[19].q = float(grip_val)
    cmd.motor_cmd[19].kp = 50.0
    cmd.motor_cmd[19].kd = 1.0

    pub.Write(cmd)


print("=" * 60)
print("  FULL TAKEOVER ARM CONTROLLER")
print("=" * 60)
print("\n⚠️  STOP g1_ctrl first (Ctrl+C) - this controller")
print("    takes COMPLETE control of the robot!")
print("\nCONTROLS:")
print("  W/S - goal Up/Down")
print("  A/D - goal Left/Right")
print("  Q/E - goal Forward/Back")
print("  G   - Toggle gripper")
print("  R   - Reset goal")
print("  ESC - Quit")
print("\n[READY] Starting in 3 seconds...")
time.sleep(3)

old = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

# Send initial stable pose
send_full_command(HOME_LEFT, -1.2)
time.sleep(1.0)  # Give time to stabilize

print("Control active!")

try:
    while True:
        k = get_key()
        if k == 'w':
            goal[2] += step
        elif k == 's':
            goal[2] -= step
        elif k == 'a':
            goal[1] += step
        elif k == 'd':
            goal[1] -= step
        elif k == 'q':
            goal[0] += step
        elif k == 'e':
            goal[0] -= step
        elif k == 'g':
            gripper = not gripper
            print(f"\n>>> Gripper: {'CLOSED' if gripper else 'OPEN'} <<<")
        elif k == 'r':
            goal = np.array([0.3, 0.2, 0.5])
            print("\n>>> Goal reset <<<")
        elif k == '\x1b':
            break

        goal = np.clip(goal, [-0.1, -0.6, 0.4], [0.6, 0.6, 1.4])

        # Get action from policy
        obs = np.zeros(24)
        obs[:3] = goal
        action, _ = model.predict(obs, deterministic=True)

        gv = 1.2 if gripper else -1.2
        send_full_command(action, gv)

        gs = "CLOSED" if gripper else "OPEN"
        print(f"\rGoal:[{goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}] G:{gs}    ", end="")
        time.sleep(0.03)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
    # Return to relaxed home
    send_full_command(HOME_LEFT, -1.2)
    time.sleep(0.5)
    print("\n[DONE]")
