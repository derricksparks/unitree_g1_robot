#!/usr/bin/env python3
"""
Direct Arm Control Test - unitree_rl_mjlab ONLY
Run this to verify arm control works in the simulation
"""

import time
import sys
import select
import termios
import tty

# Add unitree_rl_mjlab to path if needed
sys.path.insert(0, '/home/drake/unitree_rl_mjlab')

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

print("=" * 60)
print("  DIRECT ARM CONTROL TEST - unitree_rl_mjlab")
print("=" * 60)
print("\nThis script directly controls the G1's right arm.")
print("Make sure ./unitree_mujoco is running first!")
print("\nControls:")
print("  W/S - Move arm forward/back")
print("  A/D - Move arm left/right")
print("  Z/X - Move arm up/down")
print("  G   - Toggle gripper open/close")
print("  Q   - Quit")
print("\n")

# Initialize DDS
ChannelFactoryInitialize(0, "lo")
pub = ChannelPublisher("rt/lowcmd", unitree_hg_msg_dds__LowCmd_)
pub.Init()
print("[INFO] Connected to rt/lowcmd")

# Right arm joint indices: 22=shoulder_pitch, 23=shoulder_roll, 24=shoulder_yaw
#                         25=elbow, 26=wrist_roll, 27=wrist_pitch, 28=wrist_yaw

# Starting arm position (relaxed)
arm_joints = {
    'shoulder_pitch': 0.3,    # Forward/back
    'shoulder_roll': 0.0,     # Left/right
    'shoulder_yaw': 0.0,      # Rotation
    'elbow': -0.5,            # Bend
    'wrist_roll': -1.2,       # Gripper (-1.2=open, 1.2=closed)
    'wrist_pitch': 0.0,       # Tilt
    'wrist_yaw': 0.0          # Twist
}

# Standing leg position (keep robot stable)
leg_positions = [
    0.0, 0.1, 0.0, -0.3, 0.0, -0.3,   # Left leg
    0.0, -0.1, 0.0, -0.3, 0.0, -0.3   # Right leg
]

def get_key():
    """Non-blocking keyboard input"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None

def send_command():
    """Create and send motor command"""
    cmd = unitree_hg_msg_dds__LowCmd_()
    
    # Initialize all motors to zero
    for i in range(35):
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].tau = 0.0
    
    # Set legs to standing position
    for i in range(12):
        cmd.motor_cmd[i].q = float(leg_positions[i])
        cmd.motor_cmd[i].kp = 60.0
        cmd.motor_cmd[i].kd = 2.0
    
    # Set waist to neutral
    for i in range(12, 15):
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 40.0
        cmd.motor_cmd[i].kd = 1.0
    
    # Set left arm to relaxed position
    cmd.motor_cmd[15].q = 0.3   # shoulder pitch
    cmd.motor_cmd[15].kp = 20.0
    cmd.motor_cmd[18].q = -0.5  # elbow
    cmd.motor_cmd[18].kp = 20.0
    cmd.motor_cmd[19].q = -1.2  # wrist roll (open)
    cmd.motor_cmd[19].kp = 20.0
    
    # Set right arm from our controlled values
    cmd.motor_cmd[22].q = float(arm_joints['shoulder_pitch'])
    cmd.motor_cmd[22].kp = 40.0
    cmd.motor_cmd[22].kd = 1.0
    
    cmd.motor_cmd[23].q = float(arm_joints['shoulder_roll'])
    cmd.motor_cmd[23].kp = 40.0
    cmd.motor_cmd[23].kd = 1.0
    
    cmd.motor_cmd[24].q = float(arm_joints['shoulder_yaw'])
    cmd.motor_cmd[24].kp = 40.0
    cmd.motor_cmd[24].kd = 1.0
    
    cmd.motor_cmd[25].q = float(arm_joints['elbow'])
    cmd.motor_cmd[25].kp = 40.0
    cmd.motor_cmd[25].kd = 1.0
    
    cmd.motor_cmd[26].q = float(arm_joints['wrist_roll'])
    cmd.motor_cmd[26].kp = 40.0
    cmd.motor_cmd[26].kd = 1.0
    
    cmd.motor_cmd[27].q = float(arm_joints['wrist_pitch'])
    cmd.motor_cmd[27].kp = 30.0
    cmd.motor_cmd[27].kd = 1.0
    
    cmd.motor_cmd[28].q = float(arm_joints['wrist_yaw'])
    cmd.motor_cmd[28].kp = 30.0
    cmd.motor_cmd[28].kd = 1.0
    
    pub.Write(cmd)

def print_status():
    """Print current arm status"""
    gripper_state = "CLOSED" if arm_joints['wrist_roll'] > 0 else "OPEN"
    print(f"\r[ARM] Pitch:{arm_joints['shoulder_pitch']:+.2f} "
          f"Roll:{arm_joints['shoulder_roll']:+.2f} "
          f"Elbow:{arm_joints['elbow']:+.2f} "
          f"Gripper:{gripper_state}    ", end="")

# Setup terminal for raw input
old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

print("\n[READY] Use W/S/A/D/Z/X to move arm, G to toggle gripper, Q to quit\n")
frame = 0

try:
    while True:
        k = get_key()
        
        if k == 'w':
            arm_joints['shoulder_pitch'] -= 0.03
        elif k == 's':
            arm_joints['shoulder_pitch'] += 0.03
        elif k == 'a':
            arm_joints['shoulder_roll'] -= 0.03
        elif k == 'd':
            arm_joints['shoulder_roll'] += 0.03
        elif k == 'z':
            arm_joints['elbow'] -= 0.03
        elif k == 'x':
            arm_joints['elbow'] += 0.03
        elif k == 'g':
            arm_joints['wrist_roll'] = 1.2 if arm_joints['wrist_roll'] < 0 else -1.2
        elif k == 'q':
            break
        
        # Clamp joint limits
        arm_joints['shoulder_pitch'] = max(-2.0, min(2.0, arm_joints['shoulder_pitch']))
        arm_joints['shoulder_roll'] = max(-1.0, min(1.0, arm_joints['shoulder_roll']))
        arm_joints['elbow'] = max(-1.0, min(1.5, arm_joints['elbow']))
        
        # Send command
        send_command()
        
        # Print status every 20 frames
        frame += 1
        if frame % 20 == 0:
            print_status()
        
        time.sleep(0.02)

finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print("\n\n[DONE]")
