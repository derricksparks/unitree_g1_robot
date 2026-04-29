import mujoco
import numpy as np
import time
import threading
import sys
import termios
import tty
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

# SDK Indices
L_ARM_INDICES = [15, 16, 17, 18, 19, 20, 21]
R_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]


class G1DualArmIK:
    def __init__(self, xml_path):
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"Error loading XML: {e}")
            sys.exit(1)

        # 1. Setup Sites
        self.l_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand_site")
        self.r_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_site")

        # 2. Map Joint Addresses
        l_names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                   "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        r_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                   "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

        self.l_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in l_names]
        self.r_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in r_names]

        # 3. SDK Init
        ChannelFactoryInitialize(0, "lo")
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        # 4. Targets
        mujoco.mj_forward(self.model, self.data)
        self.l_target = self.data.site_xpos[self.l_ee_id].copy()
        self.r_target = self.data.site_xpos[self.r_ee_id].copy()
        self.running = True
        self.step = 0.01

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def input_loop(self):
        print("\n--- G1 DUAL ARM INDEPENDENT CONTROL ---")
        print("LEFT ARM:  W/S (X), A/D (Y), Q/E (Z)")
        print("RIGHT ARM: I/K (X), J/L (Y), U/O (Z)")
        print("EXIT:      ESC")
        while self.running:
            k = self.get_key()
            if k == 'w':
                self.l_target[0] += self.step
            elif k == 's':
                self.l_target[0] -= self.step
            elif k == 'a':
                self.l_target[1] += self.step
            elif k == 'd':
                self.l_target[1] -= self.step
            elif k == 'q':
                self.l_target[2] += self.step
            elif k == 'e':
                self.l_target[2] -= self.step
            elif k == 'i':
                self.r_target[0] += self.step
            elif k == 'k':
                self.r_target[0] -= self.step
            elif k == 'j':
                self.r_target[1] += self.step
            elif k == 'l':
                self.r_target[1] -= self.step
            elif k == 'u':
                self.r_target[2] += self.step
            elif k == 'o':
                self.r_target[2] -= self.step
            elif k == '\x1b':
                self.running = False

    def solve_dual_ik(self):
        # 1. Update Kinematics
        mujoco.mj_forward(self.model, self.data)

        # 2. Setup Jacobians and Masks
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

        # 3. Calculate Error with a Gain
        # Reducing gain to 0.5 prevents 'overshooting' the limit which causes shaking
        err_l = (self.l_target - self.data.site_xpos[self.l_ee_id]) * 0.5
        err_r = (self.r_target - self.data.site_xpos[self.r_ee_id]) * 0.5
        err_combined = np.concatenate((err_l, err_r))

        # 4. Damped Least Squares (Increased damping from 0.02 to 0.05)
        # Higher damping (0.05) makes the arm 'soften' near its limits instead of shaking
        n_task = 6
        damping = 0.05**2 * np.eye(n_task)
        dq = jac_combined.T @ np.linalg.solve(jac_combined @ jac_combined.T + damping, err_combined)

        # 5. Integrate and CLAMP (Crucial for stability)
        mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)

        # Enforce Joint Limits: Prevent qpos from exceeding XML ranges
        # This stops the 'loss of control' when hitting a limit
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                low, high = self.model.jnt_range[i]
                addr = self.model.jnt_qposadr[i]
                self.data.qpos[addr] = np.clip(self.data.qpos[addr], low, high)

        mujoco.mj_forward(self.model, self.data)

        l_q = [self.data.qpos[idx] for idx in self.l_adr]
        r_q = [self.data.qpos[idx] for idx in self.r_adr]
        return l_q, r_q

    def run(self):
        threading.Thread(target=self.input_loop, daemon=True).start()
        while self.running:
            start = time.perf_counter()

            # Solve dual IK in one pass
            l_q, r_q = self.solve_dual_ik()

            cmd = unitree_hg_msg_dds__LowCmd_()
            for i, motor_idx in enumerate(L_ARM_INDICES):
                cmd.motor_cmd[motor_idx].q = float(l_q[i])
                cmd.motor_cmd[motor_idx].kp = 30.0
                cmd.motor_cmd[motor_idx].kd = 2.0

            for i, motor_idx in enumerate(R_ARM_INDICES):
                cmd.motor_cmd[motor_idx].q = float(r_q[i])
                cmd.motor_cmd[motor_idx].kp = 30.0
                cmd.motor_cmd[motor_idx].kd = 2.0

            self.pub.Write(cmd)
            time.sleep(max(0, 0.005 - (time.perf_counter() - start)))


if __name__ == "__main__":
    IK_APP = G1DualArmIK(
        "/home/drake/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml")
    IK_APP.run()
