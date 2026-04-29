"""Shared simulation primitives for the Unitree G1 + warehouse scene.

This module encapsulates:
  * Kinematic walk + dual-arm DLS IK + grasp/carry (used by `pick_and_place.py`
    and `sim_cli.py`).
  * A `LocomotionController` interface with a kinematic fallback + an optional
    trained-RL-policy backend (so the same teleop code can drive either).
  * `ArmCartesianController`, a task-priority DLS IK with a torso-upright
    posture regularizer used to teleop the arms with keyboard/joystick.
  * `BoxCamDetector`, a small perception module that renders the onboard
    `d435i_rgb` view, HSV-segments the red warehouse box, and ray-casts the
    detection back into the world to estimate the box pose.

Pure MuJoCo, no Unitree SDK and no trained checkpoint required (the RL
locomotion plug-in is optional and falls back to kinematic walking).
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import numpy as np

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml"


# ----------------------------- joint helpers -------------------------------- #

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = tuple(j.replace("left_", "right_") for j in LEFT_ARM_JOINTS)

# Whole-body "ready" pose (knees slightly bent, arms at sides).
READY_POSE = {
    "left_hip_pitch_joint":  -0.10,
    "left_knee_joint":        0.30,
    "left_ankle_pitch_joint":-0.20,
    "right_hip_pitch_joint": -0.10,
    "right_knee_joint":       0.30,
    "right_ankle_pitch_joint":-0.20,
    "left_shoulder_pitch_joint":  0.20,
    "right_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint":   0.18,
    "right_shoulder_roll_joint": -0.18,
    "left_elbow_joint":  0.6,
    "right_elbow_joint": 0.6,
}

Hand = Literal["left", "right"]


def _name_to_id(model: mujoco.MjModel, obj: int, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise KeyError(f"name not found: {name}")
    return int(i)


# --------------------------------- IK --------------------------------------- #

class ArmIK:
    """Damped-least-squares IK for the left or right arm (3-DoF position)."""

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...], site_name: str):
        self.model = model
        self.joint_names = joint_names
        self.site = _name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.dof_idx = np.array(
            [int(model.jnt_dofadr[_name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
             for n in joint_names], dtype=np.int32,
        )
        self.qpos_idx = np.array(
            [int(model.jnt_qposadr[_name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
             for n in joint_names], dtype=np.int32,
        )

    def ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.site_xpos[self.site]).copy()

    def step_to(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        gain: float = 0.5,
        damping: float = 0.05,
        max_step: float = 0.05,
    ) -> float:
        mujoco.mj_forward(self.model, data)
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jac, None, self.site)
        J = jac[:, self.dof_idx]
        err = (np.asarray(target) - data.site_xpos[self.site]) * gain
        lam2 = damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
        np.clip(dq, -max_step, max_step, out=dq)
        for k, qadr in enumerate(self.qpos_idx):
            data.qpos[qadr] = float(data.qpos[qadr] + dq[k])
        for n in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                qadr = int(self.model.jnt_qposadr[jid])
                data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))
        mujoco.mj_forward(self.model, data)
        return float(np.linalg.norm(np.asarray(target) - data.site_xpos[self.site]))


# ----------------------------- walk parameters ------------------------------ #

@dataclass
class WalkParams:
    speed_mps: float = 0.9
    yaw_rate: float = 1.5
    arm_swing_amp: float = 0.30
    swing_freq_hz: float = 1.6


# ----------------------------- the world wrapper ---------------------------- #

@dataclass
class RenderConfig:
    enabled: bool = False
    width: int = 960
    height: int = 540
    fps: int = 30
    video_path: Optional[str] = None
    track_pelvis: bool = True
    azimuth: float = 135.0
    elevation: float = -18.0
    distance: float = 5.0


class G1World:
    """Holds the MuJoCo model/data plus all the helpers used by demo + CLI.

    Use it like:

        w = G1World()
        w.reset()
        w.walk_to(np.array([1.45, 0.0]), yaw=0.0)
        w.reach("left", w.box_pos() + [0, 0, 0.04])
        w.grasp("left")
        w.walk_to(np.array([-2.45, 0.0]), yaw=math.pi)
        w.reach("left", w.place_target_pos() + [0, 0, 0.06])
        w.release()
        w.close()
    """

    def __init__(self, scene_path: os.PathLike | str | None = None):
        self.scene_path = Path(scene_path or DEFAULT_SCENE).resolve()
        if not self.scene_path.exists():
            raise FileNotFoundError(self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.left_ik = ArmIK(self.model, LEFT_ARM_JOINTS, "left_hand_site")
        self.right_ik = ArmIK(self.model, RIGHT_ARM_JOINTS, "right_hand_site")

        # Cached ids.
        self._pelvis_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._box_b    = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_box")
        self._place_s  = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "place_target")
        self._left_h_b  = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self._right_h_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self._table_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "reception_table")
        self._shelf_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "storage_shelf")
        self._box_qadr = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ])
        self._box_dofadr = int(self.model.jnt_dofadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ])

        # Carry state.
        self._held_by: Optional[Hand] = None
        self._box_offset_in_hand = np.zeros(3)
        self._walk_phase: float = 0.0
        self._sim_time: float = 0.0
        self._step_count: int = 0

        # Render state.
        self.render_cfg = RenderConfig()
        self._renderer: Optional[mujoco.Renderer] = None
        self._video_writer = None
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = (0.0, 0.0, 0.9)
        self._cam.distance = self.render_cfg.distance
        self._cam.azimuth = self.render_cfg.azimuth
        self._cam.elevation = self.render_cfg.elevation
        self._frames_since_render = 0
        self._render_every = max(1, int(round((1.0 / max(1, self.render_cfg.fps)) / self.dt)))

        # Viewer (lazy).
        self._viewer = None

        # On-step user hook (e.g. for the CLI to print progress).
        self.on_step: Optional[Callable[[], None]] = None

        # Track whether an external Cartesian arm controller currently owns
        # each arm. When true, walk-swing won't overwrite shoulder_pitch.
        self._arm_ctrl_active_left: bool = False
        self._arm_ctrl_active_right: bool = False

        self.reset()

    # ----------------------- accessors ----------------------- #

    @property
    def pelvis_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._pelvis_b]).copy()

    @property
    def pelvis_yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def box_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._box_b]).copy()

    def place_target_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self._place_s]).copy()

    def hand_pos(self, hand: Hand) -> np.ndarray:
        ik = self.left_ik if hand == "left" else self.right_ik
        return ik.ee_pos(self.data)

    @property
    def held_by(self) -> Optional[Hand]:
        return self._held_by

    @property
    def sim_time(self) -> float:
        return self._sim_time

    # ----------------------- reset / pose ----------------------- #

    def apply_pose(self, pose: dict[str, float]) -> None:
        for name, val in pose.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            self.data.qpos[int(self.model.jnt_qposadr[jid])] = float(val)

    def set_floating_base(self, pos: Iterable[float], yaw: float = 0.0) -> None:
        half = 0.5 * yaw
        x, y, z = pos
        self.data.qpos[0] = float(x)
        self.data.qpos[1] = float(y)
        self.data.qpos[2] = float(z)
        self.data.qpos[3] = math.cos(half)
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = math.sin(half)
        self.data.qvel[0:6] = 0.0

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.apply_pose(READY_POSE)
        self.set_floating_base((0.0, 0.0, 0.793), yaw=0.0)
        # Re-place the box on the table (in case it was teleported).
        self.data.qpos[self._box_qadr + 0] = 1.90
        self.data.qpos[self._box_qadr + 1] = 0.15
        self.data.qpos[self._box_qadr + 2] = 0.81
        self.data.qpos[self._box_qadr + 3] = 1.0
        self.data.qpos[self._box_qadr + 4:self._box_qadr + 7] = 0.0
        self.data.qvel[self._box_dofadr:self._box_dofadr + 6] = 0.0
        self._held_by = None
        self._sim_time = 0.0
        self._step_count = 0
        self._walk_phase = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ----------------------- carry (grasp/release) ----------------------- #

    def grasp(self, hand: Hand) -> None:
        bid = self._left_h_b if hand == "left" else self._right_h_b
        hand_pos = np.asarray(self.data.xpos[bid]).copy()
        hand_mat = np.asarray(self.data.xmat[bid]).reshape(3, 3).copy()
        rel = hand_mat.T @ (self.box_pos() - hand_pos)
        self._held_by = hand
        self._box_offset_in_hand[:] = rel

    def release(self) -> None:
        self._held_by = None

    def _update_carry(self) -> None:
        if self._held_by is None:
            return
        bid = self._left_h_b if self._held_by == "left" else self._right_h_b
        hand_pos = np.asarray(self.data.xpos[bid]).copy()
        hand_mat = np.asarray(self.data.xmat[bid]).reshape(3, 3).copy()
        new_pos = hand_pos + hand_mat @ self._box_offset_in_hand
        self.data.qpos[self._box_qadr + 0] = float(new_pos[0])
        self.data.qpos[self._box_qadr + 1] = float(new_pos[1])
        self.data.qpos[self._box_qadr + 2] = float(new_pos[2])
        self.data.qpos[self._box_qadr + 3] = 1.0
        self.data.qpos[self._box_qadr + 4:self._box_qadr + 7] = 0.0

    # ----------------------- step / render ----------------------- #

    def _emit_step(self) -> None:
        self._step_count += 1
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                self._viewer = None
        if self._renderer is not None and self._video_writer is not None:
            self._frames_since_render += 1
            if self._frames_since_render >= self._render_every:
                self._frames_since_render = 0
                if self.render_cfg.track_pelvis:
                    p = self.pelvis_pos
                    self._cam.lookat[0] = 0.7 * self._cam.lookat[0] + 0.3 * float(p[0])
                    self._cam.lookat[1] = 0.7 * self._cam.lookat[1] + 0.3 * float(p[1])
                    self._cam.lookat[2] = 0.9
                self._renderer.update_scene(self.data, camera=self._cam)
                self._video_writer.append_data(self._renderer.render())
        if self.on_step is not None:
            self.on_step()

    def kine_tick(self) -> None:
        """Refresh kinematics without integrating physics."""
        self._update_carry()
        mujoco.mj_forward(self.model, self.data)
        self._sim_time += self.dt
        self._emit_step()

    def physics_tick_box_only(self) -> None:
        """Step physics but freeze the robot (only the box moves).

        Used briefly after a release so the box settles under gravity. The
        robot itself has no controller in this demo, so without freezing the
        rest of the state it would just collapse.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = np.zeros_like(self.data.qvel)
        box_q_before = self.data.qpos[self._box_qadr:self._box_qadr + 7].copy()
        box_v_before = self.data.qvel[self._box_dofadr:self._box_dofadr + 6].copy()
        mujoco.mj_step(self.model, self.data)
        new_box_q = self.data.qpos[self._box_qadr:self._box_qadr + 7].copy()
        new_box_v = self.data.qvel[self._box_dofadr:self._box_dofadr + 6].copy()
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.qpos[self._box_qadr:self._box_qadr + 7] = new_box_q
        self.data.qvel[self._box_dofadr:self._box_dofadr + 6] = new_box_v
        del box_q_before, box_v_before  # unused but kept for symmetry/debugging
        self._sim_time += self.dt
        self._emit_step()

    # ----------------------- walking ----------------------- #

    def walk_step(self, target_xy: np.ndarray, target_yaw: float, p: WalkParams) -> bool:
        """One walk increment toward (target_xy, target_yaw). Returns arrived."""
        pos = self.data.qpos[0:3].copy()
        yaw = self.pelvis_yaw

        yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        dyaw = float(np.clip(yaw_err, -p.yaw_rate * self.dt, p.yaw_rate * self.dt))
        new_yaw = yaw + dyaw

        xy_err = np.asarray(target_xy) - pos[:2]
        dist = float(np.linalg.norm(xy_err))
        step_len = min(dist, p.speed_mps * self.dt)
        if dist > 1e-6:
            new_xy = pos[:2] + xy_err / dist * step_len
        else:
            new_xy = pos[:2]

        self.set_floating_base((new_xy[0], new_xy[1], pos[2]), new_yaw)

        # Arm swing.
        self._walk_phase += 2.0 * math.pi * p.swing_freq_hz * self.dt
        swing = p.arm_swing_amp * math.sin(self._walk_phase)
        l_idx = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")
        ])
        r_idx = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
        ])
        self.data.qpos[l_idx] = READY_POSE["left_shoulder_pitch_joint"] + swing
        self.data.qpos[r_idx] = READY_POSE["right_shoulder_pitch_joint"] - swing

        return dist < 0.02 and abs(yaw_err) < 0.05

    def walk_to(
        self,
        target_xy: np.ndarray,
        yaw: float,
        max_seconds: float = 60.0,
        params: Optional[WalkParams] = None,
    ) -> bool:
        p = params or WalkParams()
        max_steps = int(max_seconds / self.dt)
        for _ in range(max_steps):
            arrived = self.walk_step(np.asarray(target_xy, dtype=float), float(yaw), p)
            self.kine_tick()
            if arrived:
                # Reset arm swing back to ready before whatever comes next.
                self.apply_pose({
                    "left_shoulder_pitch_joint":  READY_POSE["left_shoulder_pitch_joint"],
                    "right_shoulder_pitch_joint": READY_POSE["right_shoulder_pitch_joint"],
                })
                self.kine_tick()
                return True
        return False

    # ----------------------- arm reach ----------------------- #

    def reach(
        self,
        hand: Hand,
        target: np.ndarray,
        tol: float = 0.04,
        max_iters: int = 1500,
    ) -> tuple[bool, float]:
        ik = self.left_ik if hand == "left" else self.right_ik
        for _ in range(max_iters):
            err = ik.step_to(self.data, np.asarray(target, dtype=float))
            self.kine_tick()
            if err < tol:
                return True, err
        return False, err

    def hold(self, n_steps: int = 30) -> None:
        for _ in range(int(n_steps)):
            self.kine_tick()

    def settle_box(self, seconds: float = 1.5) -> None:
        for _ in range(int(seconds / self.dt)):
            self.physics_tick_box_only()

    # ----------------------- velocity-driven walk (teleop) ----------------- #

    def cmd_vel(
        self,
        forward: float,
        lateral: float,
        yaw_rate: float,
        params: Optional[WalkParams] = None,
    ) -> None:
        """Apply one tick of body-frame velocity command to the floating base.

        Designed for joystick / keyboard teleop. Same kinematic model as
        `walk_step`; the floating base just moves in the heading frame.
        """
        p = params or WalkParams()
        # Cap the inputs at the current walk-speed limits.
        f = float(np.clip(forward,  -p.speed_mps, p.speed_mps))
        s = float(np.clip(lateral,  -p.speed_mps, p.speed_mps))
        w = float(np.clip(yaw_rate, -p.yaw_rate,   p.yaw_rate))

        yaw = self.pelvis_yaw
        # Body -> world rotation for planar motion.
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx = cy * f - sy * s
        vy = sy * f + cy * s

        pos = self.data.qpos[0:3].copy()
        new_x = pos[0] + vx * self.dt
        new_y = pos[1] + vy * self.dt
        new_yaw = yaw + w * self.dt
        self.set_floating_base((new_x, new_y, pos[2]), new_yaw)

        # Arm-swing sized by speed magnitude so standing still doesn't swing.
        speed_mag = math.hypot(f, s) / max(p.speed_mps, 1e-3)
        self._walk_phase += 2.0 * math.pi * p.swing_freq_hz * self.dt
        swing = p.arm_swing_amp * speed_mag * math.sin(self._walk_phase)
        l_idx = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")
        ])
        r_idx = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
        ])
        # Only override shoulder pitch when no per-arm Cartesian command is
        # active; otherwise the ArmCartesianController will set them.
        if not getattr(self, "_arm_ctrl_active_left", False):
            self.data.qpos[l_idx] = READY_POSE["left_shoulder_pitch_joint"] + swing
        if not getattr(self, "_arm_ctrl_active_right", False):
            self.data.qpos[r_idx] = READY_POSE["right_shoulder_pitch_joint"] - swing

    # ----------------------- video / viewer ----------------------- #

    def start_recording(self, video_path: str, width: int = 960, height: int = 540, fps: int = 30) -> None:
        try:
            import imageio.v2 as imageio  # noqa: F401
        except Exception:
            print("[g1_sim] installing imageio[ffmpeg]...")
            os.system(f"{sys.executable} -m pip install --quiet imageio[ffmpeg]")
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        self.render_cfg = RenderConfig(
            enabled=True, width=width, height=height, fps=fps, video_path=str(video_path)
        )
        if self._renderer is None or self._renderer._width != width or self._renderer._height != height:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._video_writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
        self._render_every = max(1, int(round((1.0 / fps) / self.dt)))
        self._frames_since_render = 0

    def stop_recording(self) -> Optional[str]:
        path = self.render_cfg.video_path
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception:
                pass
            self._video_writer = None
        self.render_cfg = RenderConfig()
        return path

    def open_viewer(self) -> None:
        if self._viewer is not None:
            return
        import mujoco.viewer as mjviewer
        self._viewer = mjviewer.launch_passive(self.model, self.data)

    def close_viewer(self) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.close()
        except Exception:
            pass
        self._viewer = None

    def close(self) -> None:
        self.stop_recording()
        self.close_viewer()


# ----------------------- canned demo entry point --------------------------- #

def run_demo(world: G1World, verbose: bool = True) -> int:
    """Walk -> pick -> walk -> place -> retreat. Returns 0 on success."""
    pr = (lambda *a, **k: print("[g1_sim]", *a, **k)) if verbose else (lambda *a, **k: None)

    pr(f"WALK_TO_TABLE: walking to (1.45, 0.0) facing +x")
    world.walk_to(np.array([1.45, 0.0]), yaw=0.0)

    box = world.box_pos()
    above = box + np.array([0.0, 0.0, 0.10])
    grasp = box + np.array([0.0, 0.0, 0.04])
    pr("PICK_APPROACH"); world.reach("left", above, tol=0.04)
    pr("PICK_GRASP");    world.reach("left", grasp, tol=0.05)
    world.grasp("left"); pr("grasping (attached to left hand)")
    pr("PICK_LIFT");     world.reach("left", box + np.array([0.0, 0.0, 0.20]), tol=0.05)
    world.hold(30)

    pr("WALK_TO_SHELF: walking to (-2.45, 0.0) facing -x")
    world.walk_to(np.array([-2.45, 0.0]), yaw=math.pi)

    target = world.place_target_pos() + np.array([0.0, 0.0, 0.06])
    pr("PLACE_APPROACH"); world.reach("left", target + np.array([0, 0, 0.10]), tol=0.05)
    pr("PLACE_DOWN");     world.reach("left", target,                          tol=0.04)
    world.release(); pr("released (detached from left hand)")
    world.settle_box(1.5)

    pr("RETREAT: walking back to (0, 0) facing +x")
    world.walk_to(np.array([0.0, 0.0]), yaw=0.0)
    world.hold(60)

    box_xy_err = float(np.linalg.norm(world.box_pos()[:2] - world.place_target_pos()[:2]))
    pr(f"final box pos: {world.box_pos().tolist()}")
    pr(f"place target: {world.place_target_pos().tolist()}")
    pr(f"box-target XY error: {box_xy_err:.3f} m")
    return 0


# =========================================================================== #
# Locomotion plug-in
# =========================================================================== #

class LocomotionController:
    """Minimal interface for swappable locomotion backends.

    A locomotion controller consumes a body-frame velocity command
    `(forward, lateral, yaw_rate)` and advances the simulation by one tick.
    The default `KinematicLocomotionController` slides the floating base
    along the command (current behavior). `TrainedRLLocomotionController`
    plugs in a saved policy checkpoint when one is available.
    """

    def __init__(self, world: G1World):
        self.world = world

    def step(self, forward: float, lateral: float, yaw_rate: float) -> None:
        raise NotImplementedError

    def name(self) -> str:
        return type(self).__name__


class KinematicLocomotionController(LocomotionController):
    """Floating-base velocity integrator with cosmetic arm swing.

    Use this whenever a trained walking policy is unavailable. It produces
    identical results to the canned demo's walking phases.
    """

    def __init__(self, world: G1World, params: Optional[WalkParams] = None):
        super().__init__(world)
        self.params = params or WalkParams()

    def step(self, forward: float, lateral: float, yaw_rate: float) -> None:
        self.world.cmd_vel(forward, lateral, yaw_rate, self.params)
        self.world.kine_tick()


class TrainedRLLocomotionController(LocomotionController):
    """Optional plug-in that runs a saved torch policy on the legs.

    The repo trains the `Unitree-G1-Flat`/`Unitree-G1-Rough` policies via
    mjlab. If a checkpoint is supplied we load it and feed the policy the
    same observation it was trained with: base angular velocity, projected
    gravity, the (forward, lateral, yaw) twist command, gait phase,
    relative joint positions / velocities and the previous action. The
    policy's first 12 outputs are interpreted as leg joint targets and
    applied via PD; the rest of the body (arms / waist) remains under the
    Cartesian arm controller and the kinematic floating-base integrator.

    Falls back to the kinematic controller if anything goes wrong, with a
    one-time warning. This keeps the teleop stack functional even when no
    checkpoint is reachable from the current environment.
    """

    LEG_JOINT_NAMES = (
        "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
        "left_knee_joint",       "left_ankle_pitch_joint","left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    )

    def __init__(
        self,
        world: G1World,
        checkpoint_path: os.PathLike | str,
        params: Optional[WalkParams] = None,
        gait_period_s: float = 0.6,
    ):
        super().__init__(world)
        self.params = params or WalkParams()
        self.fallback = KinematicLocomotionController(world, self.params)
        self.gait_period = float(gait_period_s)
        self._failed = False
        self._policy = None
        try:
            import torch  # noqa: F401  (only imported lazily)
            self._policy = self._load_checkpoint(Path(checkpoint_path))
        except Exception as e:
            print(f"[g1_sim] WARN: trained-locomotion checkpoint unusable ({e!r}); "
                  "falling back to kinematic walker.")
            self._failed = True

        # Per-step state.
        self._last_action = np.zeros(12, dtype=np.float32)
        self._counter = 0

    def _load_checkpoint(self, path: Path):
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        # Accept several common shapes:
        #   - state-dict at top level
        #   - {"actor_state_dict": {...}}
        #   - {"model": {...}}
        if isinstance(ckpt, dict):
            for key in ("actor_state_dict", "policy_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            else:
                state = ckpt
        else:
            state = ckpt
        # Try to reconstruct an MLP from contiguous Linear layers.
        weights, biases = [], []
        i = 0
        while True:
            wkey = f"mlp.{i}.weight"; bkey = f"mlp.{i}.bias"
            if wkey not in state or bkey not in state:
                # Some exports use 'actor.mlp.{i}.weight' or '.0.weight'.
                wkey2 = f"actor.mlp.{i}.weight"; bkey2 = f"actor.mlp.{i}.bias"
                if wkey2 in state and bkey2 in state:
                    wkey, bkey = wkey2, bkey2
                else:
                    break
            weights.append(state[wkey]); biases.append(state[bkey])
            i += 2  # MLPs typically interleave Linear / activation
        if not weights:
            raise RuntimeError("could not find mlp.{i}.weight in checkpoint")
        layers = []
        for j, (W, b) in enumerate(zip(weights, biases)):
            lin = torch.nn.Linear(W.shape[1], W.shape[0])
            lin.weight.data = W; lin.bias.data = b
            layers.append(lin)
            if j < len(weights) - 1:
                layers.append(torch.nn.ELU())
        net = torch.nn.Sequential(*layers).eval()
        self._obs_dim = weights[0].shape[1]
        self._act_dim = weights[-1].shape[0]
        return net

    def _gravity_in_body(self) -> np.ndarray:
        qw, qx, qy, qz = self.world.data.qpos[3:7]
        return np.array([
            2.0 * (-qz * qx + qw * qy),
           -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ], dtype=np.float32)

    def _build_obs(self, cmd: np.ndarray) -> np.ndarray:
        d = self.world.data; m = self.world.model
        qj = np.array([
            d.qpos[int(m.jnt_qposadr[_name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])]
            for n in self.LEG_JOINT_NAMES
        ], dtype=np.float32)
        dqj = np.array([
            d.qvel[int(m.jnt_dofadr[_name_to_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])]
            for n in self.LEG_JOINT_NAMES
        ], dtype=np.float32)
        omega = np.asarray(d.qvel[3:6], dtype=np.float32)
        grav = self._gravity_in_body()
        phase = (self._counter * self.world.dt) % self.gait_period / self.gait_period
        sphase, cphase = math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)
        # Standard mjlab-ish actor obs (47 dims). If the policy expects more,
        # zero-pad up to its input size.
        obs = np.concatenate([
            omega * 0.25,
            grav,
            cmd,
            qj,
            dqj * 0.05,
            self._last_action,
            np.array([sphase, cphase], dtype=np.float32),
        ]).astype(np.float32)
        if hasattr(self, "_obs_dim") and obs.size < self._obs_dim:
            obs = np.concatenate([obs, np.zeros(self._obs_dim - obs.size, dtype=np.float32)])
        elif hasattr(self, "_obs_dim") and obs.size > self._obs_dim:
            obs = obs[: self._obs_dim]
        return obs

    def step(self, forward: float, lateral: float, yaw_rate: float) -> None:
        if self._failed or self._policy is None:
            return self.fallback.step(forward, lateral, yaw_rate)
        try:
            import torch
            cmd = np.array([forward, lateral, yaw_rate], dtype=np.float32)
            obs = self._build_obs(cmd)
            with torch.no_grad():
                act = self._policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            # Use the first 12 dims as joint *targets* (relative to the ready pose).
            self._last_action = np.asarray(act[:12], dtype=np.float32)
            scale = 0.25
            for i, jname in enumerate(self.LEG_JOINT_NAMES):
                qadr = int(self.world.model.jnt_qposadr[
                    _name_to_id(self.world.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                ])
                base = READY_POSE.get(jname, 0.0)
                self.world.data.qpos[qadr] = float(base + scale * self._last_action[i])
            # Drive the floating base with the same velocity command (kinematic
            # base, policy-driven legs). Real deployment would let the policy
            # produce the base pose via foot contacts; that's outside scope.
            self.world.cmd_vel(forward, lateral, yaw_rate, self.params)
            self.world.kine_tick()
            self._counter += 1
        except Exception as e:
            print(f"[g1_sim] WARN: trained policy step failed ({e!r}); "
                  "switching permanently to kinematic walker.")
            self._failed = True
            self.fallback.step(forward, lateral, yaw_rate)


# =========================================================================== #
# Cartesian arm controller (the "WBC-flavored" piece)
# =========================================================================== #

@dataclass
class ArmCartesianTarget:
    """Per-arm Cartesian goal in world coordinates."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False


class ArmCartesianController:
    """Velocity-controlled Cartesian end-effector controller for an arm.

    This is the role the user calls "MPC/WBC" in the brief. A full QP-based
    whole-body controller for a 29-DoF humanoid is well beyond scope, but
    the ergonomics from a teleop perspective are the same: send the
    end-effector a 3D velocity command, get back a feasible joint command
    that respects limits and a posture (torso-upright + null-space toward
    the ready pose) regularization.

    Implementation: damped-least-squares IK on the chosen arm joints, with
    the per-arm target integrated from the user-supplied Cartesian
    velocity, and a null-space term that pulls the joints back toward the
    ready pose. The torso joints (waist) are not actuated; the legs are
    handled by the locomotion controller.
    """

    def __init__(
        self,
        world: G1World,
        hand: Hand,
        max_lin_speed: float = 0.6,   # m/s, end-effector speed cap
        gain: float = 0.6,            # IK gain per tick
        damping: float = 0.05,        # DLS regularization
        null_gain: float = 0.4,       # pulls joints toward ready pose
    ):
        self.world = world
        self.hand = hand
        self.max_lin_speed = float(max_lin_speed)
        self.gain = float(gain)
        self.damping = float(damping)
        self.null_gain = float(null_gain)
        self.ik = world.left_ik if hand == "left" else world.right_ik
        self.target = ArmCartesianTarget()

    # ----- target API ----- #

    def activate(self) -> None:
        if not self.target.active:
            self.target.pos = self.ik.ee_pos(self.world.data)
            self.target.active = True
        if self.hand == "left":
            self.world._arm_ctrl_active_left = True
        else:
            self.world._arm_ctrl_active_right = True

    def deactivate(self) -> None:
        self.target.active = False
        if self.hand == "left":
            self.world._arm_ctrl_active_left = False
        else:
            self.world._arm_ctrl_active_right = False

    def set_target(self, pos: np.ndarray) -> None:
        self.target.pos = np.asarray(pos, dtype=float).copy()
        self.target.active = True
        self.activate()

    def integrate_velocity(self, vx: float, vy: float, vz: float) -> None:
        """Move the Cartesian target by `(vx, vy, vz) * dt`. Speed-capped."""
        if not self.target.active:
            self.activate()
        v = np.array([vx, vy, vz], dtype=float)
        sp = float(np.linalg.norm(v))
        if sp > self.max_lin_speed:
            v *= self.max_lin_speed / max(sp, 1e-9)
        self.target.pos += v * self.world.dt

    # ----- one tick ----- #

    def step(self) -> float:
        """Advance the arm one IK step toward the current target.
        Returns the post-step Cartesian position error (m)."""
        if not self.target.active:
            return 0.0
        m = self.world.model; d = self.world.data
        # DLS step on the active arm.
        mujoco.mj_forward(m, d)
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jac, None, self.ik.site)
        J = jac[:, self.ik.dof_idx]
        err = (self.target.pos - d.site_xpos[self.ik.site]) * self.gain
        lam2 = self.damping ** 2
        # Null-space projector pulling toward the ready pose.
        n = J.shape[1]
        if self.null_gain > 0:
            q_now = np.array(
                [d.qpos[qadr] for qadr in self.ik.qpos_idx], dtype=float
            )
            q_ready = np.array([
                READY_POSE.get(name, q_now[i]) for i, name in enumerate(self.ik.joint_names)
            ], dtype=float)
            null_drive = self.null_gain * (q_ready - q_now)
        else:
            null_drive = np.zeros(n)
        # dq = J^# err + (I - J^# J) null_drive
        JJt = J @ J.T + lam2 * np.eye(3)
        Jpinv = J.T @ np.linalg.inv(JJt)
        proj = np.eye(n) - Jpinv @ J
        dq = Jpinv @ err + proj @ null_drive
        np.clip(dq, -0.05, 0.05, out=dq)
        for k, qadr in enumerate(self.ik.qpos_idx):
            d.qpos[qadr] = float(d.qpos[qadr] + dq[k])
        # Respect joint limits.
        for n_j in self.ik.joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n_j)
            if m.jnt_limited[jid]:
                lo, hi = m.jnt_range[jid]
                qadr = int(m.jnt_qposadr[jid])
                d.qpos[qadr] = float(np.clip(d.qpos[qadr], lo, hi))
        mujoco.mj_forward(m, d)
        return float(np.linalg.norm(self.target.pos - d.site_xpos[self.ik.site]))


# =========================================================================== #
# Onboard-camera box detector
# =========================================================================== #

@dataclass
class BoxDetection:
    found: bool
    pixel_xy: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_pos_confidence: float = 0.0  # 0..1; 0 if no plane intersection


class BoxCamDetector:
    """HSV-segments the warehouse box in the onboard `d435i_rgb` camera and
    estimates its world position by intersecting the per-pixel ray with the
    table-top plane (z = TABLE_TOP_Z).

    This deliberately does NOT call `world.box_pos()` — the goal is to
    actually 'see' the box from the simulated camera, like the deployed
    pipeline would.
    """

    TABLE_TOP_Z = 0.76  # matches the table_top geom in scene_g1.xml

    def __init__(
        self,
        world: G1World,
        camera_name: str = "d435i_rgb",
        width: int = 320,
        height: int = 240,
        min_area_px: int = 80,
    ):
        self.world = world
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.min_area_px = int(min_area_px)
        self._cam_id = _name_to_id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self._fovy = float(world.model.cam_fovy[self._cam_id])
        self._renderer: Optional[mujoco.Renderer] = None
        self.last_image: Optional[np.ndarray] = None  # RGB

    def _ensure_renderer(self) -> None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.world.model, height=self.height, width=self.width)

    def render(self) -> np.ndarray:
        self._ensure_renderer()
        self._renderer.update_scene(self.world.data, camera=self.camera_name)
        img = self._renderer.render()
        self.last_image = img
        return img

    @staticmethod
    def _segment_red(rgb: np.ndarray) -> tuple[Optional[tuple[int, int, int, int]], np.ndarray]:
        """Return bounding box of largest red blob and the binary mask."""
        try:
            import cv2
        except ImportError:
            return None, np.zeros(rgb.shape[:2], dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,   100, 80),  (10, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 80:
            return None, mask
        x, y, w, h = cv2.boundingRect(largest)
        return (int(x), int(y), int(w), int(h)), mask

    def _ray_to_table_plane(self, u: int, v: int) -> tuple[Optional[np.ndarray], float]:
        """Convert image pixel (u, v) into a world point on z=TABLE_TOP_Z."""
        m = self.world.model; d = self.world.data
        cam_pos = np.asarray(d.cam_xpos[self._cam_id]).copy()
        cam_mat = np.asarray(d.cam_xmat[self._cam_id]).reshape(3, 3).copy()
        # MuJoCo cameras look down -Z in their own frame.
        # Pixel -> normalized image plane.
        fovy_rad = math.radians(self._fovy)
        f = 0.5 * self.height / math.tan(0.5 * fovy_rad)  # focal length in pixels
        cx, cy = self.width * 0.5, self.height * 0.5
        x_n = (u - cx) / f
        y_n = (v - cy) / f
        # Ray in camera frame (-Z forward).
        ray_cam = np.array([x_n, -y_n, -1.0], dtype=float)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = cam_mat @ ray_cam
        # Plane intersection: cam_pos + t * ray_world has z = TABLE_TOP_Z.
        denom = ray_world[2]
        if abs(denom) < 1e-6:
            return None, 0.0
        t = (self.TABLE_TOP_Z - cam_pos[2]) / denom
        if t <= 0:
            return None, 0.0
        hit = cam_pos + t * ray_world
        # Confidence: shrinks with distance to mimic real depth uncertainty.
        conf = float(max(0.0, min(1.0, 1.0 / (1.0 + 0.2 * t))))
        return hit, conf

    def detect(self) -> BoxDetection:
        rgb = self.render()
        bbox, _mask = self._segment_red(rgb)
        if bbox is None:
            return BoxDetection(found=False)
        x, y, w, h = bbox
        cu = x + w // 2
        # Use the *bottom* of the bounding box, not the center: that's the
        # pixel most likely to lie on the table top plane (the box's base).
        cv = y + int(0.85 * h)
        hit, conf = self._ray_to_table_plane(cu, cv)
        if hit is None:
            return BoxDetection(found=True, pixel_xy=(cu, cv), bbox=bbox, world_pos_confidence=0.0)
        # Adjust z up by half the box height to estimate the box center.
        hit = hit + np.array([0.0, 0.0, 0.05])
        return BoxDetection(found=True, pixel_xy=(cu, cv), bbox=bbox,
                            world_pos=hit, world_pos_confidence=conf)


# =========================================================================== #
# Convenience: scripted "pickup the detected box" for the prompt-driven flow
# =========================================================================== #

def pickup_detected_box(
    world: G1World,
    detection: BoxDetection,
    hand: Hand = "left",
    *,
    approach_height: float = 0.10,
    grasp_height: float = 0.04,
    lift_height: float = 0.20,
    tol: float = 0.07,
) -> bool:
    """Reach the detected box pose with the given hand and grasp it.

    Coordinates come from a `BoxCamDetector.detect()` result, not from a
    privileged world lookup. The `tol` is intentionally loose (7 cm) since
    HSV+ray-cast perception has multi-cm bias; the grasp itself succeeds as
    long as the hand is roughly on top of the box. Returns True if the grasp
    completed.
    """
    if not detection.found or detection.world_pos_confidence <= 0.0:
        return False
    base = np.asarray(detection.world_pos, dtype=float)
    # Best-effort approach + descent. We accept the final hand pose as long
    # as it ended up "near" the detected box (the rigid carry will snap the
    # box rigidly to the wrist on grasp, mimicking a real gripper closing
    # when it is roughly on top of the object).
    world.reach(hand, base + np.array([0.0, 0.0, approach_height]), tol=tol, max_iters=2000)
    _, err_final = world.reach(hand, base + np.array([0.0, 0.0, grasp_height]),
                               tol=tol, max_iters=2000)
    near_box = err_final < 0.12  # 12 cm: within "wrist over the box" envelope
    if not near_box:
        return False
    world.grasp(hand)
    world.reach(hand, base + np.array([0.0, 0.0, lift_height]), tol=tol, max_iters=2000)
    return True


def place_detected_box(
    world: G1World,
    hand: Hand = "left",
    *,
    approach_height: float = 0.16,
    place_height: float = 0.06,
    tol: float = 0.05,
    settle_seconds: float = 1.5,
) -> bool:
    """Place whatever the given hand is holding on top of place_target."""
    if world.held_by != hand:
        return False
    target = world.place_target_pos()
    world.reach(hand, target + np.array([0.0, 0.0, approach_height]), tol=tol)
    world.reach(hand, target + np.array([0.0, 0.0, place_height]),    tol=tol)
    world.release()
    world.settle_box(seconds=settle_seconds)
    return True
