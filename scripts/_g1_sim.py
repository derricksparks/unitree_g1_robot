"""Shared simulation primitives for the Unitree G1 + warehouse scene.

This module encapsulates the kinematic walk + dual-arm DLS IK + grasp/carry
logic used by both the canned demo (`pick_and_place.py`) and the interactive
CLI (`sim_cli.py`). Pure MuJoCo, no Unitree SDK and no trained checkpoint
required.
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
