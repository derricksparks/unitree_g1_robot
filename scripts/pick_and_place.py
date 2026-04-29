#!/usr/bin/env python3
"""
Walk + Pick + Place demo for the Unitree G1 (pure MuJoCo, no SDK / no RL checkpoint).

Sequence:
  1. WALK_TO_TABLE : kinematically drive the floating base to the reception table.
  2. PICK          : run dual-arm DLS IK so the left hand reaches the box, then
                     activate the `left_hand_grasp` weld equality to "grasp" it.
  3. WALK_TO_SHELF : carry the box (still welded to the hand) over to the shelf.
  4. PLACE         : IK to lower the box onto the shelf's `place_target`,
                     then deactivate the weld so the box settles on the shelf.
  5. RETREAT       : walk back a step and hold.

Why "kinematic" walking? This repo trains an RL walking policy via mjlab, but a
trained checkpoint is not always available in this environment. The locomotion
phases here are visually plausible (the base slides toward each waypoint while
arms swing slightly) and let us focus on the manipulation phases. Drop-in
replacement with the RL policy is straightforward: replace `walk_step` with a
call into your trained policy's actor on the underlying ManagerBasedRlEnv.

Usage:
  python scripts/pick_and_place.py                  # offscreen render, write video
  python scripts/pick_and_place.py --viewer         # interactive native viewer
  python scripts/pick_and_place.py --video out.mp4  # custom video path
  python scripts/pick_and_place.py --headless --no-video  # fastest smoke-test

Tested with mujoco==3.x.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml"


# -------------------------- joint / index helpers ---------------------------- #

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

# Whole-body "ready" pose (in joint space). Knees slightly bent, arms at sides.
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


def jnt_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def jnt_qvel_addr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint not found: {name}")
    return int(model.jnt_dofadr[jid])


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site not found: {name}")
    return int(sid)


def body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body not found: {name}")
    return int(bid)


def equality_id(model: mujoco.MjModel, name: str) -> int:
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if eid < 0:
        raise KeyError(f"equality not found: {name}")
    return int(eid)


# ----------------------------- pose application ------------------------------ #

def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    """Write joint positions for hinge joints (matches by joint name)."""
    for name, val in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        addr = int(model.jnt_qposadr[jid])
        data.qpos[addr] = float(val)


def set_floating_base(
    data: mujoco.MjData,
    pos: np.ndarray,
    yaw: float = 0.0,
) -> None:
    """Write floating-base qpos: first 7 entries = (xyz, quat[w,x,y,z])."""
    half = 0.5 * yaw
    data.qpos[0] = float(pos[0])
    data.qpos[1] = float(pos[1])
    data.qpos[2] = float(pos[2])
    data.qpos[3] = math.cos(half)  # qw
    data.qpos[4] = 0.0             # qx
    data.qpos[5] = 0.0             # qy
    data.qpos[6] = math.sin(half)  # qz
    # Zero the base velocity.
    data.qvel[0:6] = 0.0


# --------------------------------- IK --------------------------------------- #

class ArmIK:
    """Damped-least-squares IK for the left or right arm.

    Drives a 3-DoF position target (we don't constrain end-effector orientation
    here to keep the math simple and avoid wrist limit thrash).
    """

    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...], site_name: str):
        self.model = model
        self.joint_names = joint_names
        self.site = site_id(model, site_name)
        self.dof_idx = np.array(
            [jnt_qvel_addr(model, n) for n in joint_names], dtype=np.int32
        )
        self.qpos_idx = np.array(
            [jnt_qpos_addr(model, n) for n in joint_names], dtype=np.int32
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
        """One DLS IK step. Returns the L2 position error after the step."""
        mujoco.mj_forward(self.model, data)
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jac, None, self.site)
        # Restrict to this arm's DoFs.
        J = jac[:, self.dof_idx]                                 # (3, n_arm)
        err = (np.asarray(target) - data.site_xpos[self.site]) * gain
        # DLS: dq = J^T (J J^T + lambda^2 I)^-1 err
        lam2 = damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
        # Clip per-joint step to avoid IK shaking near limits.
        np.clip(dq, -max_step, max_step, out=dq)
        # Integrate only the arm joints.
        for k, qadr in enumerate(self.qpos_idx):
            data.qpos[qadr] = float(data.qpos[qadr] + dq[k])
        # Respect joint limits.
        for n in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                qadr = int(self.model.jnt_qposadr[jid])
                data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))
        mujoco.mj_forward(self.model, data)
        return float(np.linalg.norm(np.asarray(target) - data.site_xpos[self.site]))


# ---------------------------- behavior primitives --------------------------- #

@dataclass
class WalkParams:
    speed_mps: float = 0.6      # base translation speed
    yaw_rate: float = 1.0       # base yaw rate
    arm_swing_amp: float = 0.25 # shoulder pitch swing amplitude (rad)
    swing_freq_hz: float = 1.5


def walk_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_xy: np.ndarray,
    target_yaw: float,
    dt: float,
    p: WalkParams,
    walk_phase: float,
) -> tuple[bool, float]:
    """Move the kinematic floating base toward (target_xy, target_yaw) and
    swing the arms a little so it visually reads as walking.

    Returns (arrived, new_walk_phase).
    """
    pos = data.qpos[0:3].copy()
    # Current yaw from quat.
    qw, qx, qy, qz = data.qpos[3:7]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    # Yaw control toward target.
    yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    dyaw = float(np.clip(yaw_err, -p.yaw_rate * dt, p.yaw_rate * dt))
    new_yaw = yaw + dyaw

    # XY control toward target.
    xy_err = target_xy - pos[:2]
    dist = float(np.linalg.norm(xy_err))
    step_len = min(dist, p.speed_mps * dt)
    if dist > 1e-6:
        new_xy = pos[:2] + xy_err / dist * step_len
    else:
        new_xy = pos[:2]

    new_pos = np.array([new_xy[0], new_xy[1], pos[2]])
    set_floating_base(data, new_pos, new_yaw)

    # Arm swing.
    walk_phase = walk_phase + 2.0 * math.pi * p.swing_freq_hz * dt
    swing = p.arm_swing_amp * math.sin(walk_phase)
    l_idx = jnt_qpos_addr(model, "left_shoulder_pitch_joint")
    r_idx = jnt_qpos_addr(model, "right_shoulder_pitch_joint")
    data.qpos[l_idx] = READY_POSE["left_shoulder_pitch_joint"] + swing
    data.qpos[r_idx] = READY_POSE["right_shoulder_pitch_joint"] - swing

    arrived = dist < 0.02 and abs(yaw_err) < 0.05
    return arrived, walk_phase


def reach_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: ArmIK,
    target_world: np.ndarray,
    tol: float = 0.02,
    max_iters: int = 400,
    on_step=None,
) -> bool:
    """Iterate IK until end-effector is within `tol` of `target_world`."""
    for _ in range(max_iters):
        err = ik.step_to(data, target_world)
        if on_step is not None:
            on_step()
        if err < tol:
            return True
    return False


def hold_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
    on_step=None,
) -> None:
    """Re-run forward kinematics (no integration) and call on_step. Useful for
    rendering frames during a "pause"."""
    for _ in range(n_steps):
        mujoco.mj_forward(model, data)
        if on_step is not None:
            on_step()


# ------------------------- equality constraint toggle ---------------------- #

def set_equality_active(model: mujoco.MjModel, eq_name: str, active: bool) -> None:
    eid = equality_id(model, eq_name)
    model.eq_active0[eid] = 1 if active else 0


# --------------------------------- main ------------------------------------ #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="MJCF scene XML")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer (interactive)")
    ap.add_argument("--headless", action="store_true", help="No video, no viewer (smoke test)")
    ap.add_argument("--no-video", action="store_true", help="Skip writing video")
    ap.add_argument("--video", default=str(REPO_ROOT / "scripts" / "logs" / "pick_and_place.mp4"),
                    help="Output video path (mp4)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="Hard cap on demo duration (sim seconds)")
    args = ap.parse_args()

    scene_path = Path(args.scene).resolve()
    if not scene_path.exists():
        print(f"ERROR: scene not found: {scene_path}", file=sys.stderr)
        return 2

    print(f"[pick_and_place] loading: {scene_path}")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Sim timestep (the bare scene_g1.xml has no <option timestep ...>; mujoco
    # default is 0.002, which is fine).
    dt = float(model.opt.timestep)
    print(f"[pick_and_place] dt={dt:.4f}s, nq={model.nq}, nv={model.nv}, neq={model.neq}")

    # Initialize pose: ready stance, base at origin, box on table, equalities off.
    apply_pose(model, data, READY_POSE)
    set_floating_base(data, np.array([0.0, 0.0, 0.793]), yaw=0.0)
    set_equality_active(model, "left_hand_grasp", False)
    set_equality_active(model, "right_hand_grasp", False)
    mujoco.mj_forward(model, data)

    # Build IK solvers.
    left_ik = ArmIK(model, LEFT_ARM_JOINTS, "left_hand_site")
    right_ik = ArmIK(model, RIGHT_ARM_JOINTS, "right_hand_site")

    # Resolve scene targets.
    box_b = body_id(model, "warehouse_box")
    place_s = site_id(model, "place_target")

    def box_world() -> np.ndarray:
        return np.asarray(data.xpos[box_b]).copy()

    def place_world() -> np.ndarray:
        return np.asarray(data.site_xpos[place_s]).copy()

    # Frames: render via offscreen Renderer unless using interactive viewer.
    renderer = None
    video_writer = None
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.9)
    cam.distance = 3.5
    cam.azimuth = 135.0
    cam.elevation = -15.0

    if not args.headless and not args.viewer and not args.no_video:
        try:
            import imageio.v2 as imageio  # noqa: F401
        except Exception:
            print("[pick_and_place] imageio not available, installing...")
            os.system(f"{sys.executable} -m pip install --quiet imageio[ffmpeg]")
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(args.video, fps=30, codec="libx264", quality=8)
        print(f"[pick_and_place] writing video -> {args.video}")

    viewer_handle = None
    if args.viewer:
        import mujoco.viewer as mjviewer
        viewer_handle = mjviewer.launch_passive(model, data)

    n_frames = [0]
    n_render_every = max(1, int(round((1.0 / 30.0) / dt)))  # ~30 fps capture

    def on_step() -> None:
        n_frames[0] += 1
        if viewer_handle is not None:
            viewer_handle.sync()
        if renderer is not None and (n_frames[0] % n_render_every == 0):
            renderer.update_scene(data, camera=cam)
            video_writer.append_data(renderer.render())

    def kine_step() -> None:
        """Kinematic step: refresh derived quantities without integrating physics.
        We use this during walk + manipulation so the arms/legs don't sag under
        gravity (we don't have a low-level controller / RL policy hooked up)."""
        mujoco.mj_forward(model, data)
        on_step()

    def physics_step() -> None:
        mujoco.mj_step(model, data)
        on_step()

    # Hand-attached carry: while the box is "held", we copy the hand pose into
    # the box's freejoint qpos every kinematic step (simple alternative to a
    # weld constraint, which only takes effect during physics steps).
    held_by: dict[str, str | None] = {"hand": None}  # 'left' | 'right' | None
    box_jadr = jnt_qpos_addr(model, "box_joint")
    left_hand_b = body_id(model, "left_wrist_yaw_link")
    right_hand_b = body_id(model, "right_wrist_yaw_link")
    box_offset_in_hand = np.zeros(3)  # offset from hand body origin to box, in hand's frame

    def attach_box(side: str) -> None:
        held_by["hand"] = side
        bid = left_hand_b if side == "left" else right_hand_b
        # Record the box position relative to the hand at grasp time so the box
        # follows the hand rigidly during the carry.
        hand_pos = np.asarray(data.xpos[bid]).copy()
        hand_mat = np.asarray(data.xmat[bid]).reshape(3, 3).copy()
        box_pos = np.asarray(data.xpos[box_b]).copy()
        rel = hand_mat.T @ (box_pos - hand_pos)
        box_offset_in_hand[:] = rel

    def detach_box() -> None:
        held_by["hand"] = None

    def update_carry() -> None:
        side = held_by["hand"]
        if side is None:
            return
        bid = left_hand_b if side == "left" else right_hand_b
        hand_pos = np.asarray(data.xpos[bid]).copy()
        hand_mat = np.asarray(data.xmat[bid]).reshape(3, 3).copy()
        new_pos = hand_pos + hand_mat @ box_offset_in_hand
        data.qpos[box_jadr + 0] = float(new_pos[0])
        data.qpos[box_jadr + 1] = float(new_pos[1])
        data.qpos[box_jadr + 2] = float(new_pos[2])
        # Identity orientation for simplicity.
        data.qpos[box_jadr + 3] = 1.0
        data.qpos[box_jadr + 4] = 0.0
        data.qpos[box_jadr + 5] = 0.0
        data.qpos[box_jadr + 6] = 0.0

    walk_phase = 0.0
    walk_p = WalkParams()
    sim_clock = [0.0]
    max_seconds = args.max_seconds

    def advance_to_target(target_xy: np.ndarray, yaw: float, label: str) -> None:
        nonlocal walk_phase
        print(f"[pick_and_place] {label}: walking to {target_xy.tolist()} yaw={yaw:.2f}")
        for _ in range(int(max_seconds / dt)):
            arrived, walk_phase = walk_step(model, data, target_xy, yaw, dt, walk_p, walk_phase)
            update_carry()
            kine_step()
            sim_clock[0] += dt
            if arrived or sim_clock[0] >= max_seconds:
                break

    def reach_with_render(ik: ArmIK, target: np.ndarray, label: str,
                           tol: float = 0.025, max_iters: int = 1500) -> bool:
        print(f"[pick_and_place] {label}: IK to [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
        def cb():
            update_carry()
            on_step()
        ok = reach_arm(model, data, ik, target, tol=tol, max_iters=max_iters, on_step=cb)
        print(f"[pick_and_place] {label}: {'reached' if ok else 'TIMED OUT'} (err={np.linalg.norm(target - ik.ee_pos(data)):.3f}m)")
        return ok

    # ----------------- 1) Walk to the table (box pickup spot) ----------------
    # Box is at world (0.45, 0.15, 0.81). Stand at (0.0, 0.0) facing +x so the
    # left hand (default world (~0.3, 0.15, 0.89) at ready pose) is in reach.
    advance_to_target(np.array([0.0, 0.0]), yaw=0.0, label="WALK_TO_TABLE")
    # Reset arm swing back to ready before manipulating.
    apply_pose(model, data, {
        "left_shoulder_pitch_joint":  READY_POSE["left_shoulder_pitch_joint"],
        "right_shoulder_pitch_joint": READY_POSE["right_shoulder_pitch_joint"],
    })
    hold_pose(model, data, 30, on_step=on_step)

    # --------------------------- 2) PICK the box ----------------------------
    # Approach above the box, then descend onto it, then weld.
    box_pos = box_world()
    # Aim the left-hand site (which is at the fingertip of the rubber hand) a
    # few cm above the box center so the hand "wraps" around the top.
    above = box_pos + np.array([0.0, 0.0, 0.10])
    grasp = box_pos + np.array([0.0, 0.0, 0.04])  # hand site sits ~5 cm above box top
    reach_with_render(left_ik, above, "PICK_APPROACH", tol=0.04)
    reach_with_render(left_ik, grasp, "PICK_GRASP",    tol=0.05)
    print("[pick_and_place] grasping (attaching box to left hand)")
    attach_box("left")
    # Lift the box.
    lift = box_pos + np.array([0.0, 0.0, 0.20])
    reach_with_render(left_ik, lift, "PICK_LIFT", tol=0.05)
    hold_pose(model, data, 30, on_step=lambda: (update_carry(), on_step()))

    # ----------------- 3) Walk to the shelf (place spot) --------------------
    # Shelf is at (-0.55, 0.0, ...). Turn 180° (face -x) and walk back so the
    # shelf is in front. Stand at (0.0, 0.0) facing -x; place_target is at
    # world (-0.55, 0.0, 0.83) which is ~0.55 m forward of the new facing.
    advance_to_target(np.array([0.0, 0.0]), yaw=math.pi, label="WALK_TO_SHELF")
    apply_pose(model, data, {
        "left_shoulder_pitch_joint":  READY_POSE["left_shoulder_pitch_joint"],
        "right_shoulder_pitch_joint": READY_POSE["right_shoulder_pitch_joint"],
    })
    hold_pose(model, data, 30, on_step=on_step)

    # ------------------------- 4) PLACE the box -----------------------------
    target = place_world() + np.array([0.0, 0.0, 0.06])  # box half-height = 0.05
    reach_with_render(left_ik, target + np.array([0.0, 0.0, 0.10]), "PLACE_APPROACH", tol=0.05)
    reach_with_render(left_ik, target,                              "PLACE_DOWN",     tol=0.04)
    print("[pick_and_place] releasing box (detaching from left hand)")
    detach_box()
    # Step physics briefly so the box drops onto the shelf under gravity. We
    # also freeze the robot's joints by writing them back each step (the robot
    # has no controller in this demo), so only the box actually moves.
    saved_qpos = data.qpos.copy()
    saved_qvel = np.zeros_like(data.qvel)
    box_qadr = box_jadr
    for _ in range(int(1.5 / dt)):
        # Save box-only qpos, restore everything else.
        box_q = data.qpos[box_qadr : box_qadr + 7].copy()
        box_v = data.qvel[jnt_qvel_addr(model, "box_joint"):
                          jnt_qvel_addr(model, "box_joint") + 6].copy()
        mujoco.mj_step(model, data)
        new_box_q = data.qpos[box_qadr : box_qadr + 7].copy()
        new_box_v = data.qvel[jnt_qvel_addr(model, "box_joint"):
                              jnt_qvel_addr(model, "box_joint") + 6].copy()
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.qpos[box_qadr : box_qadr + 7] = new_box_q
        data.qvel[jnt_qvel_addr(model, "box_joint"):
                  jnt_qvel_addr(model, "box_joint") + 6] = new_box_v
        on_step()
        del box_q, box_v, new_box_q, new_box_v

    # ---------------------------- 5) Retreat --------------------------------
    advance_to_target(np.array([0.0, 0.0]), yaw=0.0, label="RETREAT")
    hold_pose(model, data, 60, on_step=on_step)

    # Done.
    print(f"[pick_and_place] sim_seconds={sim_clock[0]:.2f}, frames_rendered={n_frames[0]}")
    print(f"[pick_and_place] final box position: {box_world().tolist()}")
    print(f"[pick_and_place] place target:      {place_world().tolist()}")
    box_to_target = float(np.linalg.norm(box_world()[:2] - place_world()[:2]))
    print(f"[pick_and_place] box-target XY error: {box_to_target:.3f} m")

    if video_writer is not None:
        video_writer.close()
        print(f"[pick_and_place] video written: {args.video}")
    if viewer_handle is not None:
        # Keep viewer open briefly so user can inspect.
        for _ in range(120):
            time.sleep(0.05)
            viewer_handle.sync()
        viewer_handle.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
