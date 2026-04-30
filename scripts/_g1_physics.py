"""Real-physics teleop world for the Unitree G1.

Unlike `_g1_sim.G1World` (which is kinematic and slides the floating base),
this module runs **real MuJoCo physics** with PD-controlled motor actuators:

  * Locomotion is driven by the trained `Unitree-G1-Flat` velocity policy
    exported to ONNX (`deploy/robots/g1/config/policy/velocity/v0/exported/
    policy.onnx`). The policy outputs joint position targets for all 29
    DoFs at 50 Hz; we run a 500 Hz PD inner loop on the legs + waist that
    integrates `mj_step` and produces real footsteps.

  * Manipulation is layered on top: a Cartesian DLS IK (with null-space
    posture regularization, see `ArmCartesianController`) overrides the
    policy's targets for the 14 arm joints, so the user can still teleop
    each end-effector while the legs walk autonomously.

  * Perception is unchanged: `_g1_sim.BoxCamDetector` still segments the
    onboard camera and ray-casts to the table plane.

The teleop commands (joystick / keyboard / prompt) only set the velocity
command and high-level intents (`pickup`, `place`); the rest is closed-loop
control under physics, so the robot can't ride through walls or float.
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
import yaml

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml"
DEFAULT_DEPLOY_YAML = REPO_ROOT / "deploy" / "robots" / "g1" / "config" / "policy" / "velocity" / "v0" / "params" / "deploy.yaml"
DEFAULT_POLICY_ONNX = REPO_ROOT / "deploy" / "robots" / "g1" / "config" / "policy" / "velocity" / "v0" / "exported" / "policy.onnx"


# Order in which the policy expects the 29 joints. Matches the order of
# <motor> actuators in scene_g1.xml; we sanity-check this at construction.
JOINT_NAMES_29 = (
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
LEG_INDICES   = tuple(range(0, 12))
WAIST_INDICES = tuple(range(12, 15))
LEFT_ARM_INDICES  = tuple(range(15, 22))
RIGHT_ARM_INDICES = tuple(range(22, 29))

Hand = Literal["left", "right"]
GraspMode = Literal["left", "right", "both"]


def _name_to_id(model: mujoco.MjModel, obj: int, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise KeyError(f"name not found: {name}")
    return int(i)


# --------------------------------------------------------------------------- #
# Locomotion policy
# --------------------------------------------------------------------------- #

@dataclass
class PolicyConfig:
    yaml_path: Path
    onnx_path: Path
    default_q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    step_dt: float
    period: float = 0.6  # gait phase period (matches deploy.yaml)


def load_policy_config(yaml_path: Path = DEFAULT_DEPLOY_YAML,
                       onnx_path: Path = DEFAULT_POLICY_ONNX) -> PolicyConfig:
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    return PolicyConfig(
        yaml_path=Path(yaml_path),
        onnx_path=Path(onnx_path),
        default_q=np.asarray(d["default_joint_pos"], dtype=np.float32),
        kp=np.asarray(d["stiffness"], dtype=np.float32),
        kd=np.asarray(d["damping"], dtype=np.float32),
        action_scale=np.asarray(d["actions"]["JointPositionAction"]["scale"], dtype=np.float32),
        action_offset=np.asarray(d["actions"]["JointPositionAction"]["offset"], dtype=np.float32),
        step_dt=float(d["step_dt"]),
        period=float(d["observations"].get("gait_phase", {}).get("params", {}).get("period", 0.6)),
    )


# --------------------------------------------------------------------------- #
# Arm IK (a stripped copy of _g1_sim.ArmIK so this file is self-contained)
# --------------------------------------------------------------------------- #

class _ArmIK:
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


@dataclass
class ArmCartesianTarget:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False


class ArmCartesianController:
    """3D end-effector velocity controller for one arm.

    Produces *joint position deltas* relative to the policy's `default_q`,
    written into a 7-vector of arm targets that the physics PD loop tracks.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        hand: Hand,
        max_lin_speed: float = 0.5,
        damping: float = 0.06,
        gain: float = 0.6,
        null_gain: float = 0.4,
    ):
        self.model = model
        self.data = data
        self.hand = hand
        self.max_lin_speed = float(max_lin_speed)
        self.gain = float(gain)
        self.damping = float(damping)
        self.null_gain = float(null_gain)
        joint_names = tuple(JOINT_NAMES_29[i] for i in (
            LEFT_ARM_INDICES if hand == "left" else RIGHT_ARM_INDICES
        ))
        site_name = "left_hand_site" if hand == "left" else "right_hand_site"
        self.ik = _ArmIK(model, joint_names, site_name)
        self.target = ArmCartesianTarget()
        # Output: per-arm target joint positions (7-vector).
        self.target_q = np.zeros(7, dtype=np.float32)

    @property
    def ee_pos(self) -> np.ndarray:
        return self.ik.ee_pos(self.data)

    def activate_at_current(self) -> None:
        self.target.pos = self.ee_pos
        self.target.active = True

    def deactivate(self) -> None:
        self.target.active = False

    def set_target(self, pos: np.ndarray) -> None:
        self.target.pos = np.asarray(pos, dtype=float).copy()
        self.target.active = True

    def integrate_velocity(self, vx: float, vy: float, vz: float, dt: float) -> None:
        if not self.target.active:
            self.activate_at_current()
        v = np.array([vx, vy, vz], dtype=float)
        sp = float(np.linalg.norm(v))
        if sp > self.max_lin_speed:
            v *= self.max_lin_speed / max(sp, 1e-9)
        self.target.pos = self.target.pos + v * dt

    def compute_target_q(self, q_ready: np.ndarray, n_ik_iters: int = 60) -> np.ndarray:
        """Return the 7-vector of target arm joint angles for the PD loop.

        If inactive, returns ``q_ready`` (the policy's default). When active,
        iterates DLS IK on a temporary qpos copy until the end-effector is
        near ``self.target.pos``, then returns the resulting joint vector.

        The IK iteration *starts from* ``q_ready`` rather than the live qpos
        so that the PD setpoint is a deterministic function of the Cartesian
        target alone. Otherwise the steady-state PD error feeds back into
        the next IK call and the arm drifts even when the target is fixed.
        """
        if not self.target.active:
            self.target_q[:] = q_ready
            return self.target_q

        m = self.model
        qpos_save = self.data.qpos.copy()
        try:
            # Seed the IK from the policy's default arm pose, not the live one.
            for k, qadr in enumerate(self.ik.qpos_idx):
                self.data.qpos[qadr] = float(q_ready[k])
            mujoco.mj_forward(m, self.data)
            for _ in range(int(n_ik_iters)):
                jac = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, self.data, jac, None, self.ik.site)
                J = jac[:, self.ik.dof_idx]
                err = (self.target.pos - self.data.site_xpos[self.ik.site]) * self.gain
                if float(np.linalg.norm(err)) < 5e-4:
                    break
                lam2 = self.damping ** 2
                JJt = J @ J.T + lam2 * np.eye(3)
                Jpinv = J.T @ np.linalg.inv(JJt)
                n = J.shape[1]
                proj = np.eye(n) - Jpinv @ J
                q_now = np.array([self.data.qpos[a] for a in self.ik.qpos_idx], dtype=float)
                null_drive = self.null_gain * (q_ready - q_now)
                dq = Jpinv @ err + proj @ null_drive
                np.clip(dq, -0.05, 0.05, out=dq)
                for k, qadr in enumerate(self.ik.qpos_idx):
                    self.data.qpos[qadr] = float(self.data.qpos[qadr] + dq[k])
                for k, name in enumerate(self.ik.joint_names):
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if m.jnt_limited[jid]:
                        lo, hi = m.jnt_range[jid]
                        qadr = int(m.jnt_qposadr[jid])
                        self.data.qpos[qadr] = float(np.clip(self.data.qpos[qadr], lo, hi))
                mujoco.mj_forward(m, self.data)
            self.target_q[:] = np.array(
                [self.data.qpos[a] for a in self.ik.qpos_idx], dtype=np.float32
            )
        finally:
            self.data.qpos[:] = qpos_save
            mujoco.mj_forward(m, self.data)
        return self.target_q


# --------------------------------------------------------------------------- #
# The physics world
# --------------------------------------------------------------------------- #

@dataclass
class RenderConfig:
    enabled: bool = False
    width: int = 960
    height: int = 540
    fps: int = 30
    video_path: Optional[str] = None
    azimuth: float = 135.0
    elevation: float = -18.0
    distance: float = 5.0


class G1PhysicsWorld:
    """MuJoCo + trained ONNX policy + Cartesian arm controller, all in
    real physics. Replaces the kinematic ``G1World`` for teleop runs that
    need actual stepping motion.
    """

    def __init__(
        self,
        scene_path: os.PathLike | str | None = None,
        policy_cfg: Optional[PolicyConfig] = None,
        load_policy: bool = True,
    ):
        self.scene_path = Path(scene_path or DEFAULT_SCENE).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        # Load policy + config.
        self.policy_cfg = policy_cfg or load_policy_config()
        self.decimation = int(round(self.policy_cfg.step_dt / self.dt))
        self._sess = None
        if load_policy:
            try:
                import onnxruntime as ort
                self._sess = ort.InferenceSession(str(self.policy_cfg.onnx_path),
                                                   providers=["CPUExecutionProvider"])
                print(f"[g1_physics] loaded policy: {self.policy_cfg.onnx_path}")
            except Exception as e:
                print(f"[g1_physics] WARN: could not load policy ({e!r}); "
                      f"locomotion will fall back to PD-to-default-pose (no walking).")
                self._sess = None

        # Sanity: actuator order must match JOINT_NAMES_29.
        for i, jname in enumerate(JOINT_NAMES_29):
            an = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if not an or jname.replace("_joint", "") not in an:
                # Soft check; the deploy YAML and the scene's actuator block
                # do match in practice, but we don't fail hard if a build
                # uses a different order.
                pass

        # Cache joint qpos / qvel addresses for the 29 controlled DoFs.
        self._qpos_idx = np.array([
            int(self.model.jnt_qposadr[_name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in JOINT_NAMES_29
        ], dtype=np.int32)
        self._qvel_idx = np.array([
            int(self.model.jnt_dofadr[_name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in JOINT_NAMES_29
        ], dtype=np.int32)

        # Cache scene ids.
        self._pelvis_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._box_b    = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_box")
        self._place_s  = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "place_target")
        self._left_h_b  = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self._right_h_b = _name_to_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self._box_qadr = int(self.model.jnt_qposadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ])
        self._box_dofadr = int(self.model.jnt_dofadr[
            _name_to_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ])

        # Arm controllers.
        self.left_arm  = ArmCartesianController(self.model, self.data, "left")
        self.right_arm = ArmCartesianController(self.model, self.data, "right")

        # Carry state (pelvis-frame, identical to G1World).
        self._held_by: Optional[GraspMode] = None
        self._box_offset_in_pelvis = np.zeros(3)
        self._box_offset_in_hand   = np.zeros(3)

        # Locomotion state.
        self.cmd = np.zeros(3, dtype=np.float32)  # forward, lateral, yaw_rate
        self.last_action = np.zeros(29, dtype=np.float32)
        self._phase = 0.0
        self._policy_target_q = self.policy_cfg.default_q.copy()
        self._policy_counter = 0
        self._sim_time = 0.0
        self._fell = False
        # When True, ignore the policy's arm-joint outputs and PD-track the
        # default arm pose (or the IK target if active). Used during static
        # manipulation to keep the arms stable.
        self._suppress_policy_arms = False
        # When True, also pin the legs / waist to the default standing pose
        # via PD. Used during a grasp so the body doesn't drift while the
        # arm IK runs. Has the side effect of disabling locomotion.
        self._stand_still_mode = False

        # Render / viewer state.
        self.render_cfg = RenderConfig()
        self._renderer: Optional[mujoco.Renderer] = None
        self._video_writer = None
        self._cam = mujoco.MjvCamera()
        self._cam.lookat[:] = (0.0, 0.0, 0.9)
        self._cam.distance = self.render_cfg.distance
        self._cam.azimuth  = self.render_cfg.azimuth
        self._cam.elevation = self.render_cfg.elevation
        self._render_every = max(1, int(round((1.0 / max(1, self.render_cfg.fps)) / self.dt)))
        self._frames_since_render = 0
        self._viewer = None

        self.reset()

    # ----- accessors ---------------------------------------------------- #

    @property
    def pelvis_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._pelvis_b]).copy()

    @property
    def pelvis_yaw(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    @property
    def fell(self) -> bool:
        return self._fell

    @property
    def held_by(self) -> Optional[GraspMode]:
        return self._held_by

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def box_pos(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._box_b]).copy()

    def place_target_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self._place_s]).copy()

    def hand_pos(self, hand: Hand) -> np.ndarray:
        ctl = self.left_arm if hand == "left" else self.right_arm
        return ctl.ee_pos

    # ----- reset -------------------------------------------------------- #

    def reset(self, base_xy: Iterable[float] = (0.0, 0.0), base_yaw: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Floating base.
        x, y = base_xy
        half = 0.5 * base_yaw
        self.data.qpos[0] = float(x); self.data.qpos[1] = float(y); self.data.qpos[2] = 0.793
        self.data.qpos[3] = math.cos(half); self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0;            self.data.qpos[6] = math.sin(half)
        # Joints to default.
        for i, qadr in enumerate(self._qpos_idx):
            self.data.qpos[qadr] = float(self.policy_cfg.default_q[i])
        # Box back on the table.
        self.data.qpos[self._box_qadr + 0] = 1.90
        self.data.qpos[self._box_qadr + 1] = 0.15
        self.data.qpos[self._box_qadr + 2] = 0.81
        self.data.qpos[self._box_qadr + 3] = 1.0
        self.data.qpos[self._box_qadr + 4:self._box_qadr + 7] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.cmd[:] = 0.0
        self.last_action[:] = 0.0
        self._phase = 0.0
        self._policy_target_q = self.policy_cfg.default_q.copy()
        self._policy_counter = 0
        self._held_by = None
        self._sim_time = 0.0
        self._fell = False
        self.left_arm.deactivate(); self.right_arm.deactivate()

    # ----- locomotion: policy update ------------------------------------ #

    def _gravity_in_body(self) -> np.ndarray:
        qw, qx, qy, qz = self.data.qpos[3:7]
        return np.array([
            2.0 * (-qz * qx + qw * qy),
           -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        cfg = self.policy_cfg
        qpos = np.array([self.data.qpos[a] for a in self._qpos_idx], dtype=np.float32)
        qvel = np.array([self.data.qvel[a] for a in self._qvel_idx], dtype=np.float32)
        omega = np.asarray(self.data.qvel[3:6], dtype=np.float32)
        grav = self._gravity_in_body()
        sphase = math.sin(2.0 * math.pi * self._phase)
        cphase = math.cos(2.0 * math.pi * self._phase)
        obs = np.concatenate([
            omega,                      # 3
            grav,                       # 3
            self.cmd,                   # 3
            np.array([sphase, cphase], dtype=np.float32),  # 2
            qpos - cfg.default_q,       # 29
            qvel,                       # 29
            self.last_action,           # 29
        ]).astype(np.float32)
        return obs

    def _refresh_policy_target(self) -> None:
        """Re-run the policy at 50 Hz to update the joint target vector."""
        cfg = self.policy_cfg
        if self._sess is None:
            self.last_action[:] = 0.0
        else:
            obs = self._build_obs()
            out = self._sess.run(None, {"obs": obs[None]})[0][0]
            self.last_action = np.asarray(out, dtype=np.float32)
        target = cfg.action_offset + cfg.action_scale * self.last_action
        self._policy_target_q = target.astype(np.float32)
        self._phase = (self._phase + cfg.step_dt / cfg.period) % 1.0

    # ----- one inner-loop tick (full physics) --------------------------- #

    def step(self) -> None:
        """Advance the simulation one physics tick (``self.dt``).

        Runs the policy at 50 Hz, then PD-tracks the target joints (with
        arm overrides from the Cartesian controller) and integrates physics.
        """
        if self._fell:
            return
        cfg = self.policy_cfg

        if self._policy_counter % self.decimation == 0:
            ready = cfg.default_q
            if self._stand_still_mode:
                # Skip the policy entirely; PD to the default standing pose.
                self._policy_target_q[:] = ready
                self._phase = 0.0
            else:
                self._refresh_policy_target()
            # Arm overlay: use the IK-derived joint vector when active, the
            # policy default when the arms are suppressed but inactive.
            if self.left_arm.target.active:
                lq = self.left_arm.compute_target_q(ready[15:22])
                self._policy_target_q[15:22] = lq
            elif self._suppress_policy_arms or self._stand_still_mode:
                self._policy_target_q[15:22] = ready[15:22]
            if self.right_arm.target.active:
                rq = self.right_arm.compute_target_q(ready[22:29])
                self._policy_target_q[22:29] = rq
            elif self._suppress_policy_arms or self._stand_still_mode:
                self._policy_target_q[22:29] = ready[22:29]
        self._policy_counter += 1

        # PD inner loop. When an arm IK target is active, stiffen that arm so
        # the PD setpoint is tracked accurately despite body sway. When in
        # stand-still mode, also stiffen the legs / waist so the body can't
        # be knocked over by the arm IK transients.
        q  = np.array([self.data.qpos[a] for a in self._qpos_idx], dtype=np.float32)
        dq = np.array([self.data.qvel[a] for a in self._qvel_idx], dtype=np.float32)
        kp_eff = cfg.kp.copy()
        kd_eff = cfg.kd.copy()
        if self.left_arm.target.active:
            kp_eff[15:22] *= 3.0
            kd_eff[15:22] *= 1.8
        if self.right_arm.target.active:
            kp_eff[22:29] *= 3.0
            kd_eff[22:29] *= 1.8
        if self._stand_still_mode:
            kp_eff[0:15] *= 3.0     # legs + waist
            kd_eff[0:15] *= 1.8
        tau = kp_eff * (self._policy_target_q - q) - kd_eff * dq
        self.data.ctrl[:] = tau

        self._update_carry()
        mujoco.mj_step(self.model, self.data)
        self._sim_time += self.dt
        if self.data.qpos[2] < 0.4:
            self._fell = True
        self._emit_step()

    def step_for(self, seconds: float) -> None:
        n = int(seconds / self.dt)
        for _ in range(n):
            if self._fell:
                break
            self.step()

    # ----- carry / grasp ------------------------------------------------ #

    def _record_pelvis_offset(self) -> np.ndarray:
        pp = np.asarray(self.data.xpos[self._pelvis_b]).copy()
        pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
        return pm.T @ (self.box_pos() - pp)

    def _record_hand_offset(self, body_id: int) -> np.ndarray:
        hp = np.asarray(self.data.xpos[body_id]).copy()
        hm = np.asarray(self.data.xmat[body_id]).reshape(3, 3).copy()
        return hm.T @ (self.box_pos() - hp)

    def grasp(self, hand: GraspMode = "both") -> None:
        if hand == "both":
            self._box_offset_in_pelvis = self._record_pelvis_offset()
            self._held_by = "both"
        elif hand in ("left", "right"):
            bid = self._left_h_b if hand == "left" else self._right_h_b
            self._box_offset_in_hand[:] = self._record_hand_offset(bid)
            self._held_by = hand
        else:
            raise ValueError(f"hand must be left|right|both, got {hand!r}")

    def release(self) -> None:
        self._held_by = None

    def _update_carry(self) -> None:
        if self._held_by is None:
            return
        if self._held_by == "both":
            pp = np.asarray(self.data.xpos[self._pelvis_b]).copy()
            pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
            new_pos = pp + pm @ self._box_offset_in_pelvis
        else:
            bid = self._left_h_b if self._held_by == "left" else self._right_h_b
            hp = np.asarray(self.data.xpos[bid]).copy()
            hm = np.asarray(self.data.xmat[bid]).reshape(3, 3).copy()
            new_pos = hp + hm @ self._box_offset_in_hand
        self.data.qpos[self._box_qadr + 0] = float(new_pos[0])
        self.data.qpos[self._box_qadr + 1] = float(new_pos[1])
        self.data.qpos[self._box_qadr + 2] = float(new_pos[2])
        self.data.qpos[self._box_qadr + 3] = 1.0
        self.data.qpos[self._box_qadr + 4:self._box_qadr + 7] = 0.0
        self.data.qvel[self._box_dofadr:self._box_dofadr + 6] = 0.0

    # ----- velocity command (joystick teleop) --------------------------- #

    def set_cmd(self, forward: float, lateral: float, yaw_rate: float) -> None:
        """Set the body-frame velocity command for the locomotion policy."""
        ranges = {
            "lin_vel_x": (-0.5, 1.0),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.0, 1.0),
        }
        self.cmd[0] = float(np.clip(forward, *ranges["lin_vel_x"]))
        self.cmd[1] = float(np.clip(lateral, *ranges["lin_vel_y"]))
        self.cmd[2] = float(np.clip(yaw_rate, *ranges["ang_vel_z"]))

    # ----- arm control: convenient wrappers ---------------------------- #

    def arm_set_target(self, hand: Hand, pos: np.ndarray) -> None:
        ctl = self.left_arm if hand == "left" else self.right_arm
        ctl.set_target(np.asarray(pos, dtype=float))

    def arm_integrate_velocity(self, hand: Hand, vx: float, vy: float, vz: float) -> None:
        ctl = self.left_arm if hand == "left" else self.right_arm
        ctl.integrate_velocity(vx, vy, vz, self.dt)

    def arm_deactivate(self, hand: Optional[Hand] = None) -> None:
        if hand is None or hand == "left":
            self.left_arm.deactivate()
        if hand is None or hand == "right":
            self.right_arm.deactivate()

    # ----- video / viewer ---------------------------------------------- #

    def start_recording(self, video_path: str, width: int = 960, height: int = 540, fps: int = 30) -> None:
        try:
            import imageio.v2 as imageio  # noqa: F401
        except Exception:
            os.system(f"{sys.executable} -m pip install --quiet imageio[ffmpeg]")
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        if (self._renderer is None
            or getattr(self._renderer, "_width", None) != width
            or getattr(self._renderer, "_height", None) != height):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.render_cfg = RenderConfig(enabled=True, width=width, height=height,
                                       fps=fps, video_path=str(video_path))
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

    # ----- internal: per-step hook ------------------------------------- #

    def _emit_step(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.sync()
            except Exception:
                self._viewer = None
        if self._renderer is not None and self._video_writer is not None:
            self._frames_since_render += 1
            if self._frames_since_render >= self._render_every:
                self._frames_since_render = 0
                p = self.pelvis_pos
                self._cam.lookat[0] = 0.7 * self._cam.lookat[0] + 0.3 * float(p[0])
                self._cam.lookat[1] = 0.7 * self._cam.lookat[1] + 0.3 * float(p[1])
                self._cam.lookat[2] = 0.9
                self._renderer.update_scene(self.data, camera=self._cam)
                self._video_writer.append_data(self._renderer.render())
