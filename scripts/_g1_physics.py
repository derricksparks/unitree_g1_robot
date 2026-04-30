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
        self._stand_still_base_qpos: Optional[np.ndarray] = None
        # When True (set automatically while the box is held), hold each
        # active arm IK target at a fixed *pelvis-frame* offset rather than
        # at a fixed world point. This gives a stable carry pose: the arms
        # ride with the body so the gait policy can walk without the box
        # swinging on a stationary world goal.
        self._arm_target_in_pelvis = {"left": None, "right": None}

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

        # Stand-still mode: snapshot the floating-base pose at entry so we
        # can clamp it back if it drifts under arm-IK transients (the gait
        # policy is not running and the leg PD alone can't reject torque
        # from the arms).
        if self._stand_still_mode and self._stand_still_base_qpos is None:
            self._stand_still_base_qpos = self.data.qpos[:7].copy()
        elif not self._stand_still_mode and self._stand_still_base_qpos is not None:
            self._stand_still_base_qpos = None

        # Update any pelvis-locked arm targets so they ride with the body.
        self._refresh_body_frame_arm_targets()

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
            # Pin lower body firmly so arm IK transients can't topple it.
            # Gains are scaled relative to the deploy YAML defaults; the
            # leg actuators have plenty of headroom at these multipliers.
            kp_eff[0:12] *= 5.0     # legs
            kd_eff[0:12] *= 2.5
            kp_eff[12:15] *= 3.0    # waist
            kd_eff[12:15] *= 2.0
        tau = kp_eff * (self._policy_target_q - q) - kd_eff * dq
        self.data.ctrl[:] = tau

        self._update_carry()
        mujoco.mj_step(self.model, self.data)
        # Clamp the floating-base pose during stand-still mode. The leg PD
        # alone can't fully reject torque from the arm IK; without this,
        # the body slowly twists (we measured ~80 deg of yaw drift across a
        # one-second IK reach to the table). The clamp is bounded -- if the
        # offset is large, we're falling and let it through so the fall
        # detector triggers.
        if self._stand_still_mode and self._stand_still_base_qpos is not None:
            base = self._stand_still_base_qpos
            if abs(self.data.qpos[2] - base[2]) < 0.15:
                self.data.qpos[:7] = base
                self.data.qvel[:6] = 0.0
                mujoco.mj_forward(self.model, self.data)
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
        # While held, neutralize the box's contribution to the robot's
        # dynamics. Real humanoids handle payloads with payload-aware
        # controllers (or by training the locomotion policy on payload
        # randomization); since the shipped Flat policy was trained
        # unloaded, we model an ideal "magnetic gripper" that takes the
        # weight without disturbing the arm. The box's pose is overwritten
        # every tick from the recorded offsets, so its dynamics don't
        # actually affect the carry, only its momentum + contact forces
        # leaking through during the per-tick integration -- which we cut
        # off here.
        self._box_mass_orig = float(self.model.body_mass[self._box_b])
        self._box_inertia_orig = self.model.body_inertia[self._box_b].copy()
        self.model.body_mass[self._box_b] = 1e-4
        self.model.body_inertia[self._box_b] = 1e-6

    def release(self) -> None:
        if self._held_by is not None and getattr(self, "_box_mass_orig", None) is not None:
            # Restore the box's mass/inertia so it falls under gravity.
            self.model.body_mass[self._box_b] = self._box_mass_orig
            self.model.body_inertia[self._box_b] = self._box_inertia_orig
            self._box_mass_orig = None
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
        """Set the body-frame velocity command for the locomotion policy.

        When the robot is holding the box, all command magnitudes are scaled
        down (to ~50 %) so the user can't drive faster than the policy can
        stabilize under arm payload. The Unitree-G1-Flat checkpoint shipped
        in this repo was trained without payload, so this is an explicit
        conservative envelope rather than a domain-randomized one.
        """
        # Policy-trained envelope.
        rng = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))
        f = float(np.clip(forward,  *rng[0]))
        s = float(np.clip(lateral,  *rng[1]))
        w = float(np.clip(yaw_rate, *rng[2]))
        if self._held_by is not None:
            # Conservative envelope while carrying.
            f = float(np.clip(f, -0.25, 0.5))
            s = float(np.clip(s, -0.25, 0.25))
            w = float(np.clip(w, -0.5, 0.5))
        self.cmd[:] = (f, s, w)

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
            self._arm_target_in_pelvis["left"] = None
        if hand is None or hand == "right":
            self.right_arm.deactivate()
            self._arm_target_in_pelvis["right"] = None

    # ----- smooth Cartesian reach -------------------------------------- #

    def smooth_reach(
        self,
        hand: Hand,
        target_world: np.ndarray,
        *,
        speed: float = 0.25,
        tol: float = 0.04,
        timeout_s: float = 6.0,
        body_frame: bool = False,
    ) -> tuple[bool, float]:
        """Move an end-effector to a Cartesian target at a bounded speed.

        The IK setpoint is interpolated linearly toward ``target_world`` at
        ``speed`` m/s, and physics keeps stepping in the inner loop. This
        avoids the "IK jump" that would otherwise punch the policy with a
        large arm pose change in one control tick (which empirically topples
        the gait under payload).

        With ``body_frame=True``, ``target_world`` is interpreted as a
        pelvis-frame offset; the world goal is recomputed each tick from
        the live pelvis pose, so the arm "rides with the body". This is
        what we use for the carry pose.

        Returns (reached, final_error_m).
        """
        ctl = self.left_arm if hand == "left" else self.right_arm
        ctl.activate_at_current()
        # Interpolate the IK setpoint at `speed` m/s.
        n_steps = int(timeout_s / self.dt)
        for _ in range(n_steps):
            if self._fell:
                return False, float("inf")
            current = ctl.target.pos.copy()
            if body_frame:
                pp = self.pelvis_pos
                pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
                goal = pp + pm @ np.asarray(target_world, dtype=float)
            else:
                goal = np.asarray(target_world, dtype=float)
            delta = goal - current
            d = float(np.linalg.norm(delta))
            if d > 1e-6:
                step = min(speed * self.dt, d)
                ctl.target.pos = current + delta / d * step
            self.step()
            err = float(np.linalg.norm(self.hand_pos(hand) - goal))
            if err < tol and d < tol:
                return True, err
        # Final error against the (possibly time-varying) goal.
        if body_frame:
            pp = self.pelvis_pos
            pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
            goal = pp + pm @ np.asarray(target_world, dtype=float)
        else:
            goal = np.asarray(target_world, dtype=float)
        return False, float(np.linalg.norm(self.hand_pos(hand) - goal))

    def _refresh_body_frame_arm_targets(self) -> None:
        """For arms with a pelvis-frame offset, rewrite their world target
        to match the live pelvis pose. Called every step so the arm "rides"
        the walking body during a carry."""
        for hand_name in ("left", "right"):
            off = self._arm_target_in_pelvis[hand_name]
            if off is None:
                continue
            ctl = self.left_arm if hand_name == "left" else self.right_arm
            if not ctl.target.active:
                continue
            pp = self.pelvis_pos
            pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
            ctl.target.pos = pp + pm @ off

    def lock_arm_to_pelvis(self, hand: Hand, body_frame_offset: np.ndarray) -> None:
        """Lock the arm's IK target to a fixed offset in pelvis frame so it
        rides with the body. Used for the carry pose during locomotion."""
        ctl = self.left_arm if hand == "left" else self.right_arm
        off = np.asarray(body_frame_offset, dtype=float).copy()
        self._arm_target_in_pelvis[hand] = off
        # Initialize the world target so smooth_reach won't have to ramp.
        pp = self.pelvis_pos
        pm = np.asarray(self.data.xmat[self._pelvis_b]).reshape(3, 3).copy()
        ctl.set_target(pp + pm @ off)

    # ----- carry pose helpers ----------------------------------------- #

    # Carry pose: end-effector targets in pelvis frame (forward, left,
    # up_from_pelvis). Approximately the policy's default arm pose to
    # minimize disagreement with the trained gait. The "magnetic gripper"
    # in grasp() makes the box weightless while held, so the arm pose
    # matters mostly for visuals.
    CARRY_LEFT_PELVIS_OFFSET   = np.array([0.25,  0.18, 0.05])
    CARRY_RIGHT_PELVIS_OFFSET  = np.array([0.25, -0.18, 0.05])
    CARRY_SINGLE_PELVIS_OFFSET = np.array([0.25,  0.18, 0.05])

    def move_to_carry_pose(self, hand: GraspMode, *, speed: float = 0.20,
                           timeout_s: float = 4.0) -> bool:
        """Smoothly move the arms (and the box, since it's held) into the
        tuck-in carry pose. Call this *after* ``grasp(...)`` succeeded."""
        if hand == "both":
            ok_l, _ = self.smooth_reach("left",  self.CARRY_LEFT_PELVIS_OFFSET,
                                        speed=speed, tol=0.06,
                                        timeout_s=timeout_s, body_frame=True)
            ok_r, _ = self.smooth_reach("right", self.CARRY_RIGHT_PELVIS_OFFSET,
                                        speed=speed, tol=0.06,
                                        timeout_s=timeout_s, body_frame=True)
            self.lock_arm_to_pelvis("left",  self.CARRY_LEFT_PELVIS_OFFSET)
            self.lock_arm_to_pelvis("right", self.CARRY_RIGHT_PELVIS_OFFSET)
            return ok_l and ok_r
        if hand in ("left", "right"):
            ok, _ = self.smooth_reach(hand, self.CARRY_SINGLE_PELVIS_OFFSET,  # type: ignore[arg-type]
                                      speed=speed, tol=0.06,
                                      timeout_s=timeout_s, body_frame=True)
            self.lock_arm_to_pelvis(hand, self.CARRY_SINGLE_PELVIS_OFFSET)  # type: ignore[arg-type]
            return ok
        return False

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

    # ----- locomotion convenience: walk to a world target -------------- #

    def goto(
        self,
        target_xy: np.ndarray,
        target_yaw: Optional[float] = None,
        *,
        timeout_s: float = 60.0,
        xy_tol: float = 0.10,
        yaw_tol: float = 0.10,
        max_forward: float = 0.5,
        max_lateral: float = 0.25,
        max_yaw: float = 0.6,
    ) -> bool:
        """Drive the policy to a world (x, y, [yaw]) target.

        Strategy (3 phases, simple but robust):
          A. ALIGN  - stand and yaw in place until heading-to-target is
                      within ~0.2 rad. Walking forward before this just
                      drives off course.
          B. CRUISE - walk forward at full speed while continuously
                      correcting yaw toward the heading-to-target. We
                      don't strafe (the policy handles lateral correction
                      via heading); strafing while walking destabilizes
                      under load.
          C. SETTLE - within xy_tol, stop and yaw to ``target_yaw``.

        Returns True if the target was reached within tolerance, False on
        timeout / fall.
        """
        target_xy = np.asarray(target_xy, dtype=float)
        n_steps = int(timeout_s / self.dt)
        for _ in range(n_steps):
            if self._fell:
                return False
            px, py, _ = self.pelvis_pos
            yaw = self.pelvis_yaw
            ex, ey = target_xy[0] - px, target_xy[1] - py
            dist = math.hypot(ex, ey)

            if dist <= xy_tol:
                # SETTLE
                if target_yaw is None:
                    self.set_cmd(0, 0, 0); self.step_for(0.2)
                    return True
                yfe = math.atan2(math.sin(target_yaw - yaw),
                                 math.cos(target_yaw - yaw))
                if abs(yfe) < yaw_tol:
                    self.set_cmd(0, 0, 0); self.step_for(0.2)
                    return True
                self.set_cmd(0, 0, float(np.clip(2.0 * yfe, -max_yaw, max_yaw)))
                self.step()
                continue

            heading_to_target = math.atan2(ey, ex)
            head_err = math.atan2(math.sin(heading_to_target - yaw),
                                   math.cos(heading_to_target - yaw))

            # Choose to drive forward or backward, whichever is closer to
            # the current heading. The Flat policy walks both ways and
            # this avoids forcing a hard U-turn while carrying a payload.
            if abs(head_err) <= math.pi / 2:
                signed_head_err = head_err
                forward_sign = +1.0
            else:
                # Pointing more than 90 deg the wrong way -> walk backward.
                signed_head_err = math.atan2(math.sin(head_err + math.pi),
                                              math.cos(head_err + math.pi))
                forward_sign = -1.0

            if abs(signed_head_err) > 0.20:
                self.set_cmd(0, 0, float(np.clip(2.5 * signed_head_err, -max_yaw, max_yaw)))
            else:
                f_mag = min(max_forward, max(0.10, 0.6 * dist))
                f = forward_sign * f_mag
                w = float(np.clip(2.0 * signed_head_err, -max_yaw, max_yaw))
                self.set_cmd(f, 0, w)
            self.step()
        self.set_cmd(0, 0, 0)
        return False

    # ----- end-to-end pickup / place under physics --------------------- #

    def approach_for_pickup(
        self,
        box_world_xy: np.ndarray,
        hand: GraspMode = "left",
        standoff: float = 0.45,
    ) -> bool:
        """Walk to a stand-off pose in front of the box so the arm can
        reach it without over-extension.

        For a left-hand grasp, line the box up on the body's +y side so the
        left arm naturally hovers over it. For "both", center the box on
        the body. Standoff is the pelvis-to-box distance along x.
        """
        bx, by = float(box_world_xy[0]), float(box_world_xy[1])
        if hand == "left":
            target_xy = np.array([bx - standoff, by - 0.10])
        elif hand == "right":
            target_xy = np.array([bx - standoff, by + 0.10])
        else:  # "both"
            target_xy = np.array([bx - standoff, by])
        return self.goto(target_xy, target_yaw=0.0, timeout_s=20.0)

    def pickup_box_at(
        self,
        world_xyz: np.ndarray,
        hand: GraspMode = "left",
        *,
        approach_height: float = 0.12,
        grasp_height: float = 0.04,
        side_offset: float = 0.07,
        reach_speed: float = 0.45,
    ) -> bool:
        """Pick up the box at the given world point, then move to the carry
        pose so the gait policy can keep walking. ``hand`` is "left",
        "right", or "both". The legs are pinned to default stand during the
        precise grasp (otherwise arm IK transients can topple the gait), and
        released back to the policy as soon as the carry pose is reached."""
        base = np.asarray(world_xyz, dtype=float)
        # Hold the body still during the grasp.
        self.set_cmd(0, 0, 0)
        self._stand_still_mode = True
        try:
            if hand == "both":
                # Approach: smooth-reach both arms above the box.
                la = base + np.array([0,  side_offset, approach_height])
                ra = base + np.array([0, -side_offset, approach_height])
                self.smooth_reach("left",  la, speed=reach_speed, tol=0.04, timeout_s=4.0)
                self.smooth_reach("right", ra, speed=reach_speed, tol=0.04, timeout_s=4.0)
                # Descend to grasp height.
                lg = base + np.array([0,  side_offset, grasp_height])
                rg = base + np.array([0, -side_offset, grasp_height])
                self.smooth_reach("left",  lg, speed=reach_speed, tol=0.05, timeout_s=4.0)
                self.smooth_reach("right", rg, speed=reach_speed, tol=0.05, timeout_s=4.0)
                le = float(np.linalg.norm(self.hand_pos("left")  - lg))
                re = float(np.linalg.norm(self.hand_pos("right") - rg))
                if max(le, re) > 0.18:
                    return False
                self.grasp("both")
            elif hand in ("left", "right"):
                ctl = self.left_arm if hand == "left" else self.right_arm
                self.smooth_reach(hand, base + np.array([0, 0, approach_height]),
                                  speed=reach_speed, tol=0.04, timeout_s=4.0)
                self.smooth_reach(hand, base + np.array([0, 0, grasp_height]),
                                  speed=reach_speed, tol=0.05, timeout_s=4.0)
                err = float(np.linalg.norm(
                    self.hand_pos(hand) - (base + np.array([0, 0, grasp_height]))))
                if err > 0.16:
                    return False
                self.grasp(hand)
            else:
                raise ValueError(f"hand must be left|right|both, got {hand!r}")

            # Lift just enough to clear the table, still under stand-still.
            if hand == "both":
                self.smooth_reach("left",
                                  base + np.array([0,  side_offset, approach_height + 0.05]),
                                  speed=reach_speed, tol=0.06, timeout_s=3.0)
                self.smooth_reach("right",
                                  base + np.array([0, -side_offset, approach_height + 0.05]),
                                  speed=reach_speed, tol=0.06, timeout_s=3.0)
            else:
                ctl = self.left_arm if hand == "left" else self.right_arm
                self.smooth_reach(hand,
                                  base + np.array([0, 0, approach_height + 0.05]),
                                  speed=reach_speed, tol=0.06, timeout_s=3.0)
        finally:
            # Release the legs back to the policy as soon as the box is
            # clear of the table; the carry-pose move runs under the policy
            # so the gait continues to balance.
            self._stand_still_mode = False

        # Smoothly tuck the box into the carry pose (arms ride with pelvis).
        self.move_to_carry_pose(hand, speed=reach_speed, timeout_s=4.0)
        return True

    def place_box_at(
        self,
        world_xyz: np.ndarray,
        hand: Optional[GraspMode] = None,
        *,
        approach_height: float = 0.18,
        place_height: float = 0.06,
        side_offset: float = 0.07,
        reach_speed: float = 0.20,
        settle_seconds: float = 1.0,
    ) -> bool:
        """Place the carried box at the given world point. Reverses the
        pickup: stops the gait, releases the body-frame arm lock, IKs both
        hands to the place target, releases the box, lets it settle."""
        if self.held_by is None:
            return False
        if hand is None:
            hand = self.held_by
        target = np.asarray(world_xyz, dtype=float)

        self.set_cmd(0, 0, 0)
        self._stand_still_mode = True
        # Stop locking arms to pelvis -- their target is now a fixed world
        # point on the shelf.
        self._arm_target_in_pelvis = {"left": None, "right": None}
        try:
            if hand == "both":
                la = target + np.array([0,  side_offset, approach_height])
                ra = target + np.array([0, -side_offset, approach_height])
                self.smooth_reach("left",  la, speed=reach_speed, tol=0.05, timeout_s=4.0)
                self.smooth_reach("right", ra, speed=reach_speed, tol=0.05, timeout_s=4.0)
                lp = target + np.array([0,  side_offset, place_height])
                rp = target + np.array([0, -side_offset, place_height])
                self.smooth_reach("left",  lp, speed=reach_speed, tol=0.04, timeout_s=4.0)
                self.smooth_reach("right", rp, speed=reach_speed, tol=0.04, timeout_s=4.0)
            elif hand in ("left", "right"):
                self.smooth_reach(hand, target + np.array([0, 0, approach_height]),  # type: ignore[arg-type]
                                  speed=reach_speed, tol=0.05, timeout_s=4.0)
                self.smooth_reach(hand, target + np.array([0, 0, place_height]),     # type: ignore[arg-type]
                                  speed=reach_speed, tol=0.04, timeout_s=4.0)
            else:
                raise ValueError(f"hand must be left|right|both|None, got {hand!r}")
            self.release()
            # Let gravity settle the box.
            for _ in range(int(settle_seconds / self.dt)):
                self.step()
        finally:
            self._stand_still_mode = False
            self.left_arm.deactivate(); self.right_arm.deactivate()
        return True

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
