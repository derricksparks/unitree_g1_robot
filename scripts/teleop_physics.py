#!/usr/bin/env python3
"""
Real-physics teleop for the Unitree G1.

Hybrid stack:
  * LEGS  -- driven by the trained `Unitree-G1-Flat` ONNX velocity policy
              (deploy/.../velocity/v0/exported/policy.onnx). The user's
              joystick / keyboard sets the body-frame velocity command
              (forward, lateral, yaw_rate) at runtime; the policy converts
              that into joint targets at 50 Hz which a 500 Hz PD loop
              tracks. The result is a real bipedal walk with visible
              footsteps under MuJoCo physics.

  * ARMS  -- driven by `ArmCartesianController` (task-priority DLS IK +
              null-space pull toward the policy's default arm pose). End-
              effector velocity in (joystick / keyboard / scripted), joint
              target out, layered onto the policy's per-joint targets in
              the same PD loop. We stiffen the arm gains while a target is
              active so the body's natural sway doesn't pull the hand off
              the goal.

  * EYES  -- onboard `d435i_rgb` camera segmented for the red box; world
              position estimated by ray-casting to the table plane.
              `pickup` / `place` commands are gated on a successful detect
              -- no privileged ground-truth box pose is used.

Sony DualShock 4 / DualSense (auto-detected via SDL/pygame):
  Left stick   X     -> body-frame lateral (strafe) velocity
  Left stick   Y     -> body-frame forward velocity (push up = forward)
  Right stick  X     -> body-frame yaw rate (twist left/right)
  Right stick  Y     -> active arm Z velocity (push up = +Z)
  D-pad              -> active arm XY velocity in world frame
  L2/R2              -> active arm Z velocity (fine)
  Cross  (A)         -> PICKUP (perception-gated, both hands)
  Circle (B)         -> RELEASE
  Square (X)         -> DETECT
  Triangle (Y)       -> cycle active arm (left -> right -> none -> ...)
  L1                 -> walk-speed -
  R1                 -> walk-speed +
  Options/Start      -> quit

Keyboard mirror (focus the small pygame window):
  W S A D Q E   -> forward, back, strafe-left, strafe-right, yaw-left, yaw-right
  1 2 `         -> activate left arm / right arm / no arm
  I K J L U O   -> active-arm end-effector velocity (world frame)
  P             -> DETECT
  G             -> PICKUP (both hands by default)
  R             -> RELEASE
  V             -> toggle mp4 recording
  X             -> reset
  Esc           -> quit

Prompt commands (typed at the running terminal):
  detect | pickup [hand] | release | place [hand] | reset |
  cmdvel f s w t | armvel hand vx vy vz | record on/off | quit

Run:
  python scripts/teleop_physics.py                         # joystick + keyboard + prompt
  python scripts/teleop_physics.py --viewer                # also open native MuJoCo viewer
  python scripts/teleop_physics.py --hud                   # OpenCV onboard-camera HUD
  python scripts/teleop_physics.py --record demo.mp4       # write video of the run
  python scripts/teleop_physics.py --no-input              # for scripted/test runs
  python scripts/teleop_physics.py --no-policy             # PD-to-default (will not walk)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _g1_physics import (
        DEFAULT_SCENE, G1PhysicsWorld, JOINT_NAMES_29,
    )
    from _g1_sim import BoxCamDetector  # reused for perception
else:
    from ._g1_physics import (
        DEFAULT_SCENE, G1PhysicsWorld, JOINT_NAMES_29,
    )
    from ._g1_sim import BoxCamDetector


# --------------------------------------------------------------------------- #
# pygame helper (works headless via SDL_VIDEODRIVER=dummy)
# --------------------------------------------------------------------------- #

def _init_pygame_input(headless_ok: bool = True):
    try:
        import pygame
    except Exception:
        return None, None
    try:
        if headless_ok and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        screen = pygame.display.set_mode((360, 96))
        pygame.display.set_caption("G1 physics teleop (focus this window for keyboard)")
        pygame.joystick.init()
        return pygame, screen
    except Exception:
        return None, None


# Sony PlayStation controller axis / button mappings under SDL.
# These are intentionally for DS4 / DualSense; some platforms re-map slightly.
DS_AXIS_LX, DS_AXIS_LY = 0, 1
DS_AXIS_RX, DS_AXIS_RY = 2, 3
DS_AXIS_L2, DS_AXIS_R2 = 4, 5
DS_BUTTON_CROSS, DS_BUTTON_CIRCLE, DS_BUTTON_SQUARE, DS_BUTTON_TRIANGLE = 0, 1, 2, 3
DS_BUTTON_L1, DS_BUTTON_R1 = 9, 10
DS_BUTTON_OPTIONS = 6


# --------------------------------------------------------------------------- #
# the teleop app
# --------------------------------------------------------------------------- #

class TeleopPhysicsApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        print(f"[teleop_physics] loading: {args.scene}")
        self.world = G1PhysicsWorld(args.scene, load_policy=not args.no_policy)
        print(f"[teleop_physics] dt={self.world.dt:.4f}s  "
              f"policy_dt={self.world.policy_cfg.step_dt:.3f}s  "
              f"decimation={self.world.decimation}  "
              f"policy={'on' if self.world._sess is not None else 'off'}")

        # Perception (renders the onboard d435i_rgb camera).
        self.det = BoxCamDetector(self.world)  # reuses the same MjModel
        self.last_detection = None

        # User-tweakable speeds.
        self.walk_speed = 0.7
        self.yaw_rate   = 0.8
        self.arm_speed  = 0.30

        # Active arm.
        self.active_arm: Optional[str] = None  # 'left' | 'right' | None

        # Optional viewer / HUD.
        if args.viewer:
            try:
                self.world.open_viewer()
                print("[teleop_physics] native viewer opened")
            except Exception as e:
                print(f"[teleop_physics] could not open viewer: {e}")
        self.hud_enabled = args.hud
        self.recording = False
        if args.record:
            self.world.start_recording(args.record)
            self.recording = True
            print(f"[teleop_physics] recording -> {args.record}")

        # Inputs.
        self.pg, self.pg_screen = _init_pygame_input()
        self.joystick = None
        if self.pg is not None and self.pg.joystick.get_count() > 0:
            self.joystick = self.pg.joystick.Joystick(0)
            self.joystick.init()
            print(f"[teleop_physics] joystick: {self.joystick.get_name()} "
                  f"({self.joystick.get_numaxes()} axes, {self.joystick.get_numbuttons()} buttons)")
        else:
            print("[teleop_physics] no joystick connected; keyboard + prompt are still available")

        self._stop = threading.Event()
        self._cmd_q: list[str] = []
        self._cmd_lock = threading.Lock()
        if not args.no_input:
            self._prompt_thread = threading.Thread(target=self._prompt_loop, daemon=True)
            self._prompt_thread.start()
        else:
            self._prompt_thread = None

    # ------------------- prompt thread -------------------------------- #

    def _prompt_loop(self) -> None:
        print("[teleop_physics] prompt commands: detect | pickup [hand] | release | "
              "place [hand] | cmdvel f s w t | armvel hand vx vy vz | "
              "reset | record on/off | status | help | quit")
        while not self._stop.is_set():
            try:
                line = input("(g1) ").strip()
            except (EOFError, KeyboardInterrupt):
                self.enqueue("quit"); return
            except Exception:
                self.enqueue("quit"); return
            if line:
                self.enqueue(line)

    def enqueue(self, cmd: str) -> None:
        with self._cmd_lock:
            self._cmd_q.append(cmd)

    def _drain_commands(self) -> None:
        with self._cmd_lock:
            cmds = self._cmd_q[:]
            self._cmd_q.clear()
        for c in cmds:
            self._handle_text(c)

    def _handle_text(self, line: str) -> None:
        toks = line.split()
        if not toks:
            return
        head = toks[0].lower()
        try:
            if head == "detect":
                self._do_detect()
            elif head in ("pickup", "pick"):
                hand = toks[1] if len(toks) > 1 else "both"
                self._do_pickup(hand)
            elif head == "release":
                self.world.release()
                self.world.left_arm.deactivate()
                self.world.right_arm.deactivate()
                self.active_arm = None
                print("[teleop_physics] released")
            elif head == "place":
                hand = toks[1] if len(toks) > 1 else (self.world.held_by or "both")
                self._do_place(hand)
            elif head == "reset":
                self.world.reset()
                self.last_detection = None
                self.active_arm = None
                print("[teleop_physics] reset")
            elif head == "cmdvel" and len(toks) == 5:
                f, s, w, dur = (float(t) for t in toks[1:])
                self.world.set_cmd(f, s, w)
                self.world.step_for(dur)
                self.world.set_cmd(0, 0, 0)
                print(f"[teleop_physics] cmdvel done; pelvis={self.world.pelvis_pos.round(3).tolist()} "
                      f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}° fell={self.world.fell}")
            elif head == "armvel" and len(toks) == 5:
                hand = toks[1]; vx, vy, vz = (float(t) for t in toks[2:])
                if hand not in ("left", "right"):
                    print("[teleop_physics] armvel needs hand=left|right"); return
                ctl = self.world.left_arm if hand == "left" else self.world.right_arm
                ctl.activate_at_current()
                self.active_arm = hand
                # Apply the velocity for 0.5 s of sim.
                end_t = self.world.sim_time + 0.5
                while self.world.sim_time < end_t and not self.world.fell:
                    ctl.integrate_velocity(vx, vy, vz, self.world.dt)
                    self.world.step()
                print(f"[teleop_physics] armvel done; {hand}_hand={self.world.hand_pos(hand).round(3).tolist()}")
            elif head == "record":
                want = (len(toks) > 1 and toks[1].lower() == "on")
                if len(toks) > 1:
                    self._toggle_record(force=want)
                else:
                    self._toggle_record()
            elif head == "status":
                self._print_status()
            elif head in ("quit", "exit", "q"):
                self._stop.set(); print("[teleop_physics] quitting")
            elif head in ("help", "?"):
                print("Commands: detect | pickup [hand] | release | place [hand] | "
                      "reset | cmdvel f s w t | armvel hand vx vy vz | "
                      "record on|off | status | quit")
            else:
                print(f"[teleop_physics] unknown command: {line!r}")
        except Exception as e:
            print(f"[teleop_physics] error: {e!r}")

    # ------------------- actions ------------------------------------- #

    def _do_detect(self) -> None:
        d = self.det.detect()
        self.last_detection = d
        if d.found:
            print(f"[teleop_physics] DETECT bbox={d.bbox} pixel={d.pixel_xy} "
                  f"world={d.world_pos.round(3).tolist()} conf={d.world_pos_confidence:.2f}")
        else:
            print("[teleop_physics] DETECT no red box visible")

    def _do_pickup(self, hand: str) -> None:
        if hand not in ("left", "right", "both"):
            print("[teleop_physics] pickup needs hand=left|right|both"); return
        if self.last_detection is None or not self.last_detection.found:
            print("[teleop_physics] no prior detection; running detect first")
            self._do_detect()
            if self.last_detection is None or not self.last_detection.found:
                print("[teleop_physics] still no detection; aborting"); return
        # Pin body to a steady stand and let the arm controller drive the
        # arms. The legs are PD'd to default stand pose so the body doesn't
        # drift while the IK runs. We resume policy locomotion right after
        # the grasp so the user can walk the box to the shelf.
        self.world.set_cmd(0, 0, 0)
        self.world._stand_still_mode = True
        try:
            self.world.step_for(0.3)
            base = np.asarray(self.last_detection.world_pos, dtype=float)
            ok = self._pickup_box_under_physics(base, hand)
        finally:
            self.world._stand_still_mode = False
        print(f"[teleop_physics] PICKUP {'ok' if ok else 'failed'}; held_by={self.world.held_by}")
        if not ok:
            self.world.left_arm.deactivate(); self.world.right_arm.deactivate()
            self.active_arm = None

    def _do_place(self, hand: str) -> None:
        if hand not in ("left", "right", "both"):
            print("[teleop_physics] place needs hand=left|right|both"); return
        if self.world.held_by is None:
            print("[teleop_physics] not holding anything"); return
        self.world.set_cmd(0, 0, 0)
        self.world._stand_still_mode = True
        try:
            self.world.step_for(0.3)
            target = self.world.place_target_pos()
            ok = self._place_box_under_physics(target, hand)
        finally:
            self.world._stand_still_mode = False
        # Release the arms back to the policy so the gait can swing them again.
        self.world.left_arm.deactivate(); self.world.right_arm.deactivate()
        self.active_arm = None
        print(f"[teleop_physics] PLACE {'ok' if ok else 'failed'}; box={self.world.box_pos().round(3).tolist()}")

    def _pickup_box_under_physics(self, base: np.ndarray, hand: str) -> bool:
        side = 0.07
        # Choose phase set per hand mode.
        if hand == "both":
            l_app = base + np.array([0,  side, 0.10]); r_app = base + np.array([0, -side, 0.10])
            l_grasp = base + np.array([0,  side, 0.04]); r_grasp = base + np.array([0, -side, 0.04])
            l_lift = base + np.array([0,  side, 0.20]); r_lift = base + np.array([0, -side, 0.20])
            self.world.left_arm.set_target(l_app);  self.world.right_arm.set_target(r_app)
            self.world.step_for(1.5)
            self.world.left_arm.set_target(l_grasp); self.world.right_arm.set_target(r_grasp)
            self.world.step_for(1.5)
            le = float(np.linalg.norm(self.world.hand_pos("left")  - l_grasp))
            re = float(np.linalg.norm(self.world.hand_pos("right") - r_grasp))
            if max(le, re) > 0.18:
                return False
            self.world.grasp("both")
            self.world.left_arm.set_target(l_lift); self.world.right_arm.set_target(r_lift)
            self.world.step_for(1.0)
            return True
        # single hand
        ctl = self.world.left_arm if hand == "left" else self.world.right_arm
        ctl.set_target(base + np.array([0, 0, 0.10]))
        self.world.step_for(1.5)
        ctl.set_target(base + np.array([0, 0, 0.04]))
        self.world.step_for(1.5)
        err = float(np.linalg.norm(self.world.hand_pos(hand) - (base + np.array([0, 0, 0.04]))))
        if err > 0.15:
            return False
        self.world.grasp(hand)  # type: ignore[arg-type]
        ctl.set_target(base + np.array([0, 0, 0.20]))
        self.world.step_for(1.0)
        return True

    def _place_box_under_physics(self, target: np.ndarray, hand: str) -> bool:
        side = 0.07
        if hand == "both":
            l_app = target + np.array([0,  side, 0.16]); r_app = target + np.array([0, -side, 0.16])
            l_pl  = target + np.array([0,  side, 0.06]); r_pl  = target + np.array([0, -side, 0.06])
            self.world.left_arm.set_target(l_app);  self.world.right_arm.set_target(r_app)
            self.world.step_for(1.5)
            self.world.left_arm.set_target(l_pl);   self.world.right_arm.set_target(r_pl)
            self.world.step_for(1.5)
            self.world.release()
            self.world.step_for(1.0)
            return True
        ctl = self.world.left_arm if hand == "left" else self.world.right_arm
        ctl.set_target(target + np.array([0, 0, 0.16]))
        self.world.step_for(1.5)
        ctl.set_target(target + np.array([0, 0, 0.06]))
        self.world.step_for(1.5)
        self.world.release()
        self.world.step_for(1.0)
        return True

    def _print_status(self) -> None:
        w = self.world
        print(f"  sim_time={w.sim_time:6.2f}s  pelvis={w.pelvis_pos.round(3).tolist()} "
              f"yaw={math.degrees(w.pelvis_yaw):+.1f}°  fell={w.fell}  "
              f"active_arm={self.active_arm}  held_by={w.held_by or '—'}  "
              f"cmd={tuple(round(float(x), 2) for x in w.cmd)}")

    def _toggle_record(self, force: Optional[bool] = None) -> None:
        want = (not self.recording) if force is None else bool(force)
        if want and not self.recording:
            self.world.start_recording(self.args.record or
                                       str(Path(__file__).resolve().parents[1] / "scripts" / "logs" / "teleop_physics.mp4"))
            self.recording = True
            print(f"[teleop_physics] recording -> {self.world.render_cfg.video_path}")
        elif (not want) and self.recording:
            p = self.world.stop_recording()
            self.recording = False
            print(f"[teleop_physics] recording stopped (wrote {p})")

    def _set_active_arm(self, hand: Optional[str]) -> None:
        if hand == "left":
            self.world.left_arm.activate_at_current(); self.world.right_arm.deactivate()
            self.active_arm = "left"
        elif hand == "right":
            self.world.right_arm.activate_at_current(); self.world.left_arm.deactivate()
            self.active_arm = "right"
        else:
            self.world.left_arm.deactivate(); self.world.right_arm.deactivate()
            self.active_arm = None
        print(f"[teleop_physics] active arm: {self.active_arm}")

    # ------------------- input -------------------------------------- #

    def _read_inputs(self) -> tuple[float, float, float, float, float, float, list[str]]:
        f = s = w = 0.0
        ax = ay = az = 0.0
        events: list[str] = []
        if self.pg is None:
            return f, s, w, ax, ay, az, events

        # Drain SDL events first.
        for ev in self.pg.event.get():
            if ev.type == self.pg.QUIT:
                events.append("quit")
            elif ev.type == self.pg.KEYDOWN:
                k = ev.key
                if   k == self.pg.K_ESCAPE: events.append("quit")
                elif k == self.pg.K_p:      events.append("detect")
                elif k == self.pg.K_g:      events.append("pickup")
                elif k == self.pg.K_r:      events.append("release")
                elif k == self.pg.K_x:      events.append("reset")
                elif k == self.pg.K_v:      events.append("record_toggle")
                elif k == self.pg.K_1:      events.append("arm:left")
                elif k == self.pg.K_2:      events.append("arm:right")
                elif k == self.pg.K_BACKQUOTE: events.append("arm:none")
                elif k == self.pg.K_SLASH:  events.append("help")
                elif k == self.pg.K_LEFTBRACKET:  events.append("walk_speed:-")
                elif k == self.pg.K_RIGHTBRACKET: events.append("walk_speed:+")
            elif ev.type == self.pg.JOYBUTTONDOWN:
                idx = ev.button
                if   idx == DS_BUTTON_CROSS:    events.append("pickup")
                elif idx == DS_BUTTON_CIRCLE:   events.append("release")
                elif idx == DS_BUTTON_SQUARE:   events.append("detect")
                elif idx == DS_BUTTON_TRIANGLE: events.append("arm_cycle")
                elif idx == DS_BUTTON_L1:       events.append("walk_speed:-")
                elif idx == DS_BUTTON_R1:       events.append("walk_speed:+")
                elif idx == DS_BUTTON_OPTIONS:  events.append("quit")

        # Continuous keyboard.
        keys = self.pg.key.get_pressed()
        if keys[self.pg.K_w]: f += self.walk_speed
        if keys[self.pg.K_s]: f -= self.walk_speed
        if keys[self.pg.K_a]: s += self.walk_speed
        if keys[self.pg.K_d]: s -= self.walk_speed
        if keys[self.pg.K_q]: w += self.yaw_rate
        if keys[self.pg.K_e]: w -= self.yaw_rate
        if self.active_arm is not None:
            if keys[self.pg.K_i]: ax += self.arm_speed
            if keys[self.pg.K_k]: ax -= self.arm_speed
            if keys[self.pg.K_j]: ay += self.arm_speed
            if keys[self.pg.K_l]: ay -= self.arm_speed
            if keys[self.pg.K_u]: az += self.arm_speed
            if keys[self.pg.K_o]: az -= self.arm_speed

        # Joystick.
        if self.joystick is not None:
            DEAD = 0.15
            def ax_f(i: int) -> float:
                try:
                    v = float(self.joystick.get_axis(i))
                except Exception:
                    return 0.0
                return v if abs(v) > DEAD else 0.0
            # Sticks. Forward = stick UP = -axis_LY.
            f += -ax_f(DS_AXIS_LY) * self.walk_speed
            s += -ax_f(DS_AXIS_LX) * self.walk_speed
            w += -ax_f(DS_AXIS_RX) * self.yaw_rate
            if self.active_arm is not None:
                az += -ax_f(DS_AXIS_RY) * self.arm_speed
                # Triggers L2/R2 are -1 .. +1. Map to (l_t, r_t) in [0,1].
                try:
                    l_t = (ax_f(DS_AXIS_L2) + 1.0) * 0.5
                    r_t = (ax_f(DS_AXIS_R2) + 1.0) * 0.5
                    az += (r_t - l_t) * self.arm_speed
                except Exception:
                    pass
                try:
                    hat = self.joystick.get_hat(0)
                    ax += hat[1] * self.arm_speed
                    ay += -hat[0] * self.arm_speed
                except Exception:
                    pass
        return f, s, w, ax, ay, az, events

    # ------------------- HUD ------------------------------------------ #

    def _render_hud(self) -> None:
        if not self.hud_enabled:
            return
        try:
            import cv2
        except ImportError:
            return
        rgb = self.det.last_image
        if rgb is None:
            rgb = self.det.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        d = self.last_detection
        if d is not None and d.found:
            x, y, w0, h0 = d.bbox
            sx = bgr.shape[1] / max(1, self.det.width)
            sy = bgr.shape[0] / max(1, self.det.height)
            cv2.rectangle(bgr, (int(x*sx), int(y*sy)),
                          (int((x+w0)*sx), int((y+h0)*sy)), (0, 255, 0), 2)
            cv2.putText(bgr, f"world=({d.world_pos[0]:.2f},{d.world_pos[1]:.2f},{d.world_pos[2]:.2f})",
                        (int(x*sx), max(0, int(y*sy) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        info = (f"cmd=({self.world.cmd[0]:+.2f},{self.world.cmd[1]:+.2f},{self.world.cmd[2]:+.2f})  "
                f"arm={self.active_arm}  held={self.world.held_by or '—'}  "
                f"rec={'on' if self.recording else 'off'}  "
                f"fell={self.world.fell}")
        cv2.putText(bgr, info, (8, bgr.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("G1 onboard camera", bgr)
        cv2.waitKey(1)

    # ------------------- main loop ------------------------------------ #

    def run(self, max_seconds: Optional[float] = None) -> int:
        target_dt_wall = 0.02
        last_wall = time.perf_counter()
        last_hud = 0.0
        t0 = last_wall
        while not self._stop.is_set():
            now = time.perf_counter()
            elapsed_wall = now - last_wall
            last_wall = now

            self._drain_commands()

            f, s, w, ax, ay, az, events = self._read_inputs()
            for evt in events:
                if   evt == "quit":          self._stop.set()
                elif evt == "detect":        self._do_detect()
                elif evt == "pickup":        self._do_pickup("both")
                elif evt == "release":       self._handle_text("release")
                elif evt == "reset":         self._handle_text("reset")
                elif evt == "record_toggle": self._toggle_record()
                elif evt == "arm:left":      self._set_active_arm("left")
                elif evt == "arm:right":     self._set_active_arm("right")
                elif evt == "arm:none":      self._set_active_arm(None)
                elif evt == "arm_cycle":
                    nxt = {"left": "right", "right": None, None: "left"}[self.active_arm]
                    self._set_active_arm(nxt)
                elif evt == "walk_speed:+":  self.walk_speed = min(1.0, self.walk_speed + 0.1)
                elif evt == "walk_speed:-":  self.walk_speed = max(0.1, self.walk_speed - 0.1)
                elif evt == "help":          print(__doc__.split("Run:")[0])

            self.world.set_cmd(f, s, w)

            # Apply arm velocity on the active arm (one tick).
            if self.active_arm == "left" and (ax or ay or az):
                self.world.arm_integrate_velocity("left", ax, ay, az)
            elif self.active_arm == "right" and (ax or ay or az):
                self.world.arm_integrate_velocity("right", ax, ay, az)

            # Step physics for ~elapsed_wall seconds (tracking wall-clock).
            n_substeps = max(1, int(round(elapsed_wall / self.world.dt)))
            n_substeps = min(n_substeps, 25)
            for _ in range(n_substeps):
                if self.world.fell or self._stop.is_set():
                    break
                self.world.step()

            if self.world.fell:
                print("[teleop_physics] robot fell; resetting in 1.0 s")
                time.sleep(1.0)
                self.world.reset()
                self._set_active_arm(None)

            if self.hud_enabled and (now - last_hud) > 0.1:
                self._render_hud()
                last_hud = now

            if self.pg is not None and self.pg_screen is not None:
                self.pg_screen.fill((20, 20, 30))
                try:
                    self.pg.display.flip()
                except Exception:
                    pass

            if max_seconds is not None and (now - t0) > max_seconds:
                self._stop.set()

            time.sleep(max(0.0, target_dt_wall - (time.perf_counter() - now)))

        # cleanup
        try:
            time.sleep(0.05)
        except Exception:
            pass
        try:
            if self.recording:
                self.world.stop_recording()
        except Exception:
            pass
        try:
            if self.pg is not None:
                self.pg.quit()
        except Exception:
            pass
        try:
            if self.hud_enabled:
                import cv2
                cv2.destroyAllWindows()
        except Exception:
            pass
        self.world.close()
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--no-policy", action="store_true",
                    help="Disable the trained ONNX policy (will not actually walk)")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer")
    ap.add_argument("--hud",    action="store_true", help="Open OpenCV onboard-camera HUD")
    ap.add_argument("--record", default=None, help="Record everything to this mp4")
    ap.add_argument("--no-input", action="store_true",
                    help="Don't start the prompt thread (for scripted/headless runs)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Auto-quit after this many wall seconds (for tests)")
    args = ap.parse_args()
    return TeleopPhysicsApp(args).run(max_seconds=args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
