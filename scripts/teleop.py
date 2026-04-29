#!/usr/bin/env python3
"""
Hybrid teleop for the Unitree G1 + warehouse box.

Combines:
  * A locomotion controller (KinematicLocomotion or, if a checkpoint is
    available, the trained RL policy) driven by joystick / keyboard
    velocity commands.
  * Two `ArmCartesianController`s (one per hand) for end-effector velocity
    teleop -- the role of an arm WBC/MPC, but implemented as task-priority
    DLS IK with a torso-upright posture regularizer.
  * A `BoxCamDetector` reading the onboard `d435i_rgb` camera. Type
    `detect` / `pickup` / `release` at the prompt to drive perception and
    grasping. The grasp is GATED on a successful detection -- we never use
    privileged ground-truth box poses.

Controls (keyboard, default):
    Locomotion (body-frame velocity):
      W / S            forward / backward
      A / D            strafe left / right
      Q / E            yaw left / right
    Active arm:
      1                left hand
      2                right hand
      ` (backtick)     deactivate arm controller
    Selected arm end-effector velocity (world frame):
      I / K            +x / -x
      J / L            +y / -y
      U / O            +z / -z
    Speeds:
      [ / ]            decrease / increase walk speed
      - / =            decrease / increase arm speed
    Box / state:
      P                run a perception step and print result
      G                grasp with the active arm (gated on successful detect)
      R                release whatever is held
      F                quick "face the box" rotate (joystick assist)
      H                hold (zero command)
      X                reset scene
    Recording:
      V                toggle mp4 recording (saved to scripts/logs/teleop.mp4)
    Help:
      ?                print these bindings again
      Esc              quit

Joystick (auto-detected if any are connected):
    Left stick   X     -> strafe (lateral)
    Left stick   Y     -> forward
    Right stick  X     -> yaw rate
    Right stick  Y     -> arm Z velocity for the active arm
    Triggers           -> arm Z velocity (LT down, RT up)
    D-pad              -> arm XY velocity for the active arm
    Buttons:
      A                grasp
      B                release
      X                run perception step
      Y                cycle active arm
      Start            quit

Prompt commands (at any time, in the running terminal):
  detect | pickup | release | reset | quit | record on/off

Run:
  python scripts/teleop.py                    # keyboard, no display
  python scripts/teleop.py --viewer           # also open native viewer
  python scripts/teleop.py --hud              # OpenCV camera + HUD window
  python scripts/teleop.py --rl-checkpoint <path-to-policy.pt>
  python scripts/teleop.py --no-input         # for scripted/test runs
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _g1_sim import (
        DEFAULT_SCENE, G1World, KinematicLocomotionController,
        TrainedRLLocomotionController, ArmCartesianController, BoxCamDetector,
        pickup_detected_box, place_detected_box,
    )
else:
    from ._g1_sim import (
        DEFAULT_SCENE, G1World, KinematicLocomotionController,
        TrainedRLLocomotionController, ArmCartesianController, BoxCamDetector,
        pickup_detected_box, place_detected_box,
    )


# --------------------- pygame input (lazy import) ----------------------------

def _init_pygame_input(headless_ok: bool = True):
    """Return a tuple (pygame, screen) or (None, None) if unavailable."""
    try:
        import pygame
    except Exception:
        return None, None
    try:
        if headless_ok and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        # Tiny window: required by SDL even in dummy mode to capture keys.
        screen = pygame.display.set_mode((320, 96))
        pygame.display.set_caption("G1 teleop (focus this window for keyboard)")
        pygame.joystick.init()
        return pygame, screen
    except Exception:
        return None, None


# --------------------- main teleop class ------------------------------------

class TeleopApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        print(f"[teleop] loading: {args.scene}")
        self.world = G1World(args.scene)
        print(f"[teleop] dt={self.world.dt:.4f}s, nq={self.world.model.nq}, nv={self.world.model.nv}")

        # Locomotion: try the trained policy, fall back to kinematic.
        if args.rl_checkpoint:
            print(f"[teleop] attempting trained-RL locomotion from {args.rl_checkpoint}")
            self.loco = TrainedRLLocomotionController(self.world, args.rl_checkpoint)
        else:
            self.loco = KinematicLocomotionController(self.world)
        print(f"[teleop] locomotion: {self.loco.name()}")

        # Arms.
        self.left_arm = ArmCartesianController(self.world, "left")
        self.right_arm = ArmCartesianController(self.world, "right")
        self.active_arm: Optional[str] = None  # 'left' | 'right' | None

        # Perception.
        self.det = BoxCamDetector(self.world)
        self.last_detection = None

        # Speeds (modifiable via UI).
        self.walk_speed = 0.9   # m/s
        self.yaw_rate   = 1.5   # rad/s
        self.arm_speed  = 0.40  # m/s (Cartesian end-effector vel cap)

        # Optional viewer / HUD.
        if args.viewer:
            try:
                self.world.open_viewer()
                print("[teleop] native viewer open")
            except Exception as e:
                print(f"[teleop] could not open viewer: {e}")
        self.hud_enabled = args.hud
        self._hud_inited = False

        # Optional recording.
        self.recording = False

        # Pygame input.
        self.pg, self.pg_screen = _init_pygame_input()
        self.joystick = None
        if self.pg is not None and self.pg.joystick.get_count() > 0:
            self.joystick = self.pg.joystick.Joystick(0)
            self.joystick.init()
            print(f"[teleop] joystick: {self.joystick.get_name()}")
        if self.pg is None:
            print("[teleop] pygame not available; keyboard/joystick disabled. "
                  "Use the prompt: detect | pickup | release | quit")

        # Prompt thread (so the terminal can drive commands while the sim ticks).
        self._stop = threading.Event()
        self._cmd_q: list[str] = []
        self._cmd_lock = threading.Lock()
        if not args.no_input:
            self._prompt_thread = threading.Thread(target=self._prompt_loop, daemon=True)
            self._prompt_thread.start()
        else:
            self._prompt_thread = None

    # ------------------- prompt -------------------------------------------- #

    def _prompt_loop(self) -> None:
        print("[teleop] prompt commands: detect | pickup | release | reset | "
              "record on/off | goto | cmdvel | place | quit (`help` for full list)")
        while not self._stop.is_set():
            try:
                line = input("(g1-teleop) ").strip()
            except (EOFError, KeyboardInterrupt):
                self.enqueue("quit"); return
            except Exception:
                # Stdin can become invalid during interpreter shutdown.
                self.enqueue("quit"); return
            if not line:
                continue
            self.enqueue(line)

    def enqueue(self, cmd: str) -> None:
        with self._cmd_lock:
            self._cmd_q.append(cmd)

    def _drain_commands(self) -> None:
        with self._cmd_lock:
            cmds = self._cmd_q[:]
            self._cmd_q.clear()
        for c in cmds:
            self._handle_text_command(c)

    def _handle_text_command(self, line: str) -> None:
        toks = line.split()
        if not toks:
            return
        head = toks[0].lower()
        if head == "detect":
            self._do_detect()
        elif head in ("pickup", "pick"):
            hand = toks[1] if len(toks) > 1 else (self.active_arm or "both")
            self._do_pickup(hand)
        elif head == "release":
            self._do_release()
        elif head == "place":
            hand = toks[1] if len(toks) > 1 else (self.world.held_by or self.active_arm or "both")
            self._do_place(hand)
        elif head == "reset":
            self.world.reset()
            self.left_arm.deactivate(); self.right_arm.deactivate()
            self.active_arm = None
            self.last_detection = None
            print("[teleop] reset")
        elif head == "record":
            on = (len(toks) > 1 and toks[1].lower() == "on")
            self._toggle_record(force=on if (len(toks) > 1) else None)
        elif head in ("quit", "exit", "q"):
            self._stop.set()
            print("[teleop] quitting")
        elif head == "status":
            self._print_status()
        elif head == "cmdvel" and len(toks) == 5:
            # cmdvel <forward> <lateral> <yaw_rate> <duration_sec>
            try:
                f, s, w_yr, dur = (float(toks[i]) for i in range(1, 5))
            except ValueError:
                print("[teleop] usage: cmdvel <forward> <lateral> <yaw> <seconds>"); return
            n = int(dur / self.world.dt)
            for _ in range(n):
                self.loco.step(f, s, w_yr)
            print(f"[teleop] cmdvel done; pelvis={self.world.pelvis_pos.round(3).tolist()} "
                  f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}°")
        elif head == "goto" and len(toks) == 4:
            # goto <x> <y> <yaw_rad>
            try:
                tx, ty, tyaw = float(toks[1]), float(toks[2]), float(toks[3])
            except ValueError:
                print("[teleop] usage: goto <x> <y> <yaw_rad>"); return
            self._goto_xy_yaw(tx, ty, tyaw)
        elif head == "armvel" and len(toks) == 5:
            # armvel <left|right> <vx> <vy> <vz>  (one tick of vel; useful for tests)
            hand = toks[1]
            if hand not in ("left", "right"):
                print("[teleop] armvel needs hand=left|right"); return
            try:
                v = (float(toks[2]), float(toks[3]), float(toks[4]))
            except ValueError:
                print("[teleop] usage: armvel <hand> <vx> <vy> <vz>"); return
            self._set_active_arm(hand)
            ctl = self.left_arm if hand == "left" else self.right_arm
            for _ in range(int(0.5 / self.world.dt)):
                ctl.integrate_velocity(*v); ctl.step(); self.world.kine_tick()
            print(f"[teleop] armvel done; {hand}_hand={self.world.hand_pos(hand).round(3).tolist()}")
        elif head == "arm" and len(toks) == 2:
            # arm left | arm right | arm none
            sub = toks[1].lower()
            self._set_active_arm(sub if sub in ("left", "right") else None)
        elif head == "help" or head == "?":
            print("Prompt commands: detect | pickup [hand] | release | place [hand] | "
                  "reset | record on|off | status | cmdvel f s w t | goto x y yaw | "
                  "arm left|right|none | armvel hand vx vy vz | quit")
        else:
            print(f"[teleop] unknown command: {line!r}")

    def _goto_xy_yaw(self, tx: float, ty: float, tyaw: float, max_seconds: float = 30.0) -> None:
        """Drive the robot to (tx, ty, tyaw) using the locomotion controller."""
        max_steps = int(max_seconds / self.world.dt)
        Kxy, Kyaw = 1.5, 1.5
        for _ in range(max_steps):
            px, py, _ = self.world.pelvis_pos
            yaw = self.world.pelvis_yaw
            ex, ey = tx - px, ty - py
            dist = math.hypot(ex, ey)
            yaw_to_target = math.atan2(ey, ex)
            yaw_align_err = math.atan2(math.sin(yaw_to_target - yaw),
                                        math.cos(yaw_to_target - yaw))
            yaw_final_err = math.atan2(math.sin(tyaw - yaw),
                                        math.cos(tyaw - yaw))
            if dist > 0.05:
                # Drive forward in heading frame.
                cy, sy = math.cos(yaw), math.sin(yaw)
                # Rotate world err into body frame.
                bx = cy * ex + sy * ey
                by = -sy * ex + cy * ey
                f = float(np.clip(Kxy * bx, -self.walk_speed, self.walk_speed))
                s = float(np.clip(Kxy * by, -self.walk_speed, self.walk_speed))
                w_yr = float(np.clip(2.0 * yaw_align_err, -self.yaw_rate, self.yaw_rate))
            else:
                f = s = 0.0
                w_yr = float(np.clip(2.0 * yaw_final_err, -self.yaw_rate, self.yaw_rate))
                if abs(yaw_final_err) < 0.05:
                    print(f"[teleop] goto arrived: pelvis={self.world.pelvis_pos.round(3).tolist()} "
                          f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}°")
                    return
            self.loco.step(f, s, w_yr)
        print(f"[teleop] goto timed out: pelvis={self.world.pelvis_pos.round(3).tolist()}")

    # ------------------- actions ------------------------------------------- #

    def _do_detect(self) -> None:
        d = self.det.detect()
        self.last_detection = d
        if d.found:
            print(f"[teleop] DETECT bbox={d.bbox} pixel={d.pixel_xy} "
                  f"world={d.world_pos.round(3).tolist()} conf={d.world_pos_confidence:.2f}")
        else:
            print("[teleop] DETECT no red box visible")

    def _do_pickup(self, hand: str) -> None:
        if hand not in ("left", "right", "both"):
            print("[teleop] pickup needs hand=left|right|both"); return
        if self.last_detection is None or not self.last_detection.found:
            print("[teleop] no prior detection; running detect first")
            self._do_detect()
            if self.last_detection is None or not self.last_detection.found:
                print("[teleop] still no detection; aborting pickup"); return
        ok = pickup_detected_box(self.world, self.last_detection, hand=hand)  # type: ignore[arg-type]
        if ok:
            label = "both hands" if hand == "both" else f"{hand} hand"
            print(f"[teleop] PICKUP ok; held by {label}")
        else:
            print("[teleop] PICKUP failed (out of reach or IK timeout)")

    def _do_release(self) -> None:
        if self.world.held_by is None:
            print("[teleop] nothing to release"); return
        self.world.release()
        print("[teleop] released")

    def _do_place(self, hand: str) -> None:
        if hand not in ("left", "right", "both"):
            print("[teleop] place needs hand=left|right|both"); return
        if self.world.held_by != hand:
            print(f"[teleop] {hand} isn't holding anything (held_by={self.world.held_by})"); return
        ok = place_detected_box(self.world, hand=hand)  # type: ignore[arg-type]
        print(f"[teleop] PLACE ok={ok}, box={self.world.box_pos().round(3).tolist()}")

    def _toggle_record(self, force: Optional[bool] = None) -> None:
        want = (not self.recording) if force is None else bool(force)
        if want and not self.recording:
            path = self.args.video
            self.world.start_recording(path)
            self.recording = True
            print(f"[teleop] recording -> {path}")
        elif (not want) and self.recording:
            p = self.world.stop_recording()
            self.recording = False
            print(f"[teleop] recording stopped (wrote {p})")

    def _print_status(self) -> None:
        w = self.world
        print(f"  sim_time={w.sim_time:6.2f}  pelvis={w.pelvis_pos.round(3).tolist()} "
              f"yaw={math.degrees(w.pelvis_yaw):+.1f}°  active_arm={self.active_arm}  "
              f"held_by={w.held_by or '—'}")

    def _set_active_arm(self, hand: Optional[str]) -> None:
        # Deactivate the other arm controller so the kinematic walker can swing it.
        if hand == "left":
            self.left_arm.activate(); self.right_arm.deactivate()
            self.active_arm = "left"
            print("[teleop] active arm: left")
        elif hand == "right":
            self.right_arm.activate(); self.left_arm.deactivate()
            self.active_arm = "right"
            print("[teleop] active arm: right")
        else:
            self.left_arm.deactivate(); self.right_arm.deactivate()
            self.active_arm = None
            print("[teleop] active arm: none")

    # ------------------- input polling ------------------------------------- #

    def _read_keyboard(self) -> tuple[float, float, float, float, float, float, list[str]]:
        """Returns (forward, lateral, yaw_rate, ax, ay, az, events).

        ax, ay, az are world-frame Cartesian velocities for the active arm.
        events is a list of one-shot strings ('grasp', 'release', ...).
        """
        f = s = w = 0.0
        ax = ay = az = 0.0
        events: list[str] = []
        if self.pg is None:
            return f, s, w, ax, ay, az, events

        for ev in self.pg.event.get():
            if ev.type == self.pg.QUIT:
                events.append("quit")
            elif ev.type == self.pg.KEYDOWN:
                k = ev.key
                if   k == self.pg.K_ESCAPE: events.append("quit")
                elif k == self.pg.K_p:     events.append("detect")
                elif k == self.pg.K_g:     events.append("grasp")
                elif k == self.pg.K_r:     events.append("release")
                elif k == self.pg.K_x:     events.append("reset")
                elif k == self.pg.K_v:     events.append("record_toggle")
                elif k == self.pg.K_1:     events.append("arm:left")
                elif k == self.pg.K_2:     events.append("arm:right")
                elif k == self.pg.K_BACKQUOTE: events.append("arm:none")
                elif k == self.pg.K_LEFTBRACKET:  events.append("walk_speed:-")
                elif k == self.pg.K_RIGHTBRACKET: events.append("walk_speed:+")
                elif k == self.pg.K_MINUS: events.append("arm_speed:-")
                elif k == self.pg.K_EQUALS: events.append("arm_speed:+")
                elif k == self.pg.K_SLASH: events.append("help")
                elif k == self.pg.K_h:     events.append("hold")
                elif k == self.pg.K_f:     events.append("face_box")

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

        return f, s, w, ax, ay, az, events

    def _read_joystick(self) -> tuple[float, float, float, float, float, float, list[str]]:
        f = s = w = 0.0
        ax = ay = az = 0.0
        events: list[str] = []
        if self.joystick is None or self.pg is None:
            return f, s, w, ax, ay, az, events
        DEAD = 0.15
        def axis(i: int) -> float:
            try:
                v = float(self.joystick.get_axis(i))
            except Exception:
                return 0.0
            return v if abs(v) > DEAD else 0.0
        # Standard Xbox layout. Tilt forward on left stick = -1 on axis 1.
        f = -axis(1) * self.walk_speed
        s = -axis(0) * self.walk_speed
        w =  axis(2) * self.yaw_rate
        if self.active_arm is not None:
            az_v = -axis(3) * self.arm_speed
            # Triggers (axes 4, 5 on most controllers) for fine z too.
            try:
                lt = (axis(4) + 1) * 0.5
                rt = (axis(5) + 1) * 0.5
                az_v += (rt - lt) * self.arm_speed
            except Exception:
                pass
            az = az_v
            try:
                hat = self.joystick.get_hat(0)  # (-1,0,1) per axis
                ax = hat[1] * self.arm_speed
                ay = -hat[0] * self.arm_speed
            except Exception:
                pass
        # Buttons (one-shot via event queue).
        for ev in self.pg.event.get():
            if ev.type == self.pg.JOYBUTTONDOWN:
                idx = ev.button
                if   idx == 0: events.append("grasp")
                elif idx == 1: events.append("release")
                elif idx == 2: events.append("detect")
                elif idx == 3: events.append("arm_cycle")
                elif idx == 7: events.append("quit")
        return f, s, w, ax, ay, az, events

    # ------------------- HUD ----------------------------------------------- #

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
        # Draw the latest detection if any.
        d = self.last_detection
        if d is not None and d.found:
            x, y, w0, h0 = d.bbox
            sx = bgr.shape[1] / max(1, self.det.width)
            sy = bgr.shape[0] / max(1, self.det.height)
            x, y, w0, h0 = int(x*sx), int(y*sy), int(w0*sx), int(h0*sy)
            cv2.rectangle(bgr, (x, y), (x+w0, y+h0), (0, 255, 0), 2)
            txt = f"world=({d.world_pos[0]:.2f},{d.world_pos[1]:.2f},{d.world_pos[2]:.2f}) conf={d.world_pos_confidence:.2f}"
            cv2.putText(bgr, txt, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        held = self.world.held_by or "—"
        info = f"loco={self.loco.name()}  active_arm={self.active_arm}  held={held}  rec={'on' if self.recording else 'off'}"
        cv2.putText(bgr, info, (8, bgr.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("G1 teleop HUD", bgr)
        cv2.waitKey(1)
        self._hud_inited = True

    # ------------------- main loop ----------------------------------------- #

    def run(self, max_seconds: Optional[float] = None) -> int:
        target_dt_wall = 0.02  # ~50 Hz UI / control update
        last_wall = time.perf_counter()
        last_hud  = 0.0
        t0 = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            elapsed_wall = now - last_wall
            last_wall = now

            # Drain prompt commands.
            self._drain_commands()

            # Read input.
            f1, s1, w1, ax1, ay1, az1, ek = self._read_keyboard()
            f2, s2, w2, ax2, ay2, az2, ej = self._read_joystick()
            f, s, w_yr = f1 + f2, s1 + s2, w1 + w2
            ax, ay, az = ax1 + ax2, ay1 + ay2, az1 + az2
            for evt in ek + ej:
                if   evt == "quit":          self._stop.set()
                elif evt == "detect":        self._do_detect()
                elif evt == "grasp":         self._do_pickup(self.active_arm or "both")
                elif evt == "release":       self._do_release()
                elif evt == "reset":         self._handle_text_command("reset")
                elif evt == "record_toggle": self._toggle_record()
                elif evt == "arm:left":      self._set_active_arm("left")
                elif evt == "arm:right":     self._set_active_arm("right")
                elif evt == "arm:none":      self._set_active_arm(None)
                elif evt == "arm_cycle":
                    nxt = {"left": "right", "right": None, None: "left"}[self.active_arm]
                    self._set_active_arm(nxt)
                elif evt == "walk_speed:+":  self.walk_speed = min(2.0, self.walk_speed + 0.1)
                elif evt == "walk_speed:-":  self.walk_speed = max(0.1, self.walk_speed - 0.1)
                elif evt == "arm_speed:+":   self.arm_speed = min(1.0, self.arm_speed + 0.05)
                elif evt == "arm_speed:-":   self.arm_speed = max(0.05, self.arm_speed - 0.05)
                elif evt == "help":
                    print(__doc__.split("Run:")[0])
                elif evt == "hold":          f = s = w_yr = ax = ay = az = 0.0
                elif evt == "face_box":      self._face_detected_box()

            # Step several sim ticks per UI update so wall-clock ~ sim-clock.
            n_substeps = max(1, int(round(elapsed_wall / self.world.dt)))
            n_substeps = min(n_substeps, 20)  # cap so we don't blow up
            for _ in range(n_substeps):
                self.loco.step(f, s, w_yr)
                if self.left_arm.target.active:
                    if self.active_arm == "left":
                        self.left_arm.integrate_velocity(ax, ay, az)
                    self.left_arm.step()
                if self.right_arm.target.active:
                    if self.active_arm == "right":
                        self.right_arm.integrate_velocity(ax, ay, az)
                    self.right_arm.step()

            # HUD at ~10 Hz.
            if self.hud_enabled and (now - last_hud) > 0.1:
                self._render_hud()
                last_hud = now

            # Pygame draws (its tiny window).
            if self.pg is not None and self.pg_screen is not None:
                self.pg_screen.fill((20, 20, 20))
                try:
                    self.pg.display.flip()
                except Exception:
                    pass

            if max_seconds is not None and (now - t0) > max_seconds:
                self._stop.set()

            # Sleep until next UI tick.
            time.sleep(max(0.0, target_dt_wall - (time.perf_counter() - now)))

        # Cleanup. Give the prompt thread a moment to notice the stop event.
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

    def _face_detected_box(self) -> None:
        """Yaw the base toward the most recent detection (joystick assist)."""
        if self.last_detection is None or not self.last_detection.found:
            print("[teleop] no detection to face"); return
        bx, by, _ = self.last_detection.world_pos
        px, py, _ = self.world.pelvis_pos
        target_yaw = math.atan2(by - py, bx - px)
        # Bang-bang yaw command for ~ a second.
        for _ in range(int(1.0 / self.world.dt)):
            err = math.atan2(math.sin(target_yaw - self.world.pelvis_yaw),
                             math.cos(target_yaw - self.world.pelvis_yaw))
            self.loco.step(0.0, 0.0, np.clip(2.0 * err, -self.yaw_rate, self.yaw_rate))
            if abs(err) < 0.05:
                break


# --------------------- main --------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--rl-checkpoint", default=None,
                    help="Path to a torch policy checkpoint for legs (optional).")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer")
    ap.add_argument("--hud",    action="store_true", help="Open OpenCV camera + HUD window")
    ap.add_argument("--video",  default=str(Path(__file__).resolve().parents[1] / "scripts" / "logs" / "teleop.mp4"))
    ap.add_argument("--no-input", action="store_true",
                    help="Don't start the prompt thread (for scripted/headless runs)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Auto-quit after this many wall seconds (for tests)")
    args = ap.parse_args()
    app = TeleopApp(args)
    return app.run(max_seconds=args.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
