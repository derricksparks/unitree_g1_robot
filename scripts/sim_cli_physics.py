#!/usr/bin/env python3
"""
Real-physics interactive CLI for the Unitree G1 + warehouse box.

Same UX as ``sim_cli.py`` (the kinematic CLI) but every locomotion command
is driven by the trained Unitree-G1-Flat ONNX policy, every PD step is
``mj_step``, and footsteps are visible. Pickup/Place are still automated:
the perception runs the onboard camera, ``goto`` walks the body to a
stand-off pose under the trained policy, and the grasp is a smooth-reach
DLS-IK in pelvis-clamped stand-still mode.

USAGE
-----
  python scripts/sim_cli_physics.py                                     # interactive prompt
  python scripts/sim_cli_physics.py --record demo.mp4                   # record everything
  python scripts/sim_cli_physics.py --script scripts/sim_cli_examples/physics_pick.cli

COMMANDS
--------
  status                       robot pose, fell flag, held_by, recording state
  reset                        reset the world to the start pose
  cmdvel f s w t               run a body-frame velocity command for t seconds
  goto x y [yaw]               walk to (x, y) [, facing yaw rad] using the policy
  detect                       perception step on the onboard d435i_rgb camera
  approach <left|right|both> [standoff]
                                walk to a stand-off in front of the last detection
                                so the chosen hand can reach it
  pickup [left|right|both]     reach + grasp + tuck into carry pose
  place [left|right|both]      reach + release + settle on the place_target
  arm <left|right|none>        activate / deactivate Cartesian arm controller
  armvel <left|right> vx vy vz [seconds]
                                drive an end-effector at (vx, vy, vz) for a duration
  carry on/off                 lock / unlock body-frame arm carry pose
  stand on/off                 force pelvis-clamped stand-still mode (manual override)
  record start [path] | stop   toggle mp4 recording
  speed [forward [yaw]]        set walk-speed limits (m/s, rad/s)
  hold [seconds]               step the policy with cmd=(0,0,0)
  run <path>                   execute a script file (one command per line)
  help                         this list (also: ?)
  quit                         leave the CLI (also: exit, q)
"""

from __future__ import annotations

import argparse
import cmd
import math
import shlex
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _g1_physics import DEFAULT_SCENE, G1PhysicsWorld, JOINT_NAMES_29
    from _g1_sim import BoxCamDetector
else:
    from ._g1_physics import DEFAULT_SCENE, G1PhysicsWorld, JOINT_NAMES_29
    from ._g1_sim import BoxCamDetector


def _build_eval_env(world: G1PhysicsWorld, det: BoxCamDetector) -> dict[str, Any]:
    box   = world.box_pos()
    plate = world.place_target_pos()
    pelvis = world.pelvis_pos
    lhand  = world.hand_pos("left")
    rhand  = world.hand_pos("right")
    last_det = getattr(det, "_last", None)
    last_det_pos = (np.asarray(last_det.world_pos)
                    if last_det is not None and last_det.found else box)
    class _Vec:
        __slots__ = ("x", "y", "z", "_a")
        def __init__(self, a: np.ndarray):
            self._a = np.asarray(a, dtype=float)
            self.x, self.y, self.z = float(a[0]), float(a[1]), float(a[2])
        def __getitem__(self, i): return float(self._a[i])
        def __repr__(self): return f"Vec({self.x:.3f},{self.y:.3f},{self.z:.3f})"
    return {
        "__builtins__": {"abs": abs, "min": min, "max": max, "round": round},
        "pi": math.pi, "tau": math.tau, "e": math.e,
        "deg": math.radians, "rad": math.degrees,
        "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
        "box": _Vec(box),
        "target": _Vec(plate), "place": _Vec(plate),
        "pelvis": _Vec(pelvis),
        "lhand": _Vec(lhand), "rhand": _Vec(rhand),
        "det": _Vec(last_det_pos),
    }


def _evalf(expr: str, env: dict[str, Any]) -> float:
    try:
        return float(eval(expr, env, env))  # noqa: S307
    except Exception as e:
        raise ValueError(f"could not parse number {expr!r}: {e}") from e


class G1PhysicsCli(cmd.Cmd):
    intro = (
        "\n=== Unitree G1 PHYSICS CLI ===\n"
        "Trained Flat policy on the legs + Cartesian DLS-IK on the arms.\n"
        "Type `help` (or `?`) for commands; `quit` to exit.\n"
        "Anchors usable in coordinate args: pi, deg(...), box.x/y/z,\n"
        "   target.x/y/z, pelvis.x/y/z, lhand.x/y/z, rhand.x/y/z, det.x/y/z\n"
    )
    prompt = "(g1-phy) "

    def __init__(self, world: G1PhysicsWorld, det: BoxCamDetector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world = world
        self.det = det
        self._lock = threading.RLock()

    def _ok(self, msg: str = "") -> None:
        print(f"  ok  {msg}" if msg else "  ok")
    def _err(self, msg: str) -> None:
        print(f"  ERR {msg}")

    def parseline(self, line):
        line = line.strip()
        if not line:
            return None, None, ""
        return super().parseline(line)
    def emptyline(self) -> bool:  # type: ignore[override]
        return False
    def default(self, line: str) -> bool:  # type: ignore[override]
        if line.strip().startswith("#"): return False
        self._err(f"unknown command: {line!r} (type `help`)"); return False

    # ----- commands ----- #

    def do_status(self, _arg: str) -> bool:
        """status        Print pose, fell flag, held_by, recording state."""
        with self._lock:
            w = self.world
            print(f"  sim_time     : {w.sim_time:7.2f} s   dt={w.dt:.4f}")
            print(f"  pelvis pos   : {w.pelvis_pos.round(3).tolist()}  "
                  f"yaw={math.degrees(w.pelvis_yaw):+.1f}°")
            print(f"  cmd          : {tuple(round(float(x),2) for x in w.cmd)}  "
                  f"fell={w.fell}  held_by={w.held_by or '—'}")
            print(f"  left  hand   : {w.hand_pos('left').round(3).tolist()}  "
                  f"active={w.left_arm.target.active}")
            print(f"  right hand   : {w.hand_pos('right').round(3).tolist()}  "
                  f"active={w.right_arm.target.active}")
            print(f"  box position : {w.box_pos().round(3).tolist()}")
            print(f"  place target : {w.place_target_pos().round(3).tolist()}")
            err = float(np.linalg.norm(w.box_pos()[:2] - w.place_target_pos()[:2]))
            print(f"  box->target  : {err:.3f} m (XY)")
            print(f"  recording    : {self.world.render_cfg.video_path or 'off'}")
        return False

    def do_reset(self, _arg: str) -> bool:
        """reset         Reset the world to the start pose."""
        with self._lock:
            self.world.reset()
            self.det._last = None  # type: ignore[attr-defined]
        self._ok("scene reset"); return False

    def do_cmdvel(self, arg: str) -> bool:
        """cmdvel <forward> <lateral> <yaw_rate> <seconds>
        Apply a body-frame velocity command for a duration. Subject to the
        carrying-payload velocity envelope when the box is held."""
        toks = arg.split()
        if len(toks) != 4:
            self._err("usage: cmdvel <forward> <lateral> <yaw> <seconds>"); return False
        try:
            f, s, w, dur = (float(t) for t in toks)
        except ValueError:
            self._err("expected 4 floats"); return False
        with self._lock:
            self.world.set_cmd(f, s, w)
            self.world.step_for(dur)
            self.world.set_cmd(0, 0, 0); self.world.step_for(0.2)
        self._ok(f"pelvis={self.world.pelvis_pos.round(3).tolist()} "
                 f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}° fell={self.world.fell}")
        return False

    def do_goto(self, arg: str) -> bool:
        """goto <x> <y> [<yaw>]       Walk to (x, y) [facing yaw rad] using
        the policy. Stops when within ~10 cm and the heading aligns."""
        toks = shlex.split(arg)
        if len(toks) not in (2, 3):
            self._err("usage: goto x y [yaw]"); return False
        env = _build_eval_env(self.world, self.det)
        try:
            xy = np.array([_evalf(toks[0], env), _evalf(toks[1], env)])
            yaw = _evalf(toks[2], env) if len(toks) == 3 else None
        except Exception as e:
            self._err(str(e)); return False
        with self._lock:
            arrived = self.world.goto(xy, target_yaw=yaw, timeout_s=80.0)
        msg = (f"pelvis={self.world.pelvis_pos.round(3).tolist()} "
               f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}° fell={self.world.fell}")
        self._ok(f"arrived={arrived}  {msg}") if arrived else self._err(f"timed out  {msg}")
        return False

    def do_detect(self, _arg: str) -> bool:
        """detect        Perception: HSV segmentation + ray-cast on d435i_rgb."""
        with self._lock:
            d = self.det.detect()
            self.det._last = d  # type: ignore[attr-defined]
        if d.found:
            self._ok(f"bbox={d.bbox} world={d.world_pos.round(3).tolist()} "
                     f"conf={d.world_pos_confidence:.2f}")
        else:
            self._err("no red box visible from d435i_rgb")
        return False

    def do_approach(self, arg: str) -> bool:
        """approach <left|right|both> [standoff]
        Walk to a stand-off pose in front of the last detection so the chosen
        hand can reach the box without over-extension. Default standoff = 0.45 m."""
        toks = arg.split()
        if not toks or toks[0] not in ("left", "right", "both"):
            self._err("usage: approach <left|right|both> [standoff]"); return False
        hand = toks[0]
        standoff = float(toks[1]) if len(toks) > 1 else 0.45
        last = getattr(self.det, "_last", None)
        if last is None or not last.found:
            with self._lock:
                last = self.det.detect()
                self.det._last = last  # type: ignore[attr-defined]
        if not last.found:
            self._err("no detection; aborting approach"); return False
        with self._lock:
            ok = self.world.approach_for_pickup(np.asarray(last.world_pos[:2]),
                                                 hand=hand, standoff=standoff)  # type: ignore[arg-type]
        msg = (f"pelvis={self.world.pelvis_pos.round(3).tolist()} "
               f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}°")
        self._ok(msg) if ok else self._err(f"timed out  {msg}")
        return False

    def do_pickup(self, arg: str) -> bool:
        """pickup [left|right|both]   Smooth reach + grasp + carry-pose tuck.

        Uses the last detection as the box target. Runs detect first if there
        is no detection yet. Single-hand "left" works most reliably with the
        shipped Flat policy; "both" is supported but harder to keep stable."""
        hand = (arg.strip().lower() or "left")
        if hand not in ("left", "right", "both"):
            self._err("usage: pickup [left|right|both]"); return False
        last = getattr(self.det, "_last", None)
        if last is None or not last.found:
            with self._lock:
                last = self.det.detect()
                self.det._last = last  # type: ignore[attr-defined]
        if not last.found:
            self._err("no detection; aborting pickup"); return False
        with self._lock:
            ok = self.world.pickup_box_at(np.asarray(last.world_pos), hand=hand)  # type: ignore[arg-type]
        if ok:
            self._ok(f"held_by={self.world.held_by} pelvis_z={self.world.pelvis_pos[2]:.3f} "
                     f"box={self.world.box_pos().round(3).tolist()}")
        else:
            self._err(f"PICKUP failed  pelvis_z={self.world.pelvis_pos[2]:.3f} fell={self.world.fell}")
        return False

    def do_place(self, arg: str) -> bool:
        """place [left|right|both]    Reach + release on place_target."""
        hand = arg.strip().lower() or None
        if hand and hand not in ("left", "right", "both"):
            self._err("usage: place [left|right|both]"); return False
        with self._lock:
            target = self.world.place_target_pos()
            ok = self.world.place_box_at(target, hand=hand)  # type: ignore[arg-type]
        err = float(np.linalg.norm(self.world.box_pos()[:2]
                                    - self.world.place_target_pos()[:2]))
        if ok:
            self._ok(f"box={self.world.box_pos().round(3).tolist()} "
                     f"err={err:.3f} m fell={self.world.fell}")
        else:
            self._err(f"PLACE failed  err={err:.3f} m fell={self.world.fell}")
        return False

    def do_arm(self, arg: str) -> bool:
        """arm <left|right|none>      Activate / deactivate arm controller."""
        sub = arg.strip().lower()
        with self._lock:
            if sub == "left":
                self.world.left_arm.activate_at_current(); self.world.right_arm.deactivate()
            elif sub == "right":
                self.world.right_arm.activate_at_current(); self.world.left_arm.deactivate()
            elif sub == "none":
                self.world.left_arm.deactivate(); self.world.right_arm.deactivate()
            else:
                self._err("usage: arm left|right|none"); return False
        self._ok(f"active={sub}"); return False

    def do_armvel(self, arg: str) -> bool:
        """armvel <left|right> vx vy vz [seconds]
        Drive the chosen end-effector at a Cartesian velocity (world frame)."""
        toks = arg.split()
        if len(toks) not in (4, 5) or toks[0] not in ("left", "right"):
            self._err("usage: armvel <left|right> vx vy vz [seconds]"); return False
        try:
            hand = toks[0]; vx, vy, vz = (float(t) for t in toks[1:4])
            dur = float(toks[4]) if len(toks) == 5 else 0.5
        except ValueError:
            self._err("expected: armvel <hand> vx vy vz [seconds]"); return False
        ctl = self.world.left_arm if hand == "left" else self.world.right_arm
        with self._lock:
            ctl.activate_at_current()
            end_t = self.world.sim_time + dur
            while self.world.sim_time < end_t and not self.world.fell:
                ctl.integrate_velocity(vx, vy, vz, self.world.dt)
                self.world.step()
        self._ok(f"{hand}_hand={self.world.hand_pos(hand).round(3).tolist()}")
        return False

    def do_carry(self, arg: str) -> bool:
        """carry on|off    Manually move arms into / out of the body-frame
        carry pose (normally pickup does this)."""
        sub = arg.strip().lower()
        if sub == "on":
            with self._lock:
                self.world.move_to_carry_pose(self.world.held_by or "left")
            self._ok("in carry pose")
        elif sub == "off":
            with self._lock:
                self.world.left_arm.deactivate(); self.world.right_arm.deactivate()
                self.world._arm_target_in_pelvis = {"left": None, "right": None}
            self._ok("carry off")
        else:
            self._err("usage: carry on|off"); return False
        return False

    def do_stand(self, arg: str) -> bool:
        """stand on|off    Pin floating base to default pose (no locomotion).

        Useful to hold the body steady while you teleop the arms by hand.
        Off resumes the trained policy."""
        sub = arg.strip().lower()
        with self._lock:
            if sub == "on":
                self.world._stand_still_mode = True
            elif sub == "off":
                self.world._stand_still_mode = False
            else:
                self._err("usage: stand on|off"); return False
        self._ok(f"stand={sub}"); return False

    def do_record(self, arg: str) -> bool:
        """record start [path] | stop | info"""
        toks = shlex.split(arg)
        if not toks:
            self._err("usage: record start [path] | stop | info"); return False
        sub = toks[0].lower()
        if sub == "start":
            path = toks[1] if len(toks) > 1 else str(
                Path(__file__).resolve().parents[1] / "scripts" / "logs" / "physics_cli.mp4")
            with self._lock:
                self.world.start_recording(path)
            self._ok(f"recording -> {path}")
        elif sub == "stop":
            with self._lock:
                p = self.world.stop_recording()
            self._ok(f"stopped (wrote {p})" if p else "no active recording")
        elif sub == "info":
            print(f"  recording: {self.world.render_cfg.video_path or 'off'}")
        else:
            self._err("usage: record start [path] | stop | info")
        return False

    def do_hold(self, arg: str) -> bool:
        """hold [seconds]   Step physics with cmd=(0,0,0) for N seconds (default 0.2)."""
        try:
            seconds = float(arg.strip()) if arg.strip() else 0.2
        except ValueError:
            self._err("usage: hold [seconds]"); return False
        with self._lock:
            self.world.set_cmd(0, 0, 0); self.world.step_for(seconds)
        self._ok(f"held {seconds:.2f}s"); return False

    def do_run(self, arg: str) -> bool:
        """run <path>      Execute a script file."""
        path = arg.strip()
        if not path or not Path(path).exists():
            self._err(f"file not found: {path!r}"); return False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                print(f"{self.prompt}{line}")
                self.onecmd(line)
        return False

    def do_quit(self, _arg: str) -> bool:
        """quit            Exit the CLI."""
        return True
    do_exit = do_quit
    do_q = do_quit
    do_EOF = do_quit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--no-policy", action="store_true",
                    help="Don't load the ONNX policy (will not actually walk)")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer at start")
    ap.add_argument("--record", default=None, help="Record everything to this mp4")
    ap.add_argument("--script", default=None, help="Run commands from this file then exit")
    ap.add_argument("--no-tty", action="store_true",
                    help="Read commands from stdin (no prompt) and exit on EOF")
    args = ap.parse_args()

    print(f"[sim_cli_physics] loading: {args.scene}")
    world = G1PhysicsWorld(args.scene, load_policy=not args.no_policy)
    det = BoxCamDetector(world)
    det._last = None  # type: ignore[attr-defined]
    print(f"[sim_cli_physics] dt={world.dt:.4f}s policy_dt={world.policy_cfg.step_dt:.3f}s "
          f"policy={'on' if world._sess is not None else 'off'}")
    if args.viewer:
        try:
            world.open_viewer(); print("[sim_cli_physics] viewer opened")
        except Exception as e:
            print(f"[sim_cli_physics] could not open viewer: {e}")
    if args.record:
        world.start_recording(args.record)
        print(f"[sim_cli_physics] recording -> {args.record}")

    cli = G1PhysicsCli(world, det)
    try:
        if args.script:
            cli.do_run(args.script)
        elif args.no_tty:
            for raw in sys.stdin:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                print(f"{cli.prompt}{line}")
                if cli.onecmd(line):
                    break
        else:
            cli.cmdloop()
    finally:
        try:
            p = world.stop_recording()
            if p:
                print(f"[sim_cli_physics] video written: {p}")
        except Exception:
            pass
        world.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
