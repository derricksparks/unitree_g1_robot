#!/usr/bin/env python3
"""
Interactive command-line interface for the Unitree G1 warehouse simulation.

Drive the simulation by hand: walk the robot to coordinates, swing arms via
inverse kinematics, grasp/release the box, run the canned demo, record video,
inspect state, etc.

USAGE
-----
  python scripts/sim_cli.py                    # CLI only (headless)
  python scripts/sim_cli.py --viewer           # also open the native MuJoCo viewer
  python scripts/sim_cli.py --record demo.mp4  # record everything to mp4
  python scripts/sim_cli.py --no-tty < script  # batch mode (one command per line)

Type `help` (or `?`) in the CLI for the list of commands. Each command also
has its own help: `help walk`, `help reach`, etc.

QUICK TOUR
----------
  (g1) status
  (g1) walk 1.45 0 0
  (g1) reach left box.x box.y box.z+0.04
  (g1) grasp left
  (g1) walk -2.45 0 pi
  (g1) reach left target.x target.y target.z+0.06
  (g1) release
  (g1) settle 1.5
  (g1) walk 0 0 0
  (g1) record start cli_run.mp4
  (g1) demo
  (g1) record stop
  (g1) quit
"""

from __future__ import annotations

import argparse
import cmd
import math
import os
import shlex
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow running as either `python scripts/sim_cli.py` or `python -m scripts.sim_cli`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _g1_sim import (
        DEFAULT_SCENE, G1World, WalkParams, run_demo,
        KinematicLocomotionController, ArmCartesianController, BoxCamDetector,
        pickup_detected_box, place_detected_box,
    )
else:
    from ._g1_sim import (
        DEFAULT_SCENE, G1World, WalkParams, run_demo,
        KinematicLocomotionController, ArmCartesianController, BoxCamDetector,
        pickup_detected_box, place_detected_box,
    )


# --------------------------- helper: expression eval --------------------------

# Whitelist of names usable inside coordinate / numeric arguments.
def _build_eval_env(world: G1World) -> dict[str, Any]:
    box = world.box_pos()
    target = world.place_target_pos()
    pelvis = world.pelvis_pos
    lhand = world.hand_pos("left")
    rhand = world.hand_pos("right")

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
        "target": _Vec(target),
        "place": _Vec(target),
        "pelvis": _Vec(pelvis),
        "lhand": _Vec(lhand),
        "rhand": _Vec(rhand),
    }


def _evalf(expr: str, env: dict[str, Any]) -> float:
    """Evaluate `expr` as a float using a restricted name-space."""
    try:
        return float(eval(expr, env, env))  # noqa: S307 - intentional, restricted env
    except Exception as e:
        raise ValueError(f"could not parse number {expr!r}: {e}") from e


# --------------------------- the CLI ------------------------------------------

class G1Cli(cmd.Cmd):
    intro = (
        "\n=== Unitree G1 simulation CLI ===\n"
        "Type `help` or `?` for commands. Type `quit` to exit.\n"
        "Coordinate arguments support arithmetic and named anchors:\n"
        "  pi, tau, deg(...), rad(...)\n"
        "  box.x box.y box.z       (current box position)\n"
        "  target.x target.y target.z   (place target on the shelf)\n"
        "  pelvis.x pelvis.y pelvis.z\n"
        "  lhand.x lhand.y lhand.z      (left  end-effector)\n"
        "  rhand.x rhand.y rhand.z      (right end-effector)\n"
        "Examples:\n"
        "  walk 1.45 0 0\n"
        "  reach left box.x box.y box.z+0.04\n"
        "  walk pelvis.x-1.0 0 pi\n"
    )
    prompt = "(g1) "

    def __init__(self, world: G1World, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world = world
        self.walk_params = WalkParams()
        # Hybrid teleop add-ons: locomotion controller, per-arm Cartesian
        # controllers, and the camera detector. Shared with sim_cli.py so the
        # same commands work both interactively and from scripts.
        self.loco = KinematicLocomotionController(world, self.walk_params)
        self.left_arm = ArmCartesianController(world, "left")
        self.right_arm = ArmCartesianController(world, "right")
        self.detector = BoxCamDetector(world)
        self.last_detection = None
        # Lock to coordinate world mutations between the CLI thread and any
        # background "run" thread (used by `bg ...` and the viewer sync).
        self._lock = threading.RLock()

    # -- internal helpers -----------------------------------------------------

    def _parse_xyz(self, args: list[str]) -> np.ndarray:
        if len(args) != 3:
            raise ValueError("expected 3 coordinates: x y z")
        env = _build_eval_env(self.world)
        return np.array([_evalf(a, env) for a in args], dtype=float)

    def _parse_xy_yaw(self, args: list[str]) -> tuple[np.ndarray, float]:
        if len(args) not in (2, 3):
            raise ValueError("expected `x y [yaw]`")
        env = _build_eval_env(self.world)
        xy = np.array([_evalf(args[0], env), _evalf(args[1], env)], dtype=float)
        yaw = _evalf(args[2], env) if len(args) == 3 else self.world.pelvis_yaw
        return xy, yaw

    def _ok(self, msg: str = "") -> None:
        if msg:
            print(f"  ok  {msg}")
        else:
            print("  ok")

    def _err(self, msg: str) -> None:
        print(f"  ERR {msg}")

    def parseline(self, line):
        # Treat empty lines as no-op (default cmd.Cmd repeats the last command).
        line = line.strip()
        if not line:
            return None, None, ""
        return super().parseline(line)

    def emptyline(self) -> bool:  # type: ignore[override]
        return False

    def default(self, line: str) -> bool:  # type: ignore[override]
        if line.strip().startswith("#"):
            return False
        self._err(f"unknown command: {line!r} (type `help`)")
        return False

    # -- commands -------------------------------------------------------------

    def do_status(self, _arg: str) -> bool:
        """status                  Show robot/box/scene state."""
        with self._lock:
            w = self.world
            held = w.held_by or "—"
            print(f"  sim_time     : {w.sim_time:7.2f} s   dt={w.dt:.4f}")
            print(f"  pelvis pos   : {w.pelvis_pos.round(3).tolist()}  yaw={math.degrees(w.pelvis_yaw):+.1f}°")
            print(f"  left  hand   : {w.hand_pos('left').round(3).tolist()}")
            print(f"  right hand   : {w.hand_pos('right').round(3).tolist()}")
            print(f"  box position : {w.box_pos().round(3).tolist()}   held_by={held}")
            print(f"  place target : {w.place_target_pos().round(3).tolist()}")
            err = float(np.linalg.norm(w.box_pos()[:2] - w.place_target_pos()[:2]))
            print(f"  box->target  : {err:.3f} m (XY)")
            rc = w.render_cfg
            if rc.enabled:
                print(f"  recording    : -> {rc.video_path} ({rc.width}x{rc.height} @ {rc.fps} fps)")
            else:
                print(f"  recording    : off")
        return False

    def do_reset(self, _arg: str) -> bool:
        """reset                   Teleport robot to origin and box back onto the table."""
        with self._lock:
            self.world.reset()
        self._ok("scene reset")
        return False

    def do_walk(self, arg: str) -> bool:
        """walk x y [yaw]          Walk the floating base to (x, y) [facing yaw rad].

        `yaw` accepts `pi`, expressions like `deg(90)`, or omitted to keep current heading.
        """
        try:
            args = shlex.split(arg)
            xy, yaw = self._parse_xy_yaw(args)
        except Exception as e:
            self._err(str(e)); return False
        with self._lock:
            t0 = time.perf_counter()
            arrived = self.world.walk_to(xy, yaw, params=self.walk_params)
        wall = time.perf_counter() - t0
        msg = f"arrived={arrived}  pelvis={self.world.pelvis_pos.round(3).tolist()}  ({wall*1000:.0f} ms wall)"
        self._ok(msg) if arrived else self._err(msg)
        return False

    def do_goto(self, arg: str) -> bool:
        """goto <where>            Walk to a named anchor: table | shelf | home.

        Stops just outside arm reach of the anchor and faces the right way.
        """
        where = arg.strip().lower()
        if where == "table":
            xy, yaw = np.array([1.45, 0.0]), 0.0
        elif where == "shelf":
            xy, yaw = np.array([-2.45, 0.0]), math.pi
        elif where in ("home", "origin"):
            xy, yaw = np.array([0.0, 0.0]), 0.0
        else:
            self._err("expected one of: table | shelf | home"); return False
        with self._lock:
            arrived = self.world.walk_to(xy, yaw, params=self.walk_params)
        self._ok(f"arrived={arrived}  pelvis={self.world.pelvis_pos.round(3).tolist()}")
        return False

    def do_reach(self, arg: str) -> bool:
        """reach <left|right> x y z   Move the chosen hand to (x, y, z) using IK."""
        try:
            tokens = shlex.split(arg)
            if len(tokens) != 4 or tokens[0] not in ("left", "right"):
                raise ValueError("usage: reach <left|right> x y z")
            hand = tokens[0]
            target = self._parse_xyz(tokens[1:])
        except Exception as e:
            self._err(str(e)); return False
        with self._lock:
            ok, err = self.world.reach(hand, target, tol=0.05)
        msg = f"hand={hand}  target={target.round(3).tolist()}  err={err:.3f} m"
        self._ok(msg) if ok else self._err(f"IK timed out: {msg}")
        return False

    def do_grasp(self, arg: str) -> bool:
        """grasp <left|right|both>      Attach the box rigidly to the chosen hand(s)."""
        hand = arg.strip().lower() or "both"
        if hand not in ("left", "right", "both"):
            self._err("usage: grasp <left|right|both>"); return False
        with self._lock:
            self.world.grasp(hand)  # type: ignore[arg-type]
        label = "both hands" if hand == "both" else f"{hand} hand"
        self._ok(f"box now held by {label}")
        return False

    def do_release(self, _arg: str) -> bool:
        """release                 Detach the box from whatever hand is holding it."""
        with self._lock:
            self.world.release()
        self._ok("released")
        return False

    def do_settle(self, arg: str) -> bool:
        """settle [seconds]        Step physics on the box only (default 1.5s) so it falls.

        The robot itself is held frozen (no controller is wired up in this demo).
        """
        try:
            seconds = float(arg.strip()) if arg.strip() else 1.5
        except ValueError:
            self._err("usage: settle [seconds]"); return False
        with self._lock:
            self.world.settle_box(seconds=seconds)
        self._ok(f"settled {seconds:.2f}s of physics")
        return False

    def do_hold(self, arg: str) -> bool:
        """hold [seconds]          Keep ticking kinematics for N seconds (default 0.1).

        Useful for 'pausing' inside scripts or to let a video recorder catch up.
        """
        try:
            seconds = float(arg.strip()) if arg.strip() else 0.1
        except ValueError:
            self._err("usage: hold [seconds]"); return False
        with self._lock:
            self.world.hold(int(seconds / self.world.dt))
        self._ok(f"held {seconds:.2f}s")
        return False

    def do_demo(self, _arg: str) -> bool:
        """demo                    Run the canned walk -> pick -> walk -> place -> retreat."""
        with self._lock:
            try:
                run_demo(self.world, verbose=True)
                self._ok("demo complete")
            except Exception as e:
                self._err(f"demo failed: {e}")
                traceback.print_exc()
        return False

    def do_record(self, arg: str) -> bool:
        """record start [path]    Begin writing an mp4 (default scripts/logs/cli.mp4).
record stop             Stop the current recording.
record info             Show current recording state."""
        toks = shlex.split(arg)
        if not toks:
            self._err("usage: record start [path]  |  record stop  |  record info"); return False
        sub = toks[0].lower()
        if sub == "start":
            path = toks[1] if len(toks) > 1 else str(Path(__file__).resolve().parents[1] / "scripts" / "logs" / "cli.mp4")
            try:
                with self._lock:
                    self.world.start_recording(path)
                self._ok(f"recording -> {path}")
            except Exception as e:
                self._err(f"could not start recording: {e}")
        elif sub == "stop":
            with self._lock:
                p = self.world.stop_recording()
            self._ok(f"stopped (wrote {p})" if p else "no active recording")
        elif sub == "info":
            rc = self.world.render_cfg
            if rc.enabled:
                print(f"  recording -> {rc.video_path} ({rc.width}x{rc.height} @ {rc.fps} fps)")
            else:
                print("  recording: off")
        else:
            self._err("usage: record start [path]  |  record stop  |  record info")
        return False

    def do_viewer(self, arg: str) -> bool:
        """viewer on               Open the native MuJoCo viewer (needs a display).
viewer off              Close the viewer."""
        sub = arg.strip().lower()
        if sub in ("on", "open", ""):
            try:
                with self._lock:
                    self.world.open_viewer()
                self._ok("viewer opened")
            except Exception as e:
                self._err(f"could not open viewer: {e}")
        elif sub in ("off", "close"):
            with self._lock:
                self.world.close_viewer()
            self._ok("viewer closed")
        else:
            self._err("usage: viewer on  |  viewer off")
        return False

    def do_speed(self, arg: str) -> bool:
        """speed [m/s] [yaw_rate]  Set walk speed and yaw rate (no args = print current)."""
        toks = arg.split()
        if not toks:
            print(f"  speed_mps={self.walk_params.speed_mps}  yaw_rate={self.walk_params.yaw_rate}")
            return False
        try:
            self.walk_params.speed_mps = float(toks[0])
            if len(toks) > 1:
                self.walk_params.yaw_rate = float(toks[1])
            self._ok(f"speed_mps={self.walk_params.speed_mps}  yaw_rate={self.walk_params.yaw_rate}")
        except ValueError:
            self._err("usage: speed [m/s] [yaw_rate]")
        return False

    def do_box(self, arg: str) -> bool:
        """box at x y z            Teleport the box to (x, y, z) (resets its velocity).
box show                Print box pose."""
        toks = shlex.split(arg)
        if not toks:
            self._err("usage: box show  |  box at x y z"); return False
        sub = toks[0].lower()
        if sub == "show":
            print(f"  box pos: {self.world.box_pos().round(3).tolist()}  held_by={self.world.held_by or '—'}")
            return False
        if sub == "at" and len(toks) == 4:
            try:
                xyz = self._parse_xyz(toks[1:])
            except Exception as e:
                self._err(str(e)); return False
            with self._lock:
                self.world.data.qpos[self.world._box_qadr + 0] = xyz[0]
                self.world.data.qpos[self.world._box_qadr + 1] = xyz[1]
                self.world.data.qpos[self.world._box_qadr + 2] = xyz[2]
                self.world.data.qpos[self.world._box_qadr + 3] = 1.0
                self.world.data.qpos[self.world._box_qadr + 4:self.world._box_qadr + 7] = 0.0
                self.world.data.qvel[self.world._box_dofadr:self.world._box_dofadr + 6] = 0.0
                self.world.kine_tick()
            self._ok(f"box -> {xyz.round(3).tolist()}")
            return False
        self._err("usage: box show  |  box at x y z")
        return False

    def do_run(self, arg: str) -> bool:
        """run <path>              Execute commands from a file (one per line; '#' = comment)."""
        path = arg.strip()
        if not path or not Path(path).exists():
            self._err(f"file not found: {path!r}"); return False
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                print(f"{self.prompt}{line}")
                self.onecmd(line)
        return False

    def do_sleep(self, arg: str) -> bool:
        """sleep <seconds>         Wall-clock sleep (does NOT advance sim time)."""
        try:
            time.sleep(float(arg.strip()))
            self._ok()
        except ValueError:
            self._err("usage: sleep <seconds>")
        return False

    def do_cmdvel(self, arg: str) -> bool:
        """cmdvel <forward> <lateral> <yaw_rate> <seconds>
        Apply a body-frame velocity command for a duration.

        Example:  cmdvel 0.9 0 0 1.5     # walk forward at 0.9 m/s for 1.5 s
                  cmdvel 0 0 1.5 2.1     # spin in place
        """
        toks = arg.split()
        if len(toks) != 4:
            self._err("usage: cmdvel <forward> <lateral> <yaw_rate> <seconds>"); return False
        try:
            f, s, w_yr, dur = (float(t) for t in toks)
        except ValueError:
            self._err("expected 4 floats"); return False
        n = int(dur / self.world.dt)
        with self._lock:
            for _ in range(n):
                self.loco.step(f, s, w_yr)
        self._ok(f"cmdvel done; pelvis={self.world.pelvis_pos.round(3).tolist()} "
                 f"yaw={math.degrees(self.world.pelvis_yaw):+.1f}°")
        return False

    def do_detect(self, _arg: str) -> bool:
        """detect                  Run a perception step on the onboard d435i_rgb camera.

        Updates the CLI's `last detection` state. The estimated world position
        of the box is then accessible via `target.x` is NOT remapped; instead
        use `det.x det.y det.z` after a successful detection."""
        with self._lock:
            d = self.detector.detect()
            self.last_detection = d
        if d.found:
            self._ok(f"bbox={d.bbox} pixel={d.pixel_xy} world={d.world_pos.round(3).tolist()} "
                     f"conf={d.world_pos_confidence:.2f}")
        else:
            self._err("no red box visible from d435i_rgb")
        return False

    def do_pickup(self, arg: str) -> bool:
        """pickup [left|right|both]   Reach the LAST detected box and grasp it.

        Default is BOTH hands. Runs `detect` first if no detection is
        active. Gated on a successful camera detection."""
        hand = (arg.strip().lower() or "both")
        if hand not in ("left", "right", "both"):
            self._err("usage: pickup [left|right|both]"); return False
        if self.last_detection is None or not self.last_detection.found:
            self.do_detect("")
        if self.last_detection is None or not self.last_detection.found:
            self._err("no detection; aborting pickup"); return False
        with self._lock:
            ok = pickup_detected_box(self.world, self.last_detection, hand=hand)
        self._ok(f"PICKUP ok; held by {hand}") if ok else self._err("PICKUP failed (out of reach or IK timeout)")
        return False

    def do_place(self, arg: str) -> bool:
        """place [left|right|both]    Place the carried box on the place_target.

        Default matches whatever's currently being held (or BOTH if nothing)."""
        hand = (arg.strip().lower() or self.world.held_by or "both")
        if hand not in ("left", "right", "both"):
            self._err("usage: place [left|right|both]"); return False
        with self._lock:
            ok = place_detected_box(self.world, hand=hand)
        self._ok(f"PLACE ok; box={self.world.box_pos().round(3).tolist()}") if ok else self._err("PLACE failed")
        return False

    def do_armvel(self, arg: str) -> bool:
        """armvel <left|right> <vx> <vy> <vz> [seconds]
        Drive the chosen end-effector at a Cartesian velocity (world frame)
        through the ArmCartesianController (the WBC-flavored DLS IK)."""
        toks = arg.split()
        if len(toks) not in (4, 5) or toks[0] not in ("left", "right"):
            self._err("usage: armvel <left|right> <vx> <vy> <vz> [seconds]"); return False
        try:
            hand = toks[0]
            vx, vy, vz = float(toks[1]), float(toks[2]), float(toks[3])
            dur = float(toks[4]) if len(toks) == 5 else 0.5
        except ValueError:
            self._err("expected: armvel <hand> <vx> <vy> <vz> [seconds]"); return False
        ctl = self.left_arm if hand == "left" else self.right_arm
        with self._lock:
            ctl.activate()
            n = int(dur / self.world.dt)
            for _ in range(n):
                ctl.integrate_velocity(vx, vy, vz)
                ctl.step()
                self.world.kine_tick()
        self._ok(f"{hand}_hand={self.world.hand_pos(hand).round(3).tolist()} "
                 f"target={ctl.target.pos.round(3).tolist()}")
        return False

    def do_arm(self, arg: str) -> bool:
        """arm left|right|none     Activate the Cartesian controller for a hand
        (or release it back to the locomotion's arm swing)."""
        sub = arg.strip().lower()
        with self._lock:
            if sub == "left":
                self.left_arm.activate(); self.right_arm.deactivate()
            elif sub == "right":
                self.right_arm.activate(); self.left_arm.deactivate()
            elif sub == "none":
                self.left_arm.deactivate(); self.right_arm.deactivate()
            else:
                self._err("usage: arm left|right|none"); return False
        self._ok(f"active={sub}")
        return False

    # Aliases / quitters.
    def do_quit(self, _arg: str) -> bool:
        """quit                    Exit the CLI (also: exit, q, EOF)."""
        return True

    do_exit = do_quit
    do_q = do_quit
    do_EOF = do_quit


# --------------------------- main --------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="MJCF scene XML")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer at start")
    ap.add_argument("--record", default=None, help="Record everything to this mp4")
    ap.add_argument("--script", default=None, help="Run commands from this file then exit")
    ap.add_argument("--no-tty", action="store_true",
                    help="Read commands from stdin (no prompt) and exit on EOF")
    args = ap.parse_args()

    print(f"[sim_cli] loading: {args.scene}")
    world = G1World(args.scene)
    print(f"[sim_cli] dt={world.dt:.4f}s, nq={world.model.nq}, nv={world.model.nv}, neq={world.model.neq}")

    if args.viewer:
        try:
            world.open_viewer()
            print("[sim_cli] viewer opened")
        except Exception as e:
            print(f"[sim_cli] could not open viewer: {e}")

    if args.record:
        world.start_recording(args.record)
        print(f"[sim_cli] recording -> {args.record}")

    cli = G1Cli(world)

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
        path = world.stop_recording()
        if path:
            print(f"[sim_cli] video written: {path}")
        world.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
