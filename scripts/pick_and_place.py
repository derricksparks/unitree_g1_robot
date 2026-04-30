#!/usr/bin/env python3
"""
Walk + Pick + Place demo for the Unitree G1 (pure MuJoCo, no SDK / no checkpoint).

Sequence:
  1. WALK_TO_TABLE : kinematically drive the floating base to the reception table.
  2. PICK          : DLS IK on the left arm, then attach the box rigidly to the hand.
  3. WALK_TO_SHELF : turn 180 and walk ~3.9 m carrying the box.
  4. PLACE         : IK to lower the box, release, settle under gravity.
  5. RETREAT       : walk back to home and idle.

Why "kinematic" walking? This repo trains an RL walking policy via mjlab, but a
trained checkpoint is not always available. The locomotion phases here are
visually plausible (the base slides toward each waypoint while the arms swing)
and let us focus on the manipulation phases. Drop-in replacement with the RL
policy is straightforward: replace G1World.walk_step with a call into the
trained policy on the underlying ManagerBasedRlEnv.

Usage:
  python scripts/pick_and_place.py                  # offscreen render, write video
  python scripts/pick_and_place.py --viewer         # interactive native viewer
  python scripts/pick_and_place.py --video out.mp4  # custom video path
  python scripts/pick_and_place.py --headless --no-video  # fastest smoke-test

Tested with mujoco==3.x.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as either `python scripts/pick_and_place.py` or `python -m scripts.pick_and_place`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _g1_sim import DEFAULT_SCENE, G1World, run_demo
else:
    from ._g1_sim import DEFAULT_SCENE, G1World, run_demo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="MJCF scene XML")
    ap.add_argument("--viewer", action="store_true", help="Open native MuJoCo viewer (interactive)")
    ap.add_argument("--headless", action="store_true", help="No video, no viewer (smoke test)")
    ap.add_argument("--no-video", action="store_true", help="Skip writing video")
    ap.add_argument("--video", default=str(Path(__file__).resolve().parents[1] / "scripts" / "logs" / "pick_and_place.mp4"),
                    help="Output video path (mp4)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    print(f"[pick_and_place] loading: {args.scene}")
    world = G1World(args.scene)
    print(f"[pick_and_place] dt={world.dt:.4f}s, nq={world.model.nq}, nv={world.model.nv}, neq={world.model.neq}")

    if args.viewer:
        world.open_viewer()
    if not args.headless and not args.viewer and not args.no_video:
        world.start_recording(args.video, width=args.width, height=args.height)
        print(f"[pick_and_place] writing video -> {args.video}")

    try:
        rc = run_demo(world, verbose=True)
    finally:
        path = world.stop_recording()
        if path:
            print(f"[pick_and_place] video written: {path}")
        if args.viewer:
            for _ in range(120):
                time.sleep(0.05)
                try:
                    world._viewer.sync()  # type: ignore[attr-defined]
                except Exception:
                    break
        world.close()

    print(f"[pick_and_place] sim_seconds={world.sim_time:.2f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
