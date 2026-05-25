"""
Load and inspect MuJoCo models for future full G1 integration.

This module is intentionally **standalone**: nothing here is imported by
`simulation/run_mujoco.py` or trial scripts so the sliding-base prototype stays
unaffected.

Usage::

    python simulation/mujoco/g1/load_g1_model.py --path simulation/mujoco/world.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


def load_mjcf(path: str | Path) -> mujoco.MjModel:
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return mujoco.MjModel.from_xml_path(str(p))


def load_urdf(path: str | Path) -> mujoco.MjModel:
    """
    Placeholder for a URDF→MuJoCo pipeline.

    Typical options to implement later:

    - ``mujoco.mj_compile`` / MuJoCo command-line converter to MJCF, then MJCF load.
    - Pinocchio or ``robot-descriptions-py`` URDF merged with MJCF includes.

    Raises:
        NotImplementedError: URDF path was given but conversion is not wired yet.
    """
    raise NotImplementedError(
        "URDF loading is not implemented yet; convert URDF→MJCF or pass an .xml path."
    )


def load_model(path: str | Path) -> mujoco.MjModel:
    """Dispatch on file extension (.xml/.mjcf vs .urdf)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xml", ".mjcf"):
        return load_mjcf(path)
    if suffix == ".urdf":
        return load_urdf(path)
    raise ValueError(
        f"Unsupported model extension {suffix!r}; use .xml, .mjcf, or future .urdf support."
    )


def print_model_summary(model: mujoco.MjModel) -> None:
    """Print joints, bodies, and actuators (names as registered in mjModel)."""

    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} njnt={model.njnt}")
    print()

    print("Bodies (id, name, parent)")
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[bid])
        label = name if name else f"<unnamed_body_{bid}>"
        parent = pname if pname else "<world>"
        print(f"  {bid:4d}  {label!r}  parent={parent!r}")
    print()

    print("Joints (id, type, name, body)")
    type_names = {v: k for k, v in vars(mujoco.mjtJoint).items() if k.startswith("mjJNT")}
    for jid in range(model.njnt):
        jtype = mujoco.mjtJoint(model.jnt_type[jid])
        tlabel = type_names.get(jtype, str(jtype))
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jbod = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[jid]
        )
        label = jname if jname else f"<joint_{jid}>"
        print(f"  {jid:4d}  {tlabel}  name={label!r}  body={jbod}")
    print()

    print("Actuators (id, name)")
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        label = name if name else f"<actuator_{aid}>"
        print(f"  {aid:4d}  {label!r}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect a MuJoCo MJCF model (future G1 path).")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .xml / .mjcf (or future .urdf once supported)",
    )
    args = parser.parse_args(argv)

    model = load_model(args.path)
    print(f"Loaded model from: {Path(args.path).resolve()}\n")
    print_model_summary(model)


if __name__ == "__main__":
    main()
