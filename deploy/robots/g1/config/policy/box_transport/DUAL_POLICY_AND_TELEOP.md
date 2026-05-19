# Dual-policy FSM + classical / teleop baselines

## Why split locomotion and manipulation

Joint **velocity tracking** and **fingerless pinch / place** impose different distributions on the MDP. Shipping **two ONNX policies** and switching via the deployed FSM is usually more robust than forcing one giant policy to multitask unless you deliberately train multimodal RL.

Recommended stack:

| Mode | Policy | Observation contract | Typical entry |
|------|--------|------------------------|----------------|
| Walking | Velocity (`config/policy/velocity`) | Twist / gait phases | FixStand → **Velocity** |
| Manipulation (stationary) | Box transport (`config/policy/box_transport/v0`) | Object-centric terms (`deploy.yaml`) | FixStand → **Box_Transport** |

Operator habit: enter **FixStand** before manipulation so wrists and torso start from a repeatable pose near the workspace.

Recommended transitions (`deploy/robots/g1/config/config.yaml` edits):

1. Under **`FixStand.transitions`**, add entry-to-manipulation, for example  
   `Box_Transport: RB + B.on_pressed`  
   (pick any chord unused elsewhere).

2. Under **`Velocity.transitions`**, avoid swapping straight into manipulation unless you deliberately tame gait-stop timing—normally **`Passive` / `FixStand`**.

Snippet file (merge into `config.yaml` **only after** ONNX exists): `deploy/robots/g1/config/fsm_box_transport_addon.example.yaml`.

**Training**

- Walking: **`Unitree-G1-Flat`** (or the velocity task you already ship).
- Manipulation: **`Unitree-G1-Box-Transport`** in sim, export ONNX matching `v0/params/deploy.yaml`.

**Perception**

`commands.box_transport` in `deploy.yaml` are placeholders until filled from perception or fixed demo transforms. Constants + **teleop** (below) suffice to exercise the FSM safely in the lab.

## Classical / teleop baseline (no RL)

Use this when you cannot finish manipulation RL but need **repeatable motion** for integration tests.

1. **Teleop joint targets** (gamepad or keyboard) in a dedicated FSM state (not `RLBase`) that streams joint positions with limits—pattern matches `State_FixStand` interpolation, extended to full chain.
2. **IK + tracking**: compute palm `left_palm` / `right_palm` targets relative to a measured box frame; solve reduced IK (e.g. arms + waist) with root fixed; run high-rate joint PD. This lives **outside** the RL ONNX path; add a new `State_*` C++ class when you are ready.
3. **Record–replay**: log successful teleop joint trajectories once; replay open-loop for demos (fragile on contact change but fine for lab layout).

RL can return later to replace only the **Box_Transport** policy while **Velocity** stays unchanged.

## Safety

- Never switch **Velocity → Box_Transport** at full stride; go through **Passive** or **FixStand**.
- Keep `safety` blocks in each `deploy.yaml` conservative when testing a new policy.
