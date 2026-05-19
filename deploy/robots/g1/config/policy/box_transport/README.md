# Box transport policy (exported ONNX)

Expected layout after training/export:

```text
box_transport/v0/
  exported/policy.onnx
  params/deploy.yaml
```

`v0/params/deploy.yaml` defines observations, command placeholders, actions, and safety clamps for the G1 deploy stack.

## Dual-policy FSM (walking + manipulation)

Walking stays on **`config/policy/velocity`**; manipulation uses **this** folder. Both use `type: RLBase` — each state loads its own `deploy.yaml` + ONNX at **controller startup**. Do **not** register `Box_Transport` under `FSM["_"]` until `exported/policy.onnx` exists, or launch will fail.

Merge instructions and teleop/classical baselines:

- `DUAL_POLICY_AND_TELEOP.md`
- `../../fsm_box_transport_addon.example.yaml`
