This directory is reserved for exported whole-body box transport policies.

Expected layout after training/export:

```text
box_transport/v0/
  exported/policy.onnx
  params/deploy.yaml
```

The FSM state should only be enabled after `exported/policy.onnx` exists. The
template `v0/params/deploy.yaml` documents the observation order and safety
settings expected by the G1 deploy controller.
