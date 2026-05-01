"""RL configuration for Unitree G1 box transport."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_box_transport_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO tuned for staged whole-body manipulation (long horizons, sparse late success).

  Matches extended curriculum thresholds in ``box_transport_env_cfg`` (~23.5k policy steps before
  final stage ramps). Increase ``max_iterations`` further if parallel env count is modest.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.92,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.012,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=6.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_box_transport",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=22000,
  )
