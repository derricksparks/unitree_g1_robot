import yaml


def test_task_config_loads():
    with open("config/task_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["robot"]["min_dof"] >= 10
    assert cfg["robot"]["payload_kg"] <= 2.0
    assert cfg["task"]["target_speed_mps"] >= 0.5
