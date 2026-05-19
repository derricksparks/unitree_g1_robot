from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

import torch

from src.scenario_lab.backend import build_vec_env_policy
from src.scenario_lab.run_config import ScenarioRolloutConfig, validate_for_run


class RolloutWorkerThread(QThread):
  """Runs headless mjlab steps off the GUI thread."""

  episode_completed = Signal(dict)
  """Fired once per finished episode on env index 0 (keys: episode_index, steps, cumulative_reward)."""

  status_update = Signal(str)
  rollout_finished = Signal()
  failed = Signal(str)

  def __init__(self, cfg: ScenarioRolloutConfig):
    super().__init__()
    self._cfg = cfg
    self._stop_requested = False

  def request_stop(self) -> None:
    self._stop_requested = True

  def run(self) -> None:  # noqa: D401
    try:
      validate_for_run(self._cfg)
    except Exception as exc:  # noqa: BLE001
      self.failed.emit(str(exc))
      self.rollout_finished.emit()
      return

    csv_path = self._cfg.log_csv_path
    csv_file_obj: Any | None = None
    csv_writer: csv.DictWriter | None = None
    if csv_path is not None:
      csv_path.parent.mkdir(parents=True, exist_ok=True)
      csv_file_obj = csv_path.open("w", newline="", encoding="utf-8")
      csv_writer = csv.DictWriter(
        csv_file_obj,
        fieldnames=[
          "episode_index",
          "wall_time_s",
          "steps_in_episode",
          "cumulative_reward_env0",
        ],
      )
      csv_writer.writeheader()

    env = None
    try:
      self.status_update.emit("Loading environment and policy…")
      env, policy = build_vec_env_policy(self._cfg)

      episodes_done = 0
      steps_global = 0
      ep_reward_sum = 0.0
      ep_steps = 0
      last_ep_reward = 0.0

      episode_index = 0
      wall_t0 = time.perf_counter()

      while episodes_done < self._cfg.episode_target and steps_global < self._cfg.max_steps:
        if self._stop_requested:
          self.status_update.emit("Stopping after user request.")
          break

        with torch.no_grad():
          obs = env.get_observations()
          actions = policy(obs)
          _obs_next, rew, dones, _extras = env.step(actions)

        steps_global += 1
        r0 = float(rew[0].item())
        ep_reward_sum += r0
        ep_steps += 1

        if dones[0].item() != 0:
          payload = {
            "episode_index": episode_index,
            "steps_in_episode": ep_steps,
            "cumulative_reward_env0": ep_reward_sum,
            "wall_time_s": round(time.perf_counter() - wall_t0, 6),
          }
          self.episode_completed.emit(payload)
          last_ep_reward = ep_reward_sum

          row = {
            "episode_index": episode_index,
            "wall_time_s": payload["wall_time_s"],
            "steps_in_episode": ep_steps,
            "cumulative_reward_env0": ep_reward_sum,
          }
          row["wall_time_s"] = round(time.perf_counter() - wall_t0, 6)
          if csv_writer is not None:
            csv_writer.writerow(row)
            csv_file_obj.flush()

          episode_index += 1
          episodes_done += 1
          ep_reward_sum = 0.0
          ep_steps = 0

        if (
          steps_global % self._cfg.emit_status_every_steps == 0
          or steps_global == 1
        ):
          msg = (
            f"Steps: {steps_global} | Env0 episodes logged: {episodes_done}/{self._cfg.episode_target}"
          )
          if episodes_done > 0:
            msg += f" | last_ep_reward_env0: {last_ep_reward:.3g}"
          self.status_update.emit(msg)

      self.rollout_finished.emit()

    except Exception as exc:  # noqa: BLE001
      self.failed.emit(f"{type(exc).__name__}: {exc}")
      self.rollout_finished.emit()

    finally:
      if csv_file_obj is not None:
        csv_file_obj.close()
      if env is not None:
        try:
          env.close()
        except Exception:  # noqa: BLE001, S112
          pass


def repo_root_from_this_file() -> Path:
  return Path(__file__).resolve().parents[2]


def default_metrics_path() -> Path:
  logs = repo_root_from_this_file() / "logs" / "scenario_lab_metrics"
  logs.mkdir(parents=True, exist_ok=True)
  ts = time.strftime("%Y%m%d_%H%M%S")
  return logs / f"rollout_{ts}.csv"
