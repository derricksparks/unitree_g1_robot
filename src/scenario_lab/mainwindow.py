from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
  QComboBox,
  QFileDialog,
  QFormLayout,
  QGroupBox,
  QHBoxLayout,
  QLabel,
  QLineEdit,
  QMainWindow,
  QMessageBox,
  QPlainTextEdit,
  QPushButton,
  QSpinBox,
  QTableWidget,
  QTableWidgetItem,
  QVBoxLayout,
  QWidget,
)

import mjlab.tasks  # noqa: F401 — register mjlab tasks
import src.tasks  # noqa: F401 — register local tasks

from mjlab.tasks.registry import list_tasks

from src.scenario_lab.deploy_config import (
  backup_then_apply_preset,
  default_g1_ctrl_binary,
  g1_dual_preset_path,
)
from src.scenario_lab.run_config import AgentMode, ScenarioRolloutConfig, validate_for_run
from src.scenario_lab.worker import RolloutWorkerThread, default_metrics_path, repo_root_from_this_file

_AGENT_MODE_ITEMS: tuple[tuple[str, str], ...] = (
  ("Обученная политика", "trained"),
  ("Нулевая", "zero"),
  ("Случайная", "random"),
)

_G1_CONFIG_MODE_ITEMS: tuple[tuple[str, int], ...] = (
  ("Не менять config.yaml", 0),
  ("Двойной пресет (Velocity + Box_Transport), резервная копия и запуск", 1),
)


class ScenarioMainWindow(QMainWindow):
  """Warehouse scenario launcher: mjlab rollout metrics + optional visual play subprocess."""

  def __init__(self, repo_root: Path | None = None):
    super().__init__()
    self._repo_root = Path(repo_root) if repo_root is not None else repo_root_from_this_file()
    self._worker: RolloutWorkerThread | None = None
    self._visual_proc: QProcess | None = None
    self._deploy_proc: QProcess | None = None

    self.setWindowTitle("Лаборатория сценариев Unitree (mjlab / G1 — перенос коробки)")
    self.resize(980, 640)

    root = QWidget(self)
    self.setCentralWidget(root)
    outer = QVBoxLayout(root)

    form = QFormLayout()
    outer.addLayout(form)

    self._task_combo = QComboBox()
    self._task_combo.setEditable(True)
    for tid in sorted(list_tasks()):
      self._task_combo.addItem(tid)
    default_idx = self._task_combo.findText("Unitree-G1-Box-Transport")
    if default_idx >= 0:
      self._task_combo.setCurrentIndex(default_idx)
    form.addRow("Идентификатор задачи", self._task_combo)

    agent_row = QHBoxLayout()
    self._agent = QComboBox()
    for label, value in _AGENT_MODE_ITEMS:
      self._agent.addItem(label, value)
    agent_row.addWidget(self._agent, stretch=1)
    form.addRow("Агент / политика", agent_row)

    chk_row = QHBoxLayout()
    self._checkpoint = QLineEdit()
    btn_browse = QPushButton("Обзор…")
    btn_browse.clicked.connect(self._browse_checkpoint)
    chk_row.addWidget(self._checkpoint, stretch=1)
    chk_row.addWidget(btn_browse)
    form.addRow("Контрольная точка (.pt)", chk_row)

    self._motion = QLineEdit()
    self._motion.setPlaceholderText("Необязательно: файл движения .npz (для нулевого/случайного агента)")
    form.addRow("Файл движения", self._motion)

    num_row = QHBoxLayout()
    self._num_envs = QSpinBox()
    self._num_envs.setRange(1, 8192)
    self._num_envs.setValue(1)
    self._episode_target = QSpinBox()
    self._episode_target.setRange(1, 1_000_000)
    self._episode_target.setValue(10)
    self._max_steps = QSpinBox()
    self._max_steps.setRange(1, 999_999_999)
    self._max_steps.setValue(250_000)
    num_row.addWidget(QLabel("число сред"))
    num_row.addWidget(self._num_envs)
    num_row.addWidget(QLabel("эпизоды"))
    num_row.addWidget(self._episode_target)
    num_row.addWidget(QLabel("макс. шагов"))
    num_row.addWidget(self._max_steps)
    form.addRow("Ограничения", num_row)

    csv_row = QHBoxLayout()
    self._metrics_path = QLineEdit(str(default_metrics_path()))
    btn_metrics = QPushButton("Выбрать CSV…")
    btn_metrics.clicked.connect(self._browse_metrics_csv)
    csv_row.addWidget(self._metrics_path, stretch=1)
    csv_row.addWidget(btn_metrics)
    form.addRow("CSV метрик", csv_row)

    btn_row = QHBoxLayout()
    self._btn_run = QPushButton("Запуск без окна (headless)")
    self._btn_run.clicked.connect(self._toggle_rollout)
    self._btn_viewer = QPushButton("Открыть просмотр mjlab (native)")
    self._btn_viewer.clicked.connect(self._launch_visual_play)
    btn_row.addWidget(self._btn_run)
    btn_row.addWidget(self._btn_viewer)
    btn_row.addStretch(1)
    outer.addLayout(btn_row)

    deploy_group = QGroupBox(
      "deploy/robots/g1 — g1_ctrl (реальный робот или unitree_mujoco / симуляция DDS)"
    )
    deploy_outer = QVBoxLayout(deploy_group)
    df = QFormLayout()
    deploy_outer.addLayout(df)

    gx = QHBoxLayout()
    self._g1_exe = QLineEdit(str(default_g1_ctrl_binary(self._repo_root)))
    btn_gx = QPushButton("Обзор…")
    btn_gx.clicked.connect(self._browse_g1_exe)
    gx.addWidget(self._g1_exe, stretch=1)
    gx.addWidget(btn_gx)
    df.addRow("Исполняемый файл g1_ctrl", gx)

    self._g1_network = QLineEdit("lo")
    self._g1_network.setPlaceholderText(
      'Сеть DDS: «lo» для локальной симуляции, например «enp5s0» для робота'
    )
    df.addRow("DDS --network", self._g1_network)

    self._g1_config_mode = QComboBox()
    for label, _idx in _G1_CONFIG_MODE_ITEMS:
      self._g1_config_mode.addItem(label)
    df.addRow("Пресет FSM / ONNX", self._g1_config_mode)

    deploy_btn_row = QHBoxLayout()
    self._btn_g1_start = QPushButton("Запустить g1_ctrl")
    self._btn_g1_start.clicked.connect(self._launch_g1_ctrl)
    self._btn_g1_stop = QPushButton("Остановить g1_ctrl")
    self._btn_g1_stop.setEnabled(False)
    self._btn_g1_stop.clicked.connect(self._stop_g1_ctrl)
    deploy_btn_row.addWidget(self._btn_g1_start)
    deploy_btn_row.addWidget(self._btn_g1_stop)
    deploy_btn_row.addStretch(1)
    deploy_outer.addLayout(deploy_btn_row)

    hint = QLabel(
      "<b>Как это работает</b>: один процесс <code>g1_ctrl</code> загружает все RL-состояния из "
      "<code>config.yaml</code> (Velocity и Box_Transport используют ONNX в "
      "<code>config/policy/…</code>). Двойной пресет добавляет состояние FSM; "
      "переключение режимов — с геймпада (см. <code>DUAL_POLICY_AND_TELEOP.md</code>)."
    )
    hint.setWordWrap(True)
    hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    deploy_outer.addWidget(hint)

    outer.addWidget(deploy_group)

    self._table = QTableWidget(0, 4)
    self._table.setHorizontalHeaderLabels(
      ["эпизод", "шаги", "Σ награда (среда 0)", "время (CSV)"]
    )
    self._table.horizontalHeader().setStretchLastSection(True)
    outer.addWidget(self._table)

    self._log = QPlainTextEdit()
    self._log.setReadOnly(True)
    self._log.setMaximumBlockCount(5_000)
    outer.addWidget(self._log, stretch=1)

    legend = QLabel(
      "<b>MjLab</b>: поток rollout без окна; просмотр — <code>scripts/play.py</code>. "
      "<b>Развёртывание G1</b>: <code>g1_ctrl</code> загружает все RL-состояния FSM из "
      "<code>deploy/robots/g1/config/config.yaml</code>; двойной пресет объединяет ONNX "
      "ходьбы и переноса коробки (один процесс)."
    )
    legend.setWordWrap(True)
    legend.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    outer.addWidget(legend)

    self._append_log("Готово.")

  def _append_log(self, text: str) -> None:
    self._log.appendPlainText(text)

  def _browse_checkpoint(self) -> None:
    path, _ = QFileDialog.getOpenFileName(
      self, "Выбор контрольной точки", str(self._repo_root)
    )
    if path:
      self._checkpoint.setText(path)

  def _browse_metrics_csv(self) -> None:
    path, _ = QFileDialog.getSaveFileName(
      self, "CSV метрик", str(default_metrics_path())
    )
    if path:
      self._metrics_path.setText(path)

  def _collect_config(self) -> ScenarioRolloutConfig:
    task_id = self._task_combo.currentText().strip()
    agent_data = self._agent.currentData()
    agent = str(agent_data).strip().lower() if agent_data is not None else ""
    if agent not in {"trained", "zero", "random"}:
      raise ValueError(f"Недопустимый режим агента: {agent!r}")
    motion_raw = self._motion.text().strip()
    motion_path = Path(motion_raw) if motion_raw else None

    ck_raw = self._checkpoint.text().strip()
    ck = Path(ck_raw) if ck_raw else None
    typed_agent = cast(AgentMode, agent)

    return ScenarioRolloutConfig(
      task_id=task_id,
      checkpoint_file=ck,
      motion_file=motion_path,
      agent=typed_agent,
      num_envs=int(self._num_envs.value()),
      device=None,
      max_steps=int(self._max_steps.value()),
      episode_target=int(self._episode_target.value()),
      log_csv_path=Path(self._metrics_path.text().strip()).expanduser(),
      emit_status_every_steps=500,
    )

  def _toggle_rollout(self) -> None:
    if self._worker is not None:
      self._append_log("[ИНФО] Запрошена остановка.")
      self._worker.request_stop()
      self._btn_run.setEnabled(False)
      return

    cfg = self._collect_config()
    try:
      validate_for_run(cfg)
    except Exception as exc:  # noqa: BLE001
      QMessageBox.warning(self, "Неверная конфигурация", str(exc))
      return

    self._worker = RolloutWorkerThread(cfg)
    self._worker.episode_completed.connect(self._on_episode)
    self._worker.status_update.connect(self._append_log)
    self._worker.failed.connect(self._on_rollout_failed)
    self._worker.rollout_finished.connect(self._on_rollout_finished)

    self._btn_run.setText("Остановить rollout")
    self._btn_run.setEnabled(True)

    self._table.setRowCount(0)

    self._worker.start()
    self._append_log("[ИНФО] Поток rollout запущен.")

  def _on_episode(self, payload: dict) -> None:
    row = self._table.rowCount()
    self._table.insertRow(row)
    self._table.setItem(row, 0, QTableWidgetItem(str(payload["episode_index"])))
    self._table.setItem(row, 1, QTableWidgetItem(str(payload["steps_in_episode"])))
    reward = payload["cumulative_reward_env0"]
    self._table.setItem(row, 2, QTableWidgetItem(f"{reward:.4f}"))
    self._table.setItem(row, 3, QTableWidgetItem(f"{payload.get('wall_time_s', 0):.3f}s"))

  def _on_rollout_failed(self, message: str) -> None:
    QMessageBox.critical(self, "Ошибка rollout", message)

  def _on_rollout_finished(self) -> None:
    self._append_log("[ИНФО] Поток rollout завершён.")
    self._worker = None
    self._btn_run.setText("Запуск без окна (headless)")
    self._btn_run.setEnabled(True)

  def _launch_visual_play(self) -> None:
    cfg = self._collect_config()
    if cfg.agent == "trained":
      try:
        validate_for_run(cfg)
      except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(self, "Просмотр", str(exc))
        return

    play_script = self._repo_root / "scripts" / "play.py"
    args = [
      str(play_script),
      cfg.task_id,
      "--agent",
      cfg.agent,
      "--viewer",
      "native",
      "--num-envs",
      str(cfg.num_envs),
    ]
    mf = cfg.motion_file
    if mf is not None and mf.exists():
      args.extend(["--motion-file", str(mf)])
    if cfg.agent == "trained" and cfg.checkpoint_file is not None:
      args.extend(["--checkpoint-file", str(cfg.checkpoint_file)])

    proc = QProcess(self)
    proc.setWorkingDirectory(str(self._repo_root))
    proc.start(sys.executable, args)

    if not proc.waitForStarted(3000):
      QMessageBox.warning(
        self, "Не удалось запустить просмотр", "Не удалось запустить подпроцесс play."
      )
      return

    self._visual_proc = proc
    self._append_log("[ИНФО] Запущен подпроцесс просмотра mjlab (native).")

  def _browse_g1_exe(self) -> None:
    path, _ = QFileDialog.getOpenFileName(
      self, "Указать g1_ctrl", str(self._repo_root / "deploy" / "robots" / "g1")
    )
    if path:
      self._g1_exe.setText(path)

  def _launch_g1_ctrl(self) -> None:
    if self._deploy_proc is not None and (
      self._deploy_proc.state() != QProcess.ProcessState.NotRunning
    ):
      QMessageBox.information(self, "g1_ctrl", "Уже запущен.")
      return

    exe = Path(self._g1_exe.text().strip()).expanduser()
    if not exe.exists():
      QMessageBox.warning(self, "g1_ctrl", f"Исполняемый файл не найден:\n{exe}")
      return
    if not os.access(exe, os.X_OK):
      QMessageBox.warning(
        self,
        "g1_ctrl",
        f"Нет прав на выполнение ({exe}). Выполняли cmake --build?",
      )

    preset_dual = self._g1_config_mode.currentIndex() == 1
    if preset_dual:
      dual_path = g1_dual_preset_path(self._repo_root)
      if not dual_path.is_file():
        QMessageBox.critical(self, "g1_ctrl", f"Файл пресета не найден:\n{dual_path}")
        return

      ans = QMessageBox.question(
        self,
        "Применить конфигурацию dual-policy?",
        "Будет заменён deploy/robots/g1/config/config.yaml на "
        "config.dual_velocity_box_transport.yaml; текущий файл сохранится как "
        "config.yaml.bak.<timestamp>\n\n"
        "Должны существовать ONNX Velocity и Box_Transport, иначе g1_ctrl завершится.\n"
        "Продолжить?",
      )
      if ans != QMessageBox.StandardButton.Yes:
        return
      try:
        result = backup_then_apply_preset(self._repo_root, dual_path)
      except OSError as exc:
        QMessageBox.critical(self, "g1_ctrl", str(exc))
        return
      if result.backup_path is None:
        self._append_log("[DEPLOY] Создан config.yaml из пресета (раньше файла не было).")
      else:
        self._append_log(
          f"[DEPLOY] Резервная копия: {result.backup_path.name} ← применён {dual_path.name}"
        )

    net = self._g1_network.text().strip()
    proc = QProcess(self)
    proc.setWorkingDirectory(str(self._repo_root))
    proc.readyReadStandardOutput.connect(
      lambda: self._append_deploy_stream(bytes(proc.readAllStandardOutput()))
    )
    proc.readyReadStandardError.connect(
      lambda: self._append_deploy_stream(bytes(proc.readAllStandardError()))
    )
    proc.finished.connect(self._on_g1_ctrl_finished)
    prog = str(exe.resolve())
    args = []
    if net:
      args.extend(["--network", net])

    proc.setProgram(prog)
    proc.setArguments(args)
    proc.start()
    self._deploy_proc = proc
    self._btn_g1_start.setEnabled(False)
    self._btn_g1_stop.setEnabled(True)

    if not proc.waitForStarted(5_000):
      QMessageBox.critical(
        self, "g1_ctrl", "Не удалось запустить процесс (проверьте DDS и бинарник)."
      )
      self._reset_g1_ctrl_buttons()
      proc.deleteLater()
      self._deploy_proc = None
      return

    self._append_log(f"[DEPLOY] g1_ctrl запущен: {' '.join([prog] + args)}")

  def _append_deploy_stream(self, data: bytes) -> None:
    if not data:
      return
    try:
      text = data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
      return
    for line in text.splitlines():
      if line.strip():
        self._append_log(f"[g1_ctrl] {line}")

  def _stop_g1_ctrl(self) -> None:
    if self._deploy_proc is None:
      return
    self._deploy_proc.kill()
    self._append_log("[DEPLOY] Запрошена остановка (SIGKILL).")

  def _reset_g1_ctrl_buttons(self) -> None:
    self._btn_g1_start.setEnabled(True)
    self._btn_g1_stop.setEnabled(False)

  def _on_g1_ctrl_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
    exit_name = getattr(exit_status, "name", str(exit_status))
    self._append_log(f"[DEPLOY] g1_ctrl завершился: код={exit_code} статус={exit_name}")
    if self._deploy_proc:
      self._deploy_proc.deleteLater()
    self._deploy_proc = None
    self._reset_g1_ctrl_buttons()

