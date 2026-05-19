#!/usr/bin/env python3
"""Desktop UI for mjlab scenario rollouts and optional native viewer (PySide6 / LGPL).

Runs from the **repository root** (or anywhere if ``pip install -e '.[desktop]'`` was run from the repo):

  cd /path/to/unitree_rl_mjlab && python scripts/scenario_desktop.py

**Qt / Linux:** Qt 6.5+ needs ``libxcb-cursor0`` for the **X11 (xcb)** plugin. Typical error text
mentions ``Could not load the Qt platform plugin "xcb"`` and ``libxcb-cursor0``.
Install on Debian/Ubuntu::

  sudo apt install libxcb-cursor0

If Wayland fails to auto-select, force it::

  QT_QPA_PLATFORM=wayland python scripts/scenario_desktop.py

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _pick_qt_platform() -> None:
  """Prefer Wayland when present to avoid brittle XCB + libxcb-cursor setups."""
  if os.environ.get("QT_QPA_PLATFORM"):
    return
  if os.environ.get("WAYLAND_DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "wayland"


def main() -> None:
  _pick_qt_platform()
  from PySide6.QtWidgets import QApplication

  from src.scenario_lab.mainwindow import ScenarioMainWindow

  app = QApplication(sys.argv)
  window = ScenarioMainWindow(repo_root=REPO_ROOT)
  window.show()
  sys.exit(app.exec())


if __name__ == "__main__":
  main()
