"""Smoke test for Unitree G1 asset inspection (does not touch manipulation pipeline XML/scripts)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "inspect_unitree_g1_assets.py"
UNITREE_DIR = PROJECT_ROOT / "simulation" / "mujoco" / "g1" / "assets" / "unitree_g1"
REPORT_JSON = PROJECT_ROOT / "results" / "unitree_g1_asset_inspection.json"


@pytest.mark.skipif(not UNITREE_DIR.is_dir(), reason="unitree_g1 asset folder not present")
def test_unitree_g1_folder_has_xml_candidates():
    xmls = list(UNITREE_DIR.glob("*.xml"))
    assert len(xmls) >= 1, "expected at least one top-level MJCF/XML in assets/unitree_g1"


@pytest.mark.skipif(not UNITREE_DIR.is_dir(), reason="unitree_g1 asset folder not present")
def test_inspect_script_runs_and_writes_json():
    pytest.importorskip("mujoco")
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_JSON.exists():
        REPORT_JSON.unlink()

    proc = subprocess.run(
        [sys.executable, str(INSPECT_SCRIPT), "--out", str(REPORT_JSON)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert REPORT_JSON.is_file(), "inspection script should write JSON report"
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert data.get("folder_exists") is True
    cands = data.get("candidates", [])
    assert len(cands) >= 1

    loads = data.get("loads", [])
    ok_loads = [L for L in loads if L.get("ok")]
    assert len(ok_loads) >= 1, "at least one MJCF should load in MuJoCo"

    hands = next((L for L in ok_loads if L["path"].endswith("g1_with_hands.xml")), None)
    assert hands is not None
    assert len(hands.get("hand_joint_names_left", [])) >= 1
    assert len(hands.get("hand_joint_names_right", [])) >= 1
    assert hands.get("nu", 0) > 29, "with-hands model should have more actuators than rubber-hand G1"

    comp = data.get("comparison_vs_pipeline", {})
    assert comp.get("ok") is True
    wh = comp.get("g1_with_hands.xml", {})
    assert "site_names_only_in_pipeline" in wh
    assert "right_palm_site" in (wh.get("site_names_only_in_pipeline") or [])
