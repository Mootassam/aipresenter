from __future__ import annotations

from pathlib import Path

import pytest

from common.cli import main

REPO = Path(__file__).resolve().parent.parent


def test_config_show(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-c", str(REPO / "config" / "default.yaml"), "config", "show"]) == 0
    out = capsys.readouterr().out
    assert "max_session_minutes: 35" in out


def test_config_validate_ok_and_bad(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "validate", str(REPO / "config" / "default.yaml")]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("audio: {host_api: mme}\n", encoding="utf-8")
    assert main(["config", "validate", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().err


def test_paths_creates_dirs(tmp_path: Path) -> None:
    assert main(["-c", str(REPO / "config" / "default.yaml"), "paths"]) == 0
    assert (tmp_path / "data" / "characters").is_dir()


def test_doctor_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    assert "Mode" in capsys.readouterr().out
