from __future__ import annotations

from pathlib import Path

from common.paths import DataPaths, default_data_dir


def test_env_override(tmp_path: Path) -> None:
    assert default_data_dir() == tmp_path / "data"


def test_ensure_creates_all_dirs(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "root").ensure()
    for d in paths.all_dirs():
        assert d.is_dir()
        assert d.parent == paths.root
    # idempotent
    paths.ensure()
