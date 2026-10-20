from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests away from the real user data dir and from AIP_* vars set on the dev box."""
    for key in list(os.environ):
        if key.startswith("AIP_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AIP_DATA_DIR", str(tmp_path / "data"))
    yield
