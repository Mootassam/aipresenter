from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from engine import models
from engine.models import MODELS, ModelIntegrityError, ModelSpec, ensure_model


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> bytes:
    payload = b"fake model bytes"
    spec = ModelSpec("fake", "fake.task", "https://example.invalid/fake.task",
                     hashlib.sha256(payload).hexdigest(), "MIT")  # fmt: skip
    monkeypatch.setitem(MODELS, "fake", spec)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    return payload


def test_download_and_verify(tmp_path: Path, fake_model: bytes) -> None:
    path = ensure_model("fake", tmp_path)
    assert path.read_bytes() == fake_model
    assert not list(tmp_path.glob("*.part"))


def test_existing_valid_file_not_redownloaded(
    tmp_path: Path, fake_model: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    ensure_model("fake", tmp_path)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: 1 / 0)
    assert ensure_model("fake", tmp_path).exists()


def test_hash_mismatch_rejected(
    tmp_path: Path, fake_model: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"tampered"))
    with pytest.raises(ModelIntegrityError):
        ensure_model("fake", tmp_path)
    assert not list(tmp_path.iterdir())  # nothing half-written left behind


def test_registry_entries_are_pinned_and_licensed() -> None:
    for spec in MODELS.values():
        assert len(spec.sha256) == 64
        assert "/latest/" not in spec.url
        assert spec.license
