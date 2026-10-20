from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from common.config import LoggingConfig
from common.log import JsonFormatter, redact, setup_logging


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("using key rpa_ABCDEF1234567890XYZ", "rpa_ABCDEF1234567890XYZ"),
        ("api_key=sk-live-12345", "sk-live-12345"),
        ('{"api_key": "abc123secret"}', "abc123secret"),
        ("Authorization: Bearer eyJhbGciOi", "eyJhbGciOi"),
        ("license_key: ab12-cd34-ef56-gh78-ij90", "ab12-cd34-ef56-gh78-ij90"),
        ("push rtmp://a.rtmp.youtube.com/live2/xxxx-yyyy-zzzz", "xxxx-yyyy-zzzz"),
        ("stream_key=live_123_abc", "live_123_abc"),
    ],
)
def test_redact_removes_secrets(raw: str, secret: str) -> None:
    out = redact(raw)
    assert secret not in out
    assert "***" in out


def test_redact_leaves_normal_text() -> None:
    msg = "fps=29.8 latency p95=142.0 ms on RTX 5050"
    assert redact(msg) == msg


def _cleanup() -> None:
    for h in list(logging.getLogger().handlers):
        if getattr(h, "_aip_handler", False):
            logging.getLogger().removeHandler(h)
            h.close()


def test_setup_logging_writes_redacted_file(tmp_path: Path) -> None:
    try:
        log = setup_logging(LoggingConfig(level="DEBUG"), tmp_path, "engine")
        log.info("connecting with %s", "rpa_ABCDEF1234567890XYZ")
        for h in logging.getLogger().handlers:
            h.flush()
        text = (tmp_path / "engine.log").read_text(encoding="utf-8")
        assert "connecting with" in text
        assert "rpa_ABCDEF1234567890XYZ" not in text
    finally:
        _cleanup()


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    try:
        setup_logging(LoggingConfig(), tmp_path)
        setup_logging(LoggingConfig(), tmp_path)
        ours = [h for h in logging.getLogger().handlers if getattr(h, "_aip_handler", False)]
        assert len(ours) == 2  # console + file, not doubled
    finally:
        _cleanup()


def test_json_formatter() -> None:
    rec = logging.LogRecord("aip", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    data = json.loads(JsonFormatter().format(rec))
    assert data["msg"] == "hello world"
    assert data["level"] == "INFO"
