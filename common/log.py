"""Logging setup shared by the GUI, engine processes, worker and license server.

* Console + rotating file (``<data_dir>/logs/<process>.log``).
* Optional JSON lines for the cloud worker (easy to ship to a log service).
* A redaction filter strips API keys, license keys and stream keys from every record, so a
  customer can send us a log file without leaking their RunPod or YouTube credentials.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from common.config import LoggingConfig

_HANDLER_TAG = "_aip_handler"

# Order matters: specific token shapes first, then generic key=value forms.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brpa_[A-Za-z0-9]{16,}\b"), "rpa_***"),  # RunPod API key
    (re.compile(r"\b[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}\b"), "****-****"),
    (re.compile(r"(rtmps?://[^\s/]+/[^\s]*/)[^\s/]+"), r"\1***"),  # stream key at end of RTMP URL
    (
        re.compile(
            r"(?i)\b(api[_-]?key|license[_-]?key|stream[_-]?key|token|secret|password|authorization)"
            r"(\s*[:=]\s*|\"\s*:\s*\")(?:Bearer\s+)?[^\s\",]+"
        ),
        r"\1\2***",
    ),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = redact(msg)
        if clean != msg:
            record.msg = clean
            record.args = None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "proc": record.processName,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(processName)s] %(name)s: %(message)s"


def setup_logging(
    cfg: LoggingConfig,
    log_dir: Path | None = None,
    process_name: str = "main",
) -> logging.Logger:
    """Configure the root logger. Safe to call more than once (replaces its own handlers)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()

    formatter: logging.Formatter = (
        JsonFormatter() if cfg.json_format else logging.Formatter(_TEXT_FORMAT)
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_dir / f"{process_name}.log",
                maxBytes=cfg.max_file_mb * 1024 * 1024,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
        )
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(RedactingFilter())
        setattr(h, _HANDLER_TAG, True)
        root.addHandler(h)

    root.setLevel(cfg.level)
    return logging.getLogger("aip")
