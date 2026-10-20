"""Where AI Presenter keeps its files on the user's machine.

Everything user-specific (models, characters, voices, logs) lives under one data root so that
uninstalling, backing up or moving the app is simple. The root can be overridden with the
``AIP_DATA_DIR`` environment variable (useful on the RunPod dev box and in tests).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_DIR_NAME = "AIPresenter"


def default_data_dir() -> Path:
    """Return the per-user data root for the current OS."""
    override = os.environ.get("AIP_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "ai-presenter"


@dataclass(frozen=True)
class DataPaths:
    """All well-known sub-directories of the data root."""

    root: Path

    @property
    def models(self) -> Path:
        """Base models downloaded on first run (encrypted at rest)."""
        return self.root / "models"

    @property
    def characters(self) -> Path:
        """The user's "My Characters" library."""
        return self.root / "characters"

    @property
    def voices(self) -> Path:
        """The user's "My Voices" library."""
        return self.root / "voices"

    @property
    def presets(self) -> Path:
        return self.root / "presets"

    @property
    def backgrounds(self) -> Path:
        return self.root / "backgrounds"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def cache(self) -> Path:
        """Disposable files (e.g. TensorRT engines built for this GPU)."""
        return self.root / "cache"

    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.models,
            self.characters,
            self.voices,
            self.presets,
            self.backgrounds,
            self.logs,
            self.cache,
        )

    def ensure(self) -> DataPaths:
        """Create every directory if missing. Returns self for chaining."""
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self
