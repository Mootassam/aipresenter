"""Registry of downloadable model files, with SHA-256 verification.

Every model the engine uses is listed here with a pinned URL and hash, so a model can never be
silently swapped (by a CDN change or an attacker) and every file's license is traceable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_MP = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    license: str


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        ModelSpec(
            "face_landmarker",
            "face_landmarker.task",
            f"{_MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
            "Apache-2.0",
        ),
        ModelSpec(
            "hand_landmarker",
            "hand_landmarker.task",
            f"{_MP}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
            "Apache-2.0",
        ),
        ModelSpec(
            "pose_landmarker_lite",
            "pose_landmarker_lite.task",
            f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
            "Apache-2.0",
        ),
        ModelSpec(
            "pose_landmarker_full",
            "pose_landmarker_full.task",
            f"{_MP}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
            "Apache-2.0",
        ),
        ModelSpec(
            "pose_landmarker_heavy",
            "pose_landmarker_heavy.task",
            f"{_MP}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
            "Apache-2.0",
        ),
    )
}


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str, models_dir: Path, *, timeout_s: float = 60.0) -> Path:
    """Return the local path of a model, downloading and verifying it if needed."""
    spec = MODELS[name]
    dest = models_dir / spec.filename
    if dest.exists() and sha256_file(dest) == spec.sha256:
        return dest

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading model %s", spec.name)
    try:
        with urllib.request.urlopen(spec.url, timeout=timeout_s) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        digest = sha256_file(tmp)
        if digest != spec.sha256:
            raise ModelIntegrityError(
                f"{spec.name}: hash mismatch (expected {spec.sha256[:12]}, got {digest[:12]})"
            )
        os.replace(tmp, dest)  # atomic: a half-downloaded file never looks valid
    finally:
        tmp.unlink(missing_ok=True)
    return dest
