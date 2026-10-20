"""Face crop geometry for LivePortrait, driven by MediaPipe face landmarks.

Ported from LivePortrait's ``utils/crop.py`` (MIT) and adapted to MediaPipe's 478-point mesh,
so no InsightFace / 106-point landmark model is needed.

Two kinds of crops:

* **Source (character) crop:** rotation-normalised square around the face, computed once.
* **Driving (camera) crop:** axis-aligned box that stays put while the head moves inside it
  (like LivePortrait's averaged "global bbox" for driving videos). Head motion then shows up as
  keypoint motion instead of being cancelled by a moving crop. The box only re-anchors when the
  face nears its edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

# MediaPipe face mesh indices
LEFT_EYE = (362, 385, 387, 263, 373, 380)  # person's left eye (image right)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
MOUTH_CORNERS = (61, 291)
LIPS_INNER_VERTICAL = (13, 14)


@dataclass(frozen=True)
class CropTransform:
    """Similarity transform between original image and a ``dsize`` x ``dsize`` crop."""

    m_o2c: F32  # 3x3, original -> crop
    dsize: int

    @property
    def m_c2o(self) -> F32:
        return np.linalg.inv(self.m_o2c).astype(np.float32)

    def to_crop(self, pts: F32) -> F32:
        return (pts @ self.m_o2c[:2, :2].T + self.m_o2c[:2, 2]).astype(np.float32)

    def to_original(self, pts: F32) -> F32:
        m = self.m_c2o
        return (pts @ m[:2, :2].T + m[:2, 2]).astype(np.float32)


def landmarks_px(landmarks: F32, width: int, height: int) -> F32:
    """Normalised MediaPipe landmarks (N, >=2) -> pixel coordinates (N, 2)."""
    return (landmarks[:, :2] * np.array([width, height], np.float32)).astype(np.float32)


def eye_mouth_axis(pts: F32) -> tuple[F32, F32]:
    """Eye-centre and mouth-centre points; their direction cancels head roll."""
    eye_c = (pts[list(LEFT_EYE)].mean(0) + pts[list(RIGHT_EYE)].mean(0)) / 2
    mouth_c = pts[list(MOUTH_CORNERS)].mean(0)
    return eye_c.astype(np.float32), mouth_c.astype(np.float32)


def face_rect(
    pts: F32, scale: float, vx_ratio: float = 0.0, vy_ratio: float = 0.0
) -> tuple[F32, float, float]:
    """Centre, square size and roll angle (rad) of the face, as in LivePortrait."""
    eye_c, mouth_c = eye_mouth_axis(pts)
    uy = mouth_c - eye_c
    n = float(np.linalg.norm(uy))
    uy = np.array([0.0, 1.0], np.float32) if n <= 1e-3 else uy / n
    ux = np.array([uy[1], -uy[0]], np.float32)
    angle = math.acos(float(np.clip(ux[0], -1, 1)))
    if ux[1] < 0:
        angle = -angle

    rot = np.stack([ux, uy])
    c0 = pts.mean(0)
    rpts = (pts - c0) @ rot.T
    lt, rb = rpts.min(0), rpts.max(0)
    c1 = (lt + rb) / 2
    size = float(max(rb - lt)) * scale
    center = c0 + ux * c1[0] + uy * c1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    return center.astype(np.float32), size, angle


def similarity(center: F32, size: float, angle: float, dsize: int) -> CropTransform:
    """Transform that maps a rotated square (centre, size, angle) onto the crop."""
    s = dsize / size
    c, si = math.cos(angle), math.sin(angle)
    cx, cy = float(center[0]), float(center[1])
    t = dsize / 2
    m = np.array(
        [
            [s * c, s * si, t - s * (c * cx + si * cy)],
            [-s * si, s * c, t - s * (-si * cx + c * cy)],
            [0.0, 0.0, 1.0],
        ],
        np.float32,
    )
    return CropTransform(m, dsize)


def source_crop(
    pts: F32, dsize: int = 512, scale: float = 2.3, vy_ratio: float = -0.125
) -> CropTransform:
    center, size, angle = face_rect(pts, scale, vy_ratio=vy_ratio)
    return similarity(center, size, angle, dsize)


class DrivingCropper:
    """Keeps an axis-aligned driving box steady; re-anchors only when the face leaves it."""

    def __init__(
        self,
        dsize: int = 256,
        scale: float = 2.2,
        vy_ratio: float = -0.1,
        reanchor_shift: float = 0.18,
        reanchor_zoom: float = 0.25,
    ) -> None:
        self.dsize = dsize
        self.scale = scale
        self.vy_ratio = vy_ratio
        self.reanchor_shift = reanchor_shift
        self.reanchor_zoom = reanchor_zoom
        self._center: F32 | None = None
        self._size = 0.0
        self.reanchors = 0

    def reset(self) -> None:
        self._center = None

    def update(self, pts: F32) -> tuple[CropTransform, bool]:
        """Return the crop for this frame and whether the box was (re-)anchored."""
        center, size, _ = face_rect(pts, self.scale, vy_ratio=self.vy_ratio)
        moved = False
        if self._center is None:
            moved = True
        else:
            shift = float(np.linalg.norm(center - self._center)) / self._size
            zoom = abs(size / self._size - 1.0)
            moved = shift > self.reanchor_shift or zoom > self.reanchor_zoom
        if moved:
            self._center, self._size = center, size
            self.reanchors += 1
        assert self._center is not None
        return similarity(self._center, self._size, 0.0, self.dsize), moved


def eye_open_ratio(pts: F32) -> F32:
    """Eye aspect ratio per eye (left, right): lid distance / eye width. ~0.3 open, <0.1 closed."""

    def ear(idx: tuple[int, ...]) -> float:
        p = pts[list(idx)]
        v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return float(v / (2 * h + 1e-6))

    return np.array([ear(LEFT_EYE), ear(RIGHT_EYE)], np.float32)


def lip_open_ratio(pts: F32) -> float:
    """Inner-lip opening / mouth width. 0 closed, ~0.5+ wide open."""
    a, b = LIPS_INNER_VERTICAL
    c, d = MOUTH_CORNERS
    return float(np.linalg.norm(pts[a] - pts[b]) / (np.linalg.norm(pts[c] - pts[d]) + 1e-6))
