"""Tracking result types: the contract between trackers and avatar renderers.

All image-space landmarks are normalised: x, y in [0, 1] relative to the frame width/height,
z is relative depth (MediaPipe convention). Renderers must not depend on which tracker
backend produced them.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]

# MediaPipe hand landmark indices
WRIST = 0
FINGERS: dict[str, tuple[int, ...]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    *((WRIST, f[0]) for f in FINGERS.values()),
    *((a, b) for f in FINGERS.values() for a, b in itertools.pairwise(f)),
    (5, 9),
    (9, 13),
    (13, 17),
)

# MediaPipe pose indices we use (upper body only; legs are ignored by design)
POSE_UPPER: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_hip": 23,
    "r_hip": 24,
}
POSE_UPPER_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
)


@dataclass
class FaceResult:
    landmarks: F32
    """(478, 3) incl. irises."""
    blendshapes: F32
    """(52,) ARKit-style expression weights in [0, 1]; names in ``blendshape_names``."""
    transform: F32
    """(4, 4) head pose (rotation + translation) in a canonical face space."""


@dataclass
class PoseResult:
    landmarks: F32
    """(33, 4): x, y, z, visibility."""
    world: F32
    """(33, 3) metric 3D coordinates centred at the hips."""


@dataclass
class HandResult:
    side: Literal["left", "right"]
    """The person's anatomical hand (corrected for camera mirroring)."""
    landmarks: F32
    """(21, 3)."""
    world: F32
    """(21, 3) metric 3D coordinates centred at the hand."""
    score: float


@dataclass
class TrackingResult:
    t_ms: float
    frame_index: int
    width: int
    height: int
    face: FaceResult | None = None
    pose: PoseResult | None = None
    hands: list[HandResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def hand(self, side: Literal["left", "right"]) -> HandResult | None:
        return next((h for h in self.hands if h.side == side), None)
