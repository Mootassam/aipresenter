"""Debug renderer: draws tracking landmarks and a stats HUD over the camera image.

This is the Phase 1 "avatar". Later renderers implement the same ``render`` signature.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from engine.io.frames import Frame, Image
from engine.tracking.types import (
    FINGERS,
    HAND_CONNECTIONS,
    POSE_UPPER_CONNECTIONS,
    TrackingResult,
)

# BGR colours
_FACE = (200, 200, 200)
_POSE = (80, 220, 255)
_FINGER_COLOURS = {
    "thumb": (60, 60, 255),
    "index": (60, 200, 60),
    "middle": (255, 160, 40),
    "ring": (220, 60, 220),
    "pinky": (40, 220, 220),
}
_POINT_COLOUR = {i: _FINGER_COLOURS[name] for name, ids in FINGERS.items() for i in ids}


def _px(lm: npt.NDArray[np.float32], w: int, h: int) -> npt.NDArray[np.int32]:
    return np.column_stack((lm[:, 0] * w, lm[:, 1] * h)).round().astype(np.int32)


class DebugOverlayRenderer:
    def __init__(self, *, dim_background: bool = True) -> None:
        self.dim_background = dim_background

    def render(self, frame: Frame, track: TrackingResult, hud: list[str] | None = None) -> Image:
        import cv2

        img = frame.image.copy()
        if self.dim_background:
            img = np.asarray(cv2.convertScaleAbs(img, alpha=0.6), dtype=np.uint8)
        h, w = img.shape[:2]
        thick = max(1, round(min(w, h) / 360))

        if track.face is not None:
            pts = _px(track.face.landmarks, w, h)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
            pts = pts[ok]
            img[pts[:, 1], pts[:, 0]] = _FACE  # one pixel per mesh point: fast and readable

        if track.pose is not None:
            lm = track.pose.landmarks
            pts = _px(lm, w, h)
            for a, b in POSE_UPPER_CONNECTIONS:
                if lm[a, 3] > 0.5 and lm[b, 3] > 0.5:
                    cv2.line(img, tuple(pts[a]), tuple(pts[b]), _POSE, thick + 1, cv2.LINE_AA)

        for hand in track.hands:
            pts = _px(hand.landmarks, w, h)
            for a, b in HAND_CONNECTIONS:
                colour = _POINT_COLOUR.get(b, (255, 255, 255))
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), colour, thick, cv2.LINE_AA)
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p), thick + 1, _POINT_COLOUR.get(i, (255, 255, 255)), -1)
            label = f"{hand.side[0].upper()} {hand.score:.2f}"
            cv2.putText(
                img, label, tuple(pts[0] + (0, 12 * thick)), cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * thick, (255, 255, 255), thick, cv2.LINE_AA,
            )  # fmt: skip

        if hud:
            scale = 0.45 * thick
            for i, line in enumerate(hud):
                y = int((i + 1) * 22 * thick)
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                            thick + 2, cv2.LINE_AA)  # fmt: skip
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                            thick, cv2.LINE_AA)  # fmt: skip
        return img
