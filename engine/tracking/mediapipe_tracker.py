"""Face + pose + hands tracking with MediaPipe Tasks (Apache-2.0), CPU.

The three landmarkers run in parallel threads (MediaPipe releases the GIL inside its C++
graph), so the per-frame cost is roughly the slowest model, not the sum.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np

from common.config import TrackingConfig
from common.timing import Stopwatch
from engine.io.frames import Frame, Image
from engine.models import ensure_model
from engine.tracking.one_euro import OneEuroFilter
from engine.tracking.types import F32, FaceResult, HandResult, PoseResult, TrackingResult

log = logging.getLogger(__name__)

# Silence MediaPipe's C++ info/warning spam (must be set before mediapipe is imported).
os.environ.setdefault("GLOG_minloglevel", "2")

# A hand that disappears for more than this many frames gets its smoothing state reset,
# so it doesn't "fly in" from its old position when it comes back.
_HAND_RESET_AFTER_MISSES = 3


def _lm_array(landmarks: Any, with_visibility: bool = False) -> F32:
    if with_visibility:
        return np.array(
            [(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0) for p in landmarks],
            dtype=np.float32,
        )
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)


class MediaPipeTracker:
    def __init__(self, cfg: TrackingConfig, models_dir: Path) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        self.cfg = cfg
        video = vision.RunningMode.VIDEO
        self._face = self._pose = self._hands = None
        if cfg.face:
            self._face = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(ensure_model("face_landmarker", models_dir))
                    ),
                    running_mode=video,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    min_face_detection_confidence=cfg.min_confidence,
                    min_tracking_confidence=cfg.min_confidence,
                )
            )
        if cfg.pose:
            self._pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(
                            ensure_model(f"pose_landmarker_{cfg.pose_model}", models_dir)
                        )
                    ),
                    running_mode=video,
                    num_poses=1,
                    min_pose_detection_confidence=cfg.min_confidence,
                    min_tracking_confidence=cfg.min_confidence,
                )
            )
        if cfg.hands and cfg.max_hands > 0:
            self._hands = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(ensure_model("hand_landmarker", models_dir))
                    ),
                    running_mode=video,
                    num_hands=cfg.max_hands,
                    min_hand_detection_confidence=cfg.min_confidence,
                    min_tracking_confidence=cfg.min_confidence,
                )
            )

        self.blendshape_names: list[str] = []
        self._pool = ThreadPoolExecutor(3, thread_name_prefix="mp") if cfg.parallel else None
        self._last_ts = -1

        self._f_face = OneEuroFilter(cfg.smoothing)
        self._f_blend = OneEuroFilter(cfg.blendshape_smoothing)
        self._f_pose = OneEuroFilter(cfg.smoothing)
        self._f_hand = {s: OneEuroFilter(cfg.hand_smoothing) for s in ("left", "right")}
        self._hand_misses = {"left": 0, "right": 0}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for lm in (self._face, self._pose, self._hands):
            if lm is not None:
                lm.close()

    def __enter__(self) -> MediaPipeTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- per frame ---------------------------------------------------------------------------

    def process(self, frame: Frame) -> TrackingResult:
        import mediapipe as mp

        # MediaPipe VIDEO mode needs strictly increasing integer timestamps.
        ts = max(int(frame.t_capture_ms), self._last_ts + 1)
        self._last_ts = ts
        import cv2

        h, w = frame.image.shape[:2]
        img: Image = frame.image
        scale = self.cfg.max_input_side / max(h, w)
        if scale < 1.0:  # landmarks are normalised, so downscaling doesn't change them
            size = (round(w * scale), round(h * scale))
            img = np.asarray(cv2.resize(img, size, interpolation=cv2.INTER_AREA), np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timings: dict[str, float] = {}

        def run(name: str, lm: Any) -> Any:
            if lm is None:
                return None
            with Stopwatch() as sw:
                out = lm.detect_for_video(image, ts)
            timings[name] = sw.elapsed_ms
            return out

        with Stopwatch() as total:
            jobs = (("face", self._face), ("pose", self._pose), ("hands", self._hands))
            if self._pool is not None:
                futures = [self._pool.submit(run, n, lm) for n, lm in jobs]
                face_r, pose_r, hands_r = (f.result() for f in futures)
            else:
                face_r, pose_r, hands_r = (run(n, lm) for n, lm in jobs)

            t = float(ts)
            result = TrackingResult(t, frame.index, w, h, timings_ms=timings)
            result.face = self._convert_face(face_r, t)
            result.pose = self._convert_pose(pose_r, t)
            result.hands = self._convert_hands(hands_r, result.pose, t)
        timings["track_total"] = total.elapsed_ms
        return result

    # -- conversion + smoothing --------------------------------------------------------------

    def _convert_face(self, r: Any, t: float) -> FaceResult | None:
        if r is None or not r.face_landmarks:
            self._f_face.reset()
            self._f_blend.reset()
            return None
        if not self.blendshape_names and r.face_blendshapes:
            self.blendshape_names = [c.category_name for c in r.face_blendshapes[0]]
        blend = (
            np.array([c.score for c in r.face_blendshapes[0]], dtype=np.float32)
            if r.face_blendshapes
            else np.zeros(52, dtype=np.float32)
        )
        transform = (
            np.asarray(r.facial_transformation_matrixes[0], dtype=np.float32)
            if r.facial_transformation_matrixes
            else np.eye(4, dtype=np.float32)
        )
        return FaceResult(
            landmarks=self._f_face(_lm_array(r.face_landmarks[0]), t),
            blendshapes=np.clip(self._f_blend(blend, t), 0.0, 1.0),
            transform=transform,
        )

    def _convert_pose(self, r: Any, t: float) -> PoseResult | None:
        if r is None or not r.pose_landmarks:
            self._f_pose.reset()
            return None
        lm = _lm_array(r.pose_landmarks[0], with_visibility=True)
        lm[:, :3] = self._f_pose(lm[:, :3], t)
        world = (
            _lm_array(r.pose_world_landmarks[0])
            if r.pose_world_landmarks
            else np.zeros((33, 3), np.float32)
        )
        return PoseResult(landmarks=lm, world=world)

    def _convert_hands(self, r: Any, pose: PoseResult | None, t: float) -> list[HandResult]:
        seen: set[str] = set()
        hands: list[HandResult] = []
        if r is not None:
            sides = self._assign_sides(r, pose)
            for i, side in enumerate(sides):
                if side is None or side in seen:
                    continue
                seen.add(side)
                lm = self._f_hand[side](_lm_array(r.hand_landmarks[i]), t)
                world = (
                    _lm_array(r.hand_world_landmarks[i])
                    if r.hand_world_landmarks
                    else np.zeros((21, 3), np.float32)
                )
                hands.append(HandResult(side, lm, world, float(r.handedness[i][0].score)))
        for s in ("left", "right"):
            if s in seen:
                self._hand_misses[s] = 0
            else:
                self._hand_misses[s] += 1
                if self._hand_misses[s] > _HAND_RESET_AFTER_MISSES:
                    self._f_hand[s].reset()
        return hands

    def _assign_sides(
        self, r: Any, pose: PoseResult | None
    ) -> list[Literal["left", "right"] | None]:
        """Decide which detected hand is the person's left / right.

        MediaPipe's handedness label assumes a mirrored (selfie) image and is sometimes wrong
        for one frame. When the pose wrists are visible we match each hand to the nearest wrist
        instead, which is far more stable. Otherwise fall back to the (un-mirrored) label.
        """
        n = len(r.hand_landmarks)
        out: list[Literal["left", "right"] | None] = []
        if pose is not None:
            lw, rw = pose.landmarks[15], pose.landmarks[16]
            if lw[3] > 0.5 and rw[3] > 0.5:
                wrists = [(r.hand_landmarks[i][0].x, r.hand_landmarks[i][0].y) for i in range(n)]
                d = [(math.dist(p, lw[:2]), math.dist(p, rw[:2])) for p in wrists]
                if n == 1:
                    return ["left" if d[0][0] < d[0][1] else "right"]
                if n == 2:
                    straight = d[0][0] + d[1][1]
                    crossed = d[0][1] + d[1][0]
                    return ["left", "right"] if straight <= crossed else ["right", "left"]
        for i in range(n):
            label = r.handedness[i][0].category_name.lower()
            if self.cfg.mirror_input:
                out.append("left" if label == "left" else "right")
            else:
                out.append("right" if label == "left" else "left")
        return out
