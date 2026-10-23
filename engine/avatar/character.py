"""Character images: face detection, quality checks and preparation for the renderer.

Runs once when a user adds a character (and at stream start). Nothing here is per-frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from engine.avatar.liveportrait.crop import (
    CropTransform,
    eye_open_ratio,
    landmarks_px,
    lip_open_ratio,
    source_crop,
)
from engine.io.frames import Image
from engine.models import ensure_model

log = logging.getLogger(__name__)

F32 = npt.NDArray[np.float32]


@dataclass
class FaceInfo:
    landmarks: F32
    """(478, 3) normalised MediaPipe landmarks."""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class QualityIssue:
    code: str
    message: str
    blocking: bool
    """Blocking issues prevent using the image; others are warnings."""


@dataclass
class CharacterSource:
    canvas: Image
    """BGR working canvas (the character image resized to the render height)."""
    crop: CropTransform
    """canvas -> 512 face crop."""
    face_pts: F32
    """(478, 2) landmark pixels on the canvas."""
    eye_ratio: F32
    lip_ratio: float
    alpha: F32 | None = None
    """(H, W) person matte in [0, 1], or None if not computed."""
    issues: list[QualityIssue] = field(default_factory=list)


def read_image(path: Path) -> Image:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # transparent PNG: put it on mid-grey
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 128 * (1 - a)).astype(np.uint8)
    return np.asarray(img, dtype=np.uint8)


def _euler_from_matrix(m: F32) -> tuple[float, float, float]:
    """MediaPipe facial transformation matrix -> (pitch, yaw, roll) degrees."""
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def detect_faces(img: Image, models_dir: Path, max_faces: int = 3) -> list[FaceInfo]:
    """MediaPipe face landmarks on a still image."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model("face_landmarker", models_dir))),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.3,
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = []
    for i, pts in enumerate(res.face_landmarks):
        arr = np.array([(p.x, p.y, p.z) for p in pts], np.float32)
        mat = (
            np.asarray(res.facial_transformation_matrixes[i], np.float32)
            if res.facial_transformation_matrixes
            else np.eye(4, dtype=np.float32)
        )
        pitch, yaw, roll = _euler_from_matrix(mat)
        faces.append(FaceInfo(arr, yaw, pitch, roll))
    return faces


def check_quality(img: Image, faces: list[FaceInfo]) -> list[QualityIssue]:
    """What's wrong with a character image, in words a customer understands."""
    import cv2

    issues: list[QualityIssue] = []
    h, w = img.shape[:2]
    if min(h, w) < 512:
        issues.append(QualityIssue("low_res", f"Image is {w}x{h}; use at least 1024 px.", True))
    elif min(h, w) < 1024:
        issues.append(
            QualityIssue("low_res", "Image is small; 1024 px or more looks sharper.", False)
        )
    if not faces:
        issues.append(
            QualityIssue("no_face", "No face found. Use a clear front-facing portrait.", True)
        )
        return issues
    if len(faces) > 1:
        issues.append(
            QualityIssue("many_faces", "More than one face. Use an image with one person.", True)
        )

    f = faces[0]
    pts = landmarks_px(f.landmarks, w, h)
    face_w = float(pts[:, 0].max() - pts[:, 0].min())
    if face_w < 200:
        issues.append(
            QualityIssue(
                "face_small", "Face is too small. Crop closer to head and shoulders.", True
            )
        )
    elif face_w < 350:
        issues.append(
            QualityIssue(
                "face_small", "Face is a bit small; a closer crop gives more detail.", False
            )
        )
    if abs(f.yaw_deg) > 20 or abs(f.pitch_deg) > 20:
        issues.append(
            QualityIssue(
                "not_frontal", "Face should look straight at the camera.", abs(f.yaw_deg) > 35
            )
        )
    if abs(f.roll_deg) > 15:
        issues.append(QualityIssue("tilted", "Head is tilted; a straight head works best.", False))
    eyes = eye_open_ratio(pts)
    if float(eyes.min()) < 0.15:
        issues.append(QualityIssue("eyes_closed", "Eyes should be open.", True))
    if lip_open_ratio(pts) > 0.12:
        issues.append(
            QualityIssue("mouth_open", "Mouth should be closed (light smile is OK).", False)
        )

    # Sharpness inside the face box (variance of Laplacian)
    x0, y0 = np.maximum(pts.min(0).astype(int), 0)
    x1, y1 = pts.max(0).astype(int)
    face = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if face.size and cv2.Laplacian(face, cv2.CV_64F).var() < 40:
        issues.append(QualityIssue("blurry", "Face looks blurry. Use a sharper image.", False))

    # Room for the head to move: the 2.3x face crop should mostly lie inside the image
    crop = source_crop(pts)
    corners = crop.to_original(np.array([[0, 0], [512, 0], [0, 512], [512, 512]], np.float32))
    outside = np.mean(
        (corners[:, 0] < -0.15 * w) | (corners[:, 0] > 1.15 * w) | (corners[:, 1] < -0.15 * h)
    )
    if outside > 0.4:
        issues.append(
            QualityIssue(
                "tight_crop", "Leave more space around the head (hair, neck, shoulders).", False
            )
        )
    return issues


def stretch_matte(alpha: F32) -> F32:
    """Map the segmenter's floor / ceiling to 0 / 1.

    Its confidences never reach 0 on the background (~0.08 everywhere on the sample), which lets
    the old background show through faintly. The 1st / 99th percentiles are clear background /
    clear person in any usable portrait (the person covers far more than 1% and far less than 99%).
    """
    lo, hi = np.percentile(alpha, [1, 99])
    if hi - lo < 0.3:  # nothing clear to stretch (no person, or the image is all person)
        return alpha
    return np.clip((alpha - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def person_matte(img: Image, models_dir: Path) -> F32:
    """Soft person mask (hair, body, clothes) with MediaPipe's multiclass selfie segmenter."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(
            model_asset_path=str(ensure_model("selfie_multiclass", models_dir))
        ),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    # class 0 = background; person = 1 - background confidence
    bg = res.confidence_masks[0].numpy_view().astype(np.float32)
    alpha = stretch_matte(np.clip(1.0 - bg, 0.0, 1.0))
    if alpha.shape != img.shape[:2]:
        alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Edge-aware refinement so hair edges follow the image instead of the 256 px mask grid
    guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    try:
        alpha = cv2.ximgproc.guidedFilter(guide, alpha, radius=8, eps=1e-3)
    except AttributeError:  # opencv without contrib
        alpha = cv2.GaussianBlur(alpha, (0, 0), 2)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def prepare_character(
    path: Path, models_dir: Path, *, canvas_height: int = 768, with_matte: bool = False
) -> CharacterSource:
    """Load, check and prepare a character image. Raises ValueError on blocking issues."""
    import cv2

    img = read_image(path)
    faces = detect_faces(img, models_dir)
    issues = check_quality(img, faces)
    blocking = [i for i in issues if i.blocking]
    if blocking:
        raise ValueError("; ".join(i.message for i in blocking))

    h, w = img.shape[:2]
    scale = canvas_height / h
    cw = round(w * scale / 2) * 2  # even width for video encoders
    canvas = cv2.resize(
        img, (cw, canvas_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )
    pts = landmarks_px(faces[0].landmarks, cw, canvas_height)
    src = CharacterSource(
        canvas=np.asarray(canvas, np.uint8),
        crop=source_crop(pts),
        face_pts=pts,
        eye_ratio=eye_open_ratio(pts),
        lip_ratio=lip_open_ratio(pts),
        issues=issues,
    )
    if with_matte:
        src.alpha = person_matte(src.canvas, models_dir)
    for i in issues:
        log.warning("character image: %s", i.message)
    return src
