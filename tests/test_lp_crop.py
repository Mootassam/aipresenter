from __future__ import annotations

import math

import numpy as np
import pytest

from engine.avatar.liveportrait.crop import (
    LEFT_EYE,
    MOUTH_CORNERS,
    RIGHT_EYE,
    DrivingCropper,
    eye_open_ratio,
    face_rect,
    lip_open_ratio,
    similarity,
    source_crop,
)


def synthetic_face(
    cx: float = 500, cy: float = 400, size: float = 200, roll_deg: float = 0
) -> np.ndarray:
    """478 points: a ring of face points plus eyes and mouth at plausible places."""
    pts = np.zeros((478, 2), np.float32)
    ang = np.linspace(0, 2 * np.pi, 478, endpoint=False)
    pts[:, 0] = np.cos(ang) * size / 2
    pts[:, 1] = np.sin(ang) * size * 0.6
    for idx, x in ((LEFT_EYE, 0.2), (RIGHT_EYE, -0.2)):
        eye = np.array(
            [[-0.05, 0], [-0.025, -0.02], [0.025, -0.02], [0.05, 0], [0.025, 0.02], [-0.025, 0.02]]
        )
        pts[list(idx)] = (eye + np.array([x, -0.1])) * size
    pts[list(MOUTH_CORNERS)] = np.array([[-0.12, 0.25], [0.12, 0.25]]) * size
    pts[13] = np.array([0, 0.24]) * size
    pts[14] = np.array([0, 0.26]) * size
    r = math.radians(roll_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]], np.float32)
    return (pts @ rot.T + [cx, cy]).astype(np.float32)


def test_face_rect_recovers_roll() -> None:
    for roll in (-20, 0, 15):
        _, _, angle = face_rect(synthetic_face(roll_deg=roll), scale=2.0)
        assert math.degrees(angle) == pytest.approx(roll, abs=0.5)


def test_similarity_maps_center_to_crop_center() -> None:
    t = similarity(np.array([300.0, 200.0], np.float32), 100.0, 0.3, 256)
    c = t.to_crop(np.array([[300.0, 200.0]], np.float32))[0]
    assert c == pytest.approx([128, 128], abs=1e-3)
    back = t.to_original(np.array([[128.0, 128.0]], np.float32))[0]
    assert back == pytest.approx([300, 200], abs=1e-3)


def test_source_crop_cancels_roll() -> None:
    pts = synthetic_face(roll_deg=20)
    crop = source_crop(pts, dsize=512)
    c = crop.to_crop(pts)
    l_eye, r_eye = c[list(LEFT_EYE)].mean(0), c[list(RIGHT_EYE)].mean(0)
    assert l_eye[1] == pytest.approx(r_eye[1], abs=1.0)  # eyes level after cropping


def test_driving_cropper_holds_still_then_reanchors() -> None:
    dc = DrivingCropper(dsize=256)
    base = synthetic_face()
    t0, moved0 = dc.update(base)
    assert moved0
    t1, moved1 = dc.update(base + np.array([10, 5], np.float32))  # small move: box must not follow
    assert not moved1
    assert np.allclose(t0.m_o2c, t1.m_o2c)
    _, moved2 = dc.update(base + np.array([400, 0], np.float32))  # face left the box
    assert moved2
    assert dc.reanchors == 2


def test_driving_crop_is_axis_aligned() -> None:
    t, _ = DrivingCropper().update(synthetic_face(roll_deg=25))
    assert t.m_o2c[0, 1] == pytest.approx(0)  # roll is kept for the motion extractor to see


def test_eye_and_lip_ratios() -> None:
    pts = synthetic_face()
    ear = eye_open_ratio(pts)
    assert ear.shape == (2,)
    assert 0.3 < ear[0] < 0.5
    closed = pts.copy()
    for idx in (LEFT_EYE, RIGHT_EYE):
        closed[list(idx)[1:3], 1] = closed[list(idx)[4:6], 1][::-1]
    assert eye_open_ratio(closed).max() < 0.05
    assert lip_open_ratio(pts) == pytest.approx(0.02 / 0.24, rel=0.05)
