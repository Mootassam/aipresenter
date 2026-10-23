from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine.avatar.character import FaceInfo, check_quality

cv2 = pytest.importorskip("cv2")

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "character.png"


def _face(w: int, h: int, face_frac: float = 0.35, yaw: float = 0.0) -> FaceInfo:
    """Normalised landmarks of an ellipse face centred in a w x h image."""
    from tests.test_lp_crop import synthetic_face

    pts = synthetic_face(cx=w / 2, cy=h / 2, size=w * face_frac)
    norm = pts / np.array([w, h], np.float32)
    lm = np.concatenate([norm, np.zeros((478, 1), np.float32)], 1)
    return FaceInfo(lm, yaw_deg=yaw, pitch_deg=0.0, roll_deg=0.0)


def _noise(w: int, h: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def codes(issues: list) -> set[str]:  # type: ignore[type-arg]
    return {i.code for i in issues}


def test_no_face_is_blocking() -> None:
    issues = check_quality(_noise(1024, 1024), [])
    assert "no_face" in codes(issues)
    assert any(i.blocking for i in issues)


def test_good_image_has_no_blocking_issues() -> None:
    issues = check_quality(_noise(1200, 1200), [_face(1200, 1200)])
    assert not [i for i in issues if i.blocking]


def test_low_res_and_small_face() -> None:
    issues = check_quality(_noise(400, 400), [_face(400, 400, face_frac=0.3)])
    assert {"low_res", "face_small"} <= codes(issues)


def test_side_view_blocked() -> None:
    issues = check_quality(_noise(1200, 1200), [_face(1200, 1200, yaw=50)])
    assert any(i.code == "not_frontal" and i.blocking for i in issues)


def test_two_faces_blocked() -> None:
    f = _face(1200, 1200)
    assert "many_faces" in codes(check_quality(_noise(1200, 1200), [f, f]))


@pytest.mark.skipif(not SAMPLE.exists(), reason="samples/character.png not present")
def test_real_sample_is_usable() -> None:
    pytest.importorskip("mediapipe")
    from common.config import Settings
    from engine.avatar.character import detect_faces, read_image

    img = read_image(SAMPLE)
    faces = detect_faces(img, Settings().paths.models)
    assert len(faces) == 1
    assert not [i for i in check_quality(img, faces) if i.blocking]


def test_stretch_matte_cleans_the_floor() -> None:
    from engine.avatar.character import stretch_matte

    a = np.full((100, 100), 0.08, np.float32)  # the segmenter's floor on the background
    a[30:80, 20:80] = 0.97  # the person
    out = stretch_matte(a)
    assert out[0, 0] == 0.0 and out[50, 50] == 1.0


def test_stretch_matte_leaves_unclear_masks_alone() -> None:
    from engine.avatar.character import stretch_matte

    a = np.full((50, 50), 0.4, np.float32)
    np.testing.assert_array_equal(stretch_matte(a), a)
