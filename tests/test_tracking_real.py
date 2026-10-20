"""Runs the real MediaPipe tracker on samples/talk.mp4 (skipped when the file isn't there)."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Settings
from engine.io.frames import Frame

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "talk.mp4"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="samples/talk.mp4 not present")


def test_tracks_face_and_hands_in_sample(tmp_path: Path) -> None:
    from engine.tracking.mediapipe_tracker import MediaPipeTracker

    s = Settings()
    cap = cv2.VideoCapture(str(SAMPLE))
    # 4.0-5.0 s of the sample: both hands clearly visible
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    with MediaPipeTracker(s.tracking, s.paths.models) as tr:
        results = []
        for i in range(30):
            ok, img = cap.read()
            assert ok
            results.append(tr.process(Frame(img, i, i * 33.3)))
    cap.release()

    assert all(r.face is not None for r in results)
    assert sum(len(r.hands) == 2 for r in results) >= 25
    last = results[-1]
    assert last.face is not None and last.face.landmarks.shape == (478, 3)
    assert last.face.blendshapes.shape == (52,)
    assert {h.side for h in last.hands} == {"left", "right"}
    for h in last.hands:
        assert h.landmarks.shape == (21, 3)
