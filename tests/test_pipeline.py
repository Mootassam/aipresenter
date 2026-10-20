from __future__ import annotations

import time

import numpy as np
import pytest

from engine.avatar.debug_overlay import DebugOverlayRenderer
from engine.io.frames import Frame, FrameSource
from engine.output.video import NullSink, letterbox
from engine.pipeline import MemoryTrend, VideoPipeline
from engine.tracking.types import FaceResult, HandResult, PoseResult, TrackingResult

pytest.importorskip("cv2")


class SyntheticSource(FrameSource):
    def __init__(self, fps: float = 200.0, limit: int | None = None) -> None:
        super().__init__()
        self.fps = fps
        self.limit = limit

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set() and (self.limit is None or n < self.limit):
            self._publish(np.full((120, 160, 3), n % 255, np.uint8))
            n += 1
            time.sleep(1 / self.fps)


def _fake_track(frame: Frame) -> TrackingResult:
    rng = np.random.default_rng(frame.index)
    pose = np.concatenate([rng.random((33, 3), np.float32), np.ones((33, 1), np.float32)], 1)
    return TrackingResult(
        frame.t_capture_ms,
        frame.index,
        160,
        120,
        face=FaceResult(rng.random((478, 3), np.float32), np.zeros(52, np.float32), np.eye(4)),
        pose=PoseResult(pose, np.zeros((33, 3), np.float32)),
        hands=[
            HandResult("left", rng.random((21, 3), np.float32), np.zeros((21, 3)), 0.9),
            HandResult("right", rng.random((21, 3), np.float32), np.zeros((21, 3)), 0.9),
        ],
        timings_ms={"track_total": 1.0},
    )


class FakeTracker:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s

    def process(self, frame: Frame) -> TrackingResult:
        time.sleep(self.delay_s)
        return _fake_track(frame)

    def close(self) -> None:
        pass


def test_pipeline_runs_and_reports() -> None:
    sink = NullSink(320, 180)
    p = VideoPipeline(SyntheticSource(limit=40), FakeTracker(), DebugOverlayRenderer(), [sink])
    rep = p.run(duration_s=5)
    assert rep.frames == sink.frames > 0
    assert rep.face_detected_pct == 100.0
    assert {"render", "output", "track_total"} <= rep.stages_ms.keys()
    assert rep.latency_ms.count == rep.frames
    assert "fps" in rep.pretty()


def test_slow_tracker_drops_frames_instead_of_queueing() -> None:
    # Source at 200 fps, tracker takes 30 ms: latency must stay ~one processing time,
    # not grow with the run length.
    p = VideoPipeline(
        SyntheticSource(fps=200), FakeTracker(0.03), DebugOverlayRenderer(), [NullSink(160, 120)]
    )
    rep = p.run(duration_s=1.5)
    assert rep.dropped_frames > rep.frames
    assert rep.latency_ms.p95 < 80


def test_max_frames_stop() -> None:
    p = VideoPipeline(SyntheticSource(), FakeTracker(), DebugOverlayRenderer(), [NullSink(8, 8)])
    assert p.run(max_frames=10).frames == 10


def test_renderer_handles_out_of_frame_landmarks() -> None:
    frame = Frame(np.zeros((120, 160, 3), np.uint8), 0, 0.0)
    track = _fake_track(frame)
    assert track.face is not None
    track.face.landmarks[:10] = [1.5, -0.3, 0.0]  # off-screen points must not crash
    img = DebugOverlayRenderer().render(frame, track, ["hud"])
    assert img.shape == (120, 160, 3)


@pytest.mark.parametrize(("src", "dst"), [((1920, 1080), (720, 1280)), ((1080, 1920), (1280, 720))])
def test_letterbox_keeps_aspect(src: tuple[int, int], dst: tuple[int, int]) -> None:
    img = np.full((src[1], src[0], 3), 255, np.uint8)
    out = letterbox(img, *dst)
    assert out.shape == (dst[1], dst[0], 3)
    assert out[0, 0].sum() == 0 or out[0, dst[0] // 2].sum() == 0  # padded on one axis


def test_memory_trend_slope() -> None:
    m = MemoryTrend(warmup_s=0)
    for t in range(60):
        m.add(float(t), 100.0 + t / 60.0)  # +1 MB per minute
    assert m.slope_mb_per_min == pytest.approx(1.0, abs=0.01)
    assert m.peak_mb == pytest.approx(100 + 59 / 60)
