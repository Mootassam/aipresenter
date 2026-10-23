from __future__ import annotations

import time

import numpy as np
import pytest

from engine.avatar.debug_overlay import DebugOverlayRenderer
from engine.io.frames import Frame, FrameSource, Image
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


class SlowRenderer(DebugOverlayRenderer):
    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    def render(self, frame: Frame, track: TrackingResult, hud: list[str] | None = None) -> Image:
        time.sleep(self.delay_s)
        return super().render(frame, track, hud)


class FailingTracker(FakeTracker):
    def process(self, frame: Frame) -> TrackingResult:
        raise ValueError("model crashed")


def test_overlap_runs_and_reports() -> None:
    sink = NullSink(320, 180)
    p = VideoPipeline(
        SyntheticSource(limit=40), FakeTracker(), DebugOverlayRenderer(), [sink], overlap=True
    )
    rep = p.run(duration_s=5)
    assert rep.frames == sink.frames > 0
    assert rep.face_detected_pct == 100.0
    assert {"render", "output", "track_total"} <= rep.stages_ms.keys()
    assert rep.latency_ms.count == rep.frames


def test_overlap_is_faster_when_both_stages_are_slow() -> None:
    # tracking 30 ms + rendering 30 ms: one after the other ~16 fps, overlapped ~30 fps
    def run(overlap: bool) -> int:
        p = VideoPipeline(
            SyntheticSource(fps=200), FakeTracker(0.03), SlowRenderer(0.03), [NullSink(160, 120)],
            overlap=overlap,
        )  # fmt: skip
        return p.run(duration_s=1.5).frames

    assert run(True) > 1.5 * run(False)


def test_overlap_latency_does_not_build_up() -> None:
    p = VideoPipeline(
        SyntheticSource(fps=200), FakeTracker(0.02), SlowRenderer(0.04), [NullSink(160, 120)],
        overlap=True,
    )  # fmt: skip
    rep = p.run(duration_s=1.5)
    assert rep.dropped_frames > rep.frames
    assert rep.latency_ms.p95 < 150  # ~ track + render + one waiting frame, not growing


def test_overlap_reports_tracking_errors() -> None:
    p = VideoPipeline(
        SyntheticSource(), FailingTracker(), DebugOverlayRenderer(), [NullSink(8, 8)], overlap=True
    )
    with pytest.raises(RuntimeError, match="tracking failed"):
        p.run(duration_s=5)


def test_overlap_stops_at_end_of_file() -> None:
    p = VideoPipeline(
        SyntheticSource(limit=15), FakeTracker(), DebugOverlayRenderer(), [NullSink(8, 8)],
        overlap=True,
    )  # fmt: skip
    t = time.monotonic()
    rep = p.run(duration_s=10)
    assert 0 < rep.frames <= 15
    assert time.monotonic() - t < 5


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


class LosslessSource(SyntheticSource):
    """Like a file decoded with ``realtime=False``: the next frame waits until this one is taken."""

    realtime = False

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set() and (self.limit is None or n < self.limit):
            while self.slot.has_frame and not self._stop.is_set():
                time.sleep(0.0005)
            self._publish(np.full((120, 160, 3), n % 255, np.uint8))
            n += 1


class IndexSink(NullSink):
    def __init__(self) -> None:
        super().__init__(160, 120)
        self.indices: list[int] = []


class IndexRenderer(SlowRenderer):
    def __init__(self, delay_s: float, sink: IndexSink) -> None:
        super().__init__(delay_s)
        self.sink = sink

    def render(self, frame: Frame, track: TrackingResult, hud: list[str] | None = None) -> Image:
        self.sink.indices.append(frame.index)
        return super().render(frame, track, hud)


def test_overlap_keeps_every_frame_of_a_file_decoded_as_fast_as_possible() -> None:
    """Recording a file must not skip frames, or its sound no longer matches the picture."""
    sink = IndexSink()
    p = VideoPipeline(
        LosslessSource(limit=30), FakeTracker(0.005), IndexRenderer(0.02, sink), [sink],
        overlap=True,
    )  # fmt: skip
    rep = p.run(duration_s=10)
    assert sink.indices == list(range(30))
    assert rep.dropped_frames == 0


def test_lossless_overlap_stops_on_time() -> None:
    sink = IndexSink()
    p = VideoPipeline(
        LosslessSource(), FakeTracker(), IndexRenderer(0.01, sink), [sink], overlap=True
    )
    t = time.monotonic()
    p.run(duration_s=0.5)
    assert time.monotonic() - t < 3
    assert sink.indices == list(range(len(sink.indices)))
