"""Video pipeline: source -> tracker -> renderer -> sinks, with built-in measurements.

The loop runs in the calling thread. The source runs in its own thread and always hands over
the newest frame, so the reported latency is the real capture -> output delay inside the app
(camera sensor and consumer-side delays come on top and are measured separately).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import psutil

from common.timing import FpsMeter, LatencyStats, LatencySummary, Stopwatch, now_ms
from engine.io.frames import Frame, FrameSource, Image
from engine.output.video import VideoSink, letterbox
from engine.tracking.types import TrackingResult

log = logging.getLogger(__name__)


class Tracker(Protocol):
    def process(self, frame: Frame) -> TrackingResult: ...
    def close(self) -> None: ...


class Renderer(Protocol):
    def render(
        self, frame: Frame, track: TrackingResult, hud: list[str] | None = None
    ) -> Image: ...


@dataclass
class MemoryTrend:
    """RSS samples over time. ``slope_mb_per_min`` > ~1 over a long run means a leak."""

    samples: list[tuple[float, float]] = field(default_factory=list)  # (t_s, rss_mb)
    warmup_s: float = 10.0

    def add(self, t_s: float, rss_mb: float) -> None:
        self.samples.append((t_s, rss_mb))

    @property
    def slope_mb_per_min(self) -> float:
        pts = [(t, m) for t, m in self.samples if t >= self.warmup_s]
        if len(pts) < 3:
            return 0.0
        t, m = np.array(pts).T
        return float(np.polyfit(t, m, 1)[0] * 60.0)

    @property
    def peak_mb(self) -> float:
        return max((m for _, m in self.samples), default=0.0)


@dataclass
class PipelineReport:
    frames: int
    duration_s: float
    fps: float
    dropped_frames: int
    latency_ms: LatencySummary
    stages_ms: dict[str, LatencySummary]
    hands_detected_pct: float
    face_detected_pct: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_slope_mb_per_min: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            f"frames {self.frames} in {self.duration_s:.1f}s -> {self.fps:.1f} fps "
            f"(dropped at source: {self.dropped_frames})",
            f"latency capture->output: {self.latency_ms}",
        ]
        lines += [f"  {name:<12} {s}" for name, s in self.stages_ms.items()]
        lines += [
            f"face found {self.face_detected_pct:.0f}% of frames, "
            f"hands {self.hands_detected_pct:.0f}% (at least one)",
            f"memory: start {self.rss_start_mb:.0f} MB, peak {self.rss_peak_mb:.0f} MB, "
            f"trend {self.rss_slope_mb_per_min:+.2f} MB/min",
        ]
        return "\n".join(lines)


class VideoPipeline:
    def __init__(
        self,
        source: FrameSource,
        tracker: Tracker,
        renderer: Renderer,
        sinks: Sequence[VideoSink],
        *,
        hud: bool = True,
    ) -> None:
        self.source = source
        self.tracker = tracker
        self.renderer = renderer
        self.sinks = list(sinks)
        self.hud = hud
        self.fps = FpsMeter()
        self.latency = LatencyStats(4096)
        self.stages: dict[str, LatencyStats] = {}
        self._proc = psutil.Process()

    def _stage(self, name: str, ms: float) -> None:
        self.stages.setdefault(name, LatencyStats(4096)).add(ms)

    def _hud_lines(self, track: TrackingResult) -> list[str]:
        lat = self.latency.summary()
        return [
            f"{self.fps.fps:5.1f} fps   latency p50 {lat.p50:4.0f} ms  p95 {lat.p95:4.0f} ms",
            f"track {track.timings_ms.get('track_total', 0):4.1f} ms   "
            f"face {'Y' if track.face else '-'}  hands {len(track.hands)}",
        ]

    def run(
        self,
        *,
        duration_s: float | None = None,
        max_frames: int | None = None,
        stop: threading.Event | None = None,
        on_stats: Callable[[PipelineReport], None] | None = None,
        stats_every_s: float = 5.0,
    ) -> PipelineReport:
        t0 = now_ms()
        mem = MemoryTrend()
        rss_start = self._proc.memory_info().rss / 2**20
        mem.add(0.0, rss_start)
        next_mem = next_stats = t0 + 1000.0
        frames = with_face = with_hands = 0

        self.source.start()
        try:
            while True:
                if stop is not None and stop.is_set():
                    break
                if duration_s is not None and now_ms() - t0 >= duration_s * 1000:
                    break
                if max_frames is not None and frames >= max_frames:
                    break
                if any(getattr(s, "quit_requested", False) for s in self.sinks):
                    break

                frame = self.source.read(timeout_s=2.0)
                if frame is None:
                    if self.source.error is not None:
                        raise RuntimeError("frame source failed") from self.source.error
                    if self.source.slot.closed:
                        break  # end of file without loop
                    log.warning("no frame for 2 s")
                    continue

                track = self.tracker.process(frame)
                for k, v in track.timings_ms.items():
                    self._stage(k, v)

                with Stopwatch() as sw:
                    img = self.renderer.render(
                        frame, track, self._hud_lines(track) if self.hud else None
                    )
                self._stage("render", sw.elapsed_ms)
                for k, v in getattr(self.renderer, "timings_ms", {}).items():
                    self._stage(f"  {k}", v)

                with Stopwatch() as sw:
                    scaled: dict[tuple[int, int], Image] = {}
                    for sink in self.sinks:
                        size = (sink.width, sink.height)
                        if size not in scaled:
                            scaled[size] = letterbox(img, *size)
                        sink.send(scaled[size])
                self._stage("output", sw.elapsed_ms)

                t_out = now_ms()
                self.latency.add(t_out - frame.t_capture_ms)
                self.fps.tick(t_out)
                frames += 1
                with_face += track.face is not None
                with_hands += bool(track.hands)

                if t_out >= next_mem:
                    mem.add((t_out - t0) / 1000, self._proc.memory_info().rss / 2**20)
                    next_mem = t_out + 1000.0
                if on_stats is not None and t_out >= next_stats:
                    on_stats(self._report(frames, t0, mem, rss_start, with_face, with_hands))
                    next_stats = t_out + stats_every_s * 1000
        finally:
            self.source.stop()
        return self._report(frames, t0, mem, rss_start, with_face, with_hands)

    def _report(
        self, frames: int, t0: float, mem: MemoryTrend, rss_start: float, face: int, hands: int
    ) -> PipelineReport:
        dur = (now_ms() - t0) / 1000
        n = max(frames, 1)
        return PipelineReport(
            frames=frames,
            duration_s=dur,
            fps=frames / dur if dur > 0 else 0.0,
            dropped_frames=self.source.dropped,
            latency_ms=self.latency.summary(),
            stages_ms={k: v.summary() for k, v in self.stages.items()},
            hands_detected_pct=100.0 * hands / n,
            face_detected_pct=100.0 * face / n,
            rss_start_mb=rss_start,
            rss_peak_mb=mem.peak_mb,
            rss_slope_mb_per_min=mem.slope_mb_per_min,
        )
