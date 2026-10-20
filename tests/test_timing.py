from __future__ import annotations

import pytest

from common.timing import FpsMeter, LatencyStats, Stopwatch


def test_fps_meter_steady_30fps() -> None:
    m = FpsMeter(window_s=1.0)
    for i in range(90):
        m.tick(i * 1000.0 / 30.0)
    assert m.fps == pytest.approx(30.0, rel=0.02)


def test_fps_meter_window_drops_old_frames() -> None:
    m = FpsMeter(window_s=1.0)
    for i in range(30):  # 10 fps for 3 s
        m.tick(i * 100.0)
    for i in range(30):  # then 30 fps for 1 s
        m.tick(3000.0 + i * 1000.0 / 30.0)
    assert m.fps == pytest.approx(30.0, rel=0.05)


def test_fps_meter_empty() -> None:
    assert FpsMeter().fps == 0.0


def test_latency_stats_percentiles() -> None:
    s = LatencyStats(capacity=1000)
    for v in range(1, 101):
        s.add(float(v))
    summ = s.summary()
    assert summ.count == 100
    assert summ.p50 == pytest.approx(50.5)
    assert summ.max == 100.0
    assert summ.p95 == pytest.approx(95.05)


def test_latency_stats_ring_buffer_bounded() -> None:
    s = LatencyStats(capacity=10)
    for v in range(1000):
        s.add(float(v))
    assert len(s) == 10
    assert s.summary().mean == pytest.approx(994.5)  # only the last 10 samples


def test_latency_stats_empty_and_invalid() -> None:
    assert LatencyStats().summary().count == 0
    with pytest.raises(ValueError):
        LatencyStats(capacity=0)


def test_stopwatch_records() -> None:
    s = LatencyStats()
    with Stopwatch(s) as sw:
        sum(range(1000))
    assert len(s) == 1
    assert sw.elapsed_ms >= 0.0
