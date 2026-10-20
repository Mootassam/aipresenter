"""Small, allocation-free helpers for measuring fps and latency in the real-time loops.

They are used by every pipeline stage and by the benchmark script, so per-component
numbers are measured the same way everywhere.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType

import numpy as np


def now_ms() -> float:
    """Monotonic high-resolution clock in milliseconds."""
    return time.perf_counter_ns() / 1e6


class FpsMeter:
    """Frames per second over a sliding time window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_ms = window_s * 1000.0
        self._stamps: deque[float] = deque()

    def tick(self, t_ms: float | None = None) -> None:
        t = now_ms() if t_ms is None else t_ms
        self._stamps.append(t)
        cutoff = t - self._window_ms
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.popleft()

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return 0.0 if span <= 0 else (len(self._stamps) - 1) * 1000.0 / span


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float

    def __str__(self) -> str:
        return (
            f"n={self.count} mean={self.mean:.1f} p50={self.p50:.1f} "
            f"p95={self.p95:.1f} p99={self.p99:.1f} max={self.max:.1f} ms"
        )


class LatencyStats:
    """Fixed-size ring buffer of samples (ms) with percentile summary. Memory never grows."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._n = 0
        self._i = 0

    def add(self, ms: float) -> None:
        self._buf[self._i] = ms
        self._i = (self._i + 1) % self._buf.size
        self._n = min(self._n + 1, self._buf.size)

    def __len__(self) -> int:
        return self._n

    def summary(self) -> LatencySummary:
        if self._n == 0:
            return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        data = self._buf[: self._n]
        p50, p95, p99 = np.percentile(data, [50, 95, 99])
        return LatencySummary(
            self._n, float(data.mean()), float(p50), float(p95), float(p99), float(data.max())
        )


class Stopwatch:
    """``with Stopwatch(stats): ...`` records the block's duration into ``stats``."""

    def __init__(self, stats: LatencyStats | None = None) -> None:
        self._stats = stats
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._t0 = now_ms()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = now_ms() - self._t0
        if self._stats is not None:
            self._stats.add(self.elapsed_ms)
