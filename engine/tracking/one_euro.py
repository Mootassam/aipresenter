"""Vectorised One Euro filter (Casiez et al., CHI 2012).

Smooths jitter when the signal is slow and follows quickly when it moves fast: the cutoff
frequency rises with speed. One instance filters a whole landmark array (e.g. 21x3 hand) at once.

Units: our landmarks are normalised image coordinates (0..1), so speeds are a few units/s and
``beta`` is larger than in the paper's pixel-based examples.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from common.config import OneEuroConfig

FloatArray = npt.NDArray[np.float32]


def _alpha(dt_s: float, cutoff: FloatArray | float) -> FloatArray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroFilter:
    def __init__(self, cfg: OneEuroConfig) -> None:
        self.cfg = cfg
        self._x: FloatArray | None = None
        self._dx: FloatArray | None = None
        self._t_ms = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = None

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def __call__(self, x: FloatArray, t_ms: float) -> FloatArray:
        x = np.asarray(x, dtype=np.float32)
        if self._x is None or self._dx is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t_ms = t_ms
            return x.copy()

        dt_s = (t_ms - self._t_ms) / 1000.0
        if dt_s <= 0.0:
            return self._x.copy()  # same timestamp twice: keep previous output
        self._t_ms = t_ms

        dx = (x - self._x) / dt_s
        a_d = _alpha(dt_s, self.cfg.d_cutoff)
        self._dx = (a_d * dx + (1.0 - a_d) * self._dx).astype(np.float32)

        cutoff = (self.cfg.min_cutoff + self.cfg.beta * np.abs(self._dx)).astype(np.float32)
        a = _alpha(dt_s, cutoff)
        self._x = (a * x + (1.0 - a) * self._x).astype(np.float32)
        return self._x.copy()
