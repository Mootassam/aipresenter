from __future__ import annotations

import numpy as np
import pytest

from common.config import OneEuroConfig
from engine.tracking.one_euro import OneEuroFilter

CFG = OneEuroConfig(min_cutoff=1.0, beta=5.0, d_cutoff=1.0)


def _run(f: OneEuroFilter, xs: np.ndarray, fps: float = 30.0) -> np.ndarray:
    return np.array([f(np.array([x], np.float32), i * 1000 / fps)[0] for i, x in enumerate(xs)])


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter(CFG)
    x = np.array([[0.2, 0.3, 0.0]], np.float32)
    assert np.array_equal(f(x, 0.0), x)
    assert f.initialized


def test_constant_signal_stays_constant() -> None:
    out = _run(OneEuroFilter(CFG), np.full(60, 0.5))
    assert np.allclose(out, 0.5)


def test_reduces_jitter() -> None:
    rng = np.random.default_rng(0)
    noisy = 0.5 + rng.normal(0, 0.005, 300)
    out = _run(OneEuroFilter(CFG), noisy)
    assert out[30:].std() < noisy[30:].std() * 0.5


def test_follows_fast_motion_quickly() -> None:
    # a hand sweeping across the frame at 1.5 screen widths / s
    t = np.arange(30) / 30.0
    xs = 0.1 + 1.5 * t
    out = _run(OneEuroFilter(CFG), xs)
    lag = xs[-1] - out[-1]
    assert lag < 0.06  # well under 2 frames of travel (2 * 0.05)


def test_beta_reduces_lag() -> None:
    xs = 0.1 + 1.5 * np.arange(30) / 30.0
    slow = _run(OneEuroFilter(OneEuroConfig(min_cutoff=1.0, beta=0.0)), xs)
    fast = _run(OneEuroFilter(CFG), xs)
    assert (xs[-1] - fast[-1]) < (xs[-1] - slow[-1])


def test_reset_and_shape_change() -> None:
    f = OneEuroFilter(CFG)
    f(np.zeros((21, 3), np.float32), 0.0)
    f.reset()
    assert not f.initialized
    out = f(np.ones((33, 3), np.float32), 10.0)  # new shape re-initialises
    assert out.shape == (33, 3)


def test_duplicate_timestamp_returns_previous() -> None:
    f = OneEuroFilter(CFG)
    f(np.array([0.0], np.float32), 0.0)
    a = f(np.array([1.0], np.float32), 33.0)
    b = f(np.array([5.0], np.float32), 33.0)
    assert b == pytest.approx(a)
