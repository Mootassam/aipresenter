"""Lip-sync hook: voice -> mouth openness, mailbox and file track (no GPU, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.avatar.mouth import (
    HOP,
    SAMPLE_RATE,
    FileMouthTrack,
    MouthMailbox,
    VoiceMouthMeter,
    lip_target,
)
from engine.io.frames import Frame

_IMG = np.zeros((2, 2, 3), np.uint8)


def _tone(seconds: float, db: float, hz: float = 800.0) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (10 ** (db / 20) * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (10 ** (db / 20) * rng.standard_normal(round(seconds * SAMPLE_RATE))).astype(np.float32)


def _hops(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE / HOP)


def test_room_noise_keeps_mouth_closed() -> None:
    out = VoiceMouthMeter().push(np.concatenate([np.zeros(SAMPLE_RATE, np.float32),
                                                 _noise(3, -50)]))  # fmt: skip
    assert out.max() < 0.1


def test_syllable_opens_fast_and_closes_after() -> None:
    m = VoiceMouthMeter()
    m.push(_noise(2, -50))
    speech = m.push(_tone(0.3, -10) + _noise(0.3, -50, 1))
    after = m.push(_noise(0.5, -50, 2))
    assert speech[_hops(0.05)] > 0.7  # open within 50 ms
    assert after[-1] < 0.05


def test_leading_digital_silence_does_not_jam_mouth_open() -> None:
    """A stream starting with exact zeros used to pin the noise floor at -100 dB for seconds."""
    m = VoiceMouthMeter()
    m.push(np.zeros(SAMPLE_RATE // 2, np.float32))
    pause = m.push(_noise(1, -40))
    assert pause[_hops(0.2) :].max() < 0.1


def test_quieter_voice_still_opens_mouth() -> None:
    m = VoiceMouthMeter()
    m.push(_noise(1, -60))
    m.push(_tone(1, -6))
    m.push(_noise(1, -60, 1))
    quiet = m.push(np.tile(np.concatenate([_tone(0.2, -20), _noise(0.2, -60, 2)]), 15))
    assert quiet[-_hops(0.4) :].max() > 0.5  # after a few seconds the scale has adapted


def test_chunk_size_does_not_matter() -> None:
    x = np.concatenate([_noise(0.5, -50), _tone(0.4, -12), _noise(0.5, -50, 1)])
    whole = VoiceMouthMeter().push(x)
    m = VoiceMouthMeter()
    parts = [m.push(c) for c in np.array_split(x, [37, 500, 501, 9000])]
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-6)


def test_lip_target_opens_more_than_it_closes() -> None:
    assert lip_target(0.1, 1.0, 0.0, 0.4) == pytest.approx(0.1)
    assert lip_target(0.1, 1.0, 1.0, 0.4) == pytest.approx(0.4)
    assert lip_target(0.1, 0.5, 0.5, 0.4) == pytest.approx(0.15)
    # silence pulls an open mouth only half as far toward closed
    assert lip_target(0.4, 0.0, 1.0, 0.4) == pytest.approx(0.2)


def test_mailbox_forgets_old_values() -> None:
    box = MouthMailbox(max_age_ms=100)
    assert box.openness(Frame(_IMG, 0, 1000.0)) is None
    box.publish(0.7, 1000.0)
    assert box.openness(Frame(_IMG, 1, 1050.0)) == pytest.approx(0.7)
    assert box.openness(Frame(_IMG, 2, 1200.0)) is None


def test_file_track_follows_media_time_and_loops() -> None:
    vals = np.arange(100, dtype=np.float32)  # 1 s of hops
    track = FileMouthTrack(vals, fps=10.0, n_frames=10)
    assert track.openness(Frame(_IMG, 3, 0.0)) == 30.0  # frame 3 = 300 ms = hop 30
    assert track.openness(Frame(_IMG, 13, 0.0)) == 30.0  # the file looped
    late = FileMouthTrack(vals, fps=10.0, n_frames=10, offset_ms=50)
    assert late.openness(Frame(_IMG, 3, 0.0)) == 35.0
    assert FileMouthTrack(vals, 10.0, 10, offset_ms=-900).openness(Frame(_IMG, 0, 0.0)) == 0.0
