"""Lip-sync hook: how open the mouth should be, from the voice.

The audio engine (Phase 2) turns the *converted* voice into a mouth openness value (0 closed,
1 wide open) about 100 times a second and publishes it to a ``MouthMailbox``. The avatar reads
the newest value for each video frame and blends it with the tracked mouth, so the lips match
the voice the viewers hear, even when the camera sees the mouth badly (dark room, beard,
microphone in front of the face).

``VoiceMouthMeter`` is the voice -> openness step. ``FileMouthTrack`` runs it over the audio of
a video file, so the hook can be tested with ``aip track --file`` before the voice engine exists.
"""

from __future__ import annotations

import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from engine.io.frames import Frame

F32 = npt.NDArray[np.float32]

SAMPLE_RATE = 16_000
HOP = 160  # 10 ms
WINDOW = 320  # 20 ms
SPEECH_BAND_HZ = (250.0, 3500.0)
_DIGITAL_SILENCE_DB = -90.0


CLOSE_WEIGHT = 0.5
"""Silence closes a tracked open mouth only half as strongly as speech opens a closed one:
an open-mouthed laugh or surprise has no sound, and it shouldn't be pressed shut."""


def lip_target(tracked: float, voice: float, strength: float, max_ratio: float) -> float:
    """Lip opening ratio to render: the tracked one pulled toward what the voice implies."""
    want = voice * max_ratio
    k = strength if want > tracked else strength * CLOSE_WEIGHT
    return tracked + k * (want - tracked)


class MouthSignal(Protocol):
    def openness(self, frame: Frame) -> float | None:
        """Mouth openness 0-1 for this video frame, or ``None`` when there is no voice signal."""
        ...


@dataclass(frozen=True)
class MeterConfig:
    gate_db: float = 6.0
    """Sound must be this far above the noise floor to open the mouth at all."""
    range_db: float = 24.0
    """Loudness span from closed to fully open (shrinks if the voice is quieter)."""
    floor_rise_db_s: float = 3.0
    """How fast the noise floor creeps up (it drops at once when the room gets quieter)."""
    peak_fall_db_s: float = 4.0
    """How fast the remembered loudest level fades, so a quieter voice still opens the mouth."""
    attack_ms: float = 15.0
    release_ms: float = 60.0


class VoiceMouthMeter:
    """Voice audio (mono float32, 16 kHz) -> mouth openness 0-1, one value per 10 ms.

    Speech-band loudness in dB, placed between an adaptive noise floor and the recent peak, so
    it works with any microphone gain without setup. Fast attack, slower release: lips open on
    a syllable at once and close a little more gently, like a real mouth.
    """

    def __init__(self, cfg: MeterConfig | None = None) -> None:
        self.cfg = cfg or MeterConfig()
        freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
        self._band = (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])
        self._win = np.hanning(WINDOW).astype(np.float32)
        self._buf = np.zeros(WINDOW, np.float32)
        self._pending: F32 = np.zeros(0, np.float32)
        dt = HOP / SAMPLE_RATE
        self._a_up = 1 - math.exp(-dt * 1000 / self.cfg.attack_ms)
        self._a_down = 1 - math.exp(-dt * 1000 / self.cfg.release_ms)
        self._floor_step = self.cfg.floor_rise_db_s * dt
        self._peak_step = self.cfg.peak_fall_db_s * dt
        self._floor: float | None = None
        self._peak = -120.0
        self.value = 0.0

    def push(self, samples: F32) -> F32:
        """Feed any number of samples; returns one openness value per completed 10 ms hop."""
        x: F32 = np.concatenate([self._pending, np.asarray(samples, np.float32).ravel()])
        n = len(x) // HOP
        self._pending = x[n * HOP :]
        out = np.empty(n, np.float32)
        for i in range(n):
            self._buf[:-HOP] = self._buf[HOP:]
            self._buf[-HOP:] = x[i * HOP : (i + 1) * HOP]
            out[i] = self._step(self._band_db())
        return out

    def _band_db(self) -> float:
        spec = np.abs(np.fft.rfft(self._buf * self._win)) ** 2
        return 10 * math.log10(float(spec[self._band].sum()) + 1e-10)

    def _step(self, db: float) -> float:
        c = self.cfg
        if db > _DIGITAL_SILENCE_DB:  # exact zeros (stream start, muted mic) aren't room noise
            self._floor = db if self._floor is None else min(db, self._floor + self._floor_step)
        floor = db if self._floor is None else self._floor
        self._peak = max(db, self._peak - self._peak_step, floor + c.gate_db + 6)
        # Closed at the gate or `range_db` under the recent peak, whichever is higher: speech
        # sits far above room noise, so a floor-based scale alone would leave the mouth wide open.
        lo = max(floor + c.gate_db, self._peak - c.range_db)
        target = float(np.clip((db - lo) / max(self._peak - lo, 1e-3), 0.0, 1.0))
        a = self._a_up if target > self.value else self._a_down
        self.value += a * (target - self.value)
        return self.value


class MouthMailbox:
    """Newest openness from the audio engine; old values count as silence-unknown (``None``)."""

    def __init__(self, max_age_ms: float = 150.0) -> None:
        self.max_age_ms = max_age_ms
        self._lock = threading.Lock()
        self._value: float | None = None
        self._t_ms = -math.inf

    def publish(self, value: float, t_ms: float) -> None:
        with self._lock:
            self._value, self._t_ms = float(value), t_ms

    def openness(self, frame: Frame) -> float | None:
        with self._lock:
            if frame.t_capture_ms - self._t_ms > self.max_age_ms:
                return None
            return self._value


class FileMouthTrack:
    """Openness for every 10 ms of a video file's audio, looked up by the frame's media time."""

    def __init__(self, values: F32, fps: float, n_frames: int, offset_ms: float = 0.0) -> None:
        self.values, self.fps, self.n_frames, self.offset_ms = values, fps, n_frames, offset_ms

    @classmethod
    def from_video(
        cls, path: Path, fps: float, n_frames: int, *, offset_ms: float = 0.0
    ) -> FileMouthTrack:
        meter = VoiceMouthMeter()
        return cls(meter.push(read_audio(path)), fps, n_frames, offset_ms)

    def openness(self, frame: Frame) -> float | None:
        if len(self.values) == 0:
            return None
        media_ms = (frame.index % max(self.n_frames, 1)) / self.fps * 1000 + self.offset_ms
        i = round(media_ms / (HOP / SAMPLE_RATE * 1000))
        return float(self.values[min(max(i, 0), len(self.values) - 1)])


def read_audio(path: Path) -> F32:
    """Mono 16 kHz float32 audio of a media file via ffmpeg (dev tool: not shipped)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
           "-f", "f32le", "-"]  # fmt: skip
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found: needed to read the audio of a video file") from None
    return np.frombuffer(raw, np.float32).copy()
