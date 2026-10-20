from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from engine.io.frames import FileSource, Frame, LatestSlot

cv2 = pytest.importorskip("cv2")


def _frame(i: int) -> Frame:
    return Frame(np.zeros((2, 2, 3), np.uint8), i, float(i))


def test_latest_slot_keeps_newest_and_counts_drops() -> None:
    slot = LatestSlot()
    for i in range(5):
        slot.put(_frame(i))
    got = slot.get(0.1)
    assert got is not None and got.index == 4
    assert slot.dropped == 4
    assert slot.get(0.01) is None  # empty after taking


def test_latest_slot_wakes_waiting_reader() -> None:
    slot = LatestSlot()
    threading.Timer(0.05, lambda: slot.put(_frame(7))).start()
    got = slot.get(1.0)
    assert got is not None and got.index == 7


def test_latest_slot_close_unblocks() -> None:
    slot = LatestSlot()
    threading.Timer(0.05, slot.close).start()
    t0 = time.perf_counter()
    assert slot.get(2.0) is None
    assert time.perf_counter() - t0 < 1.0
    assert slot.closed


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(20):
        img = np.full((48, 64, 3), i * 10, np.uint8)
        w.write(img)
    w.release()
    return path


def test_file_source_fast_mode_delivers_every_frame(video: Path) -> None:
    with FileSource(video, loop=False, realtime=False) as src:
        frames = []
        while (f := src.read(2.0)) is not None:
            frames.append(f)
    assert [f.index for f in frames] == list(range(20))
    assert src.size == (64, 48)


def test_file_source_realtime_is_paced(video: Path) -> None:
    t0 = time.perf_counter()
    with FileSource(video, loop=False, realtime=True) as src:
        while src.read(2.0) is not None:
            pass
    assert time.perf_counter() - t0 >= 0.5  # 20 frames at 30 fps ~ 0.67 s


def test_file_source_loops(video: Path) -> None:
    with FileSource(video, loop=True, realtime=False) as src:
        indices = [src.read(2.0) for _ in range(30)]
    assert all(f is not None for f in indices)


def test_file_source_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.mp4")
