"""Video frame sources: webcam and video file (mock input).

Both sources run a background thread that always keeps only the **newest** frame. If the rest
of the pipeline is slower than the camera, old frames are dropped, never queued, so delay can't
build up over a long stream.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from common.config import VideoConfig
from common.timing import now_ms

log = logging.getLogger(__name__)

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Frame:
    image: Image
    """BGR, HxWx3, uint8."""
    index: int
    t_capture_ms: float
    """When the frame became available to the engine (monotonic clock, see common.timing)."""


class LatestSlot:
    """Thread-safe single-frame mailbox. ``put`` overwrites; ``get`` waits for a newer frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            if self._frame is not None:
                self.dropped += 1  # the previous frame was never consumed
            self._frame = frame
            self._cond.notify_all()

    def get(self, timeout_s: float | None = None) -> Frame | None:
        """Take the newest frame, or ``None`` on timeout / when closed and empty."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None or self._closed, timeout_s):
                return None
            frame, self._frame = self._frame, None
            return frame

    @property
    def has_frame(self) -> bool:
        return self._frame is not None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed


class FrameSource(ABC):
    """Base class: subclasses implement ``_run`` and push frames with ``_publish``."""

    def __init__(self) -> None:
        self.slot = LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self.error: BaseException | None = None

    def start(self) -> FrameSource:
        self._thread = threading.Thread(target=self._guarded_run, name=type(self).__name__)
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.slot.close()

    def read(self, timeout_s: float = 1.0) -> Frame | None:
        return self.slot.get(timeout_s)

    @property
    def dropped(self) -> int:
        return self.slot.dropped

    def _publish(self, image: Image) -> None:
        self.slot.put(Frame(image, self._index, now_ms()))
        self._index += 1

    def _guarded_run(self) -> None:
        try:
            self._run()
        except BaseException as e:  # surfaced to the pipeline instead of dying silently
            log.exception("frame source failed")
            self.error = e
        finally:
            self.slot.close()

    @abstractmethod
    def _run(self) -> None: ...

    def __enter__(self) -> FrameSource:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()


class FileSource(FrameSource):
    """Plays a video file as if it were a live camera.

    ``realtime=True`` paces frames at the file's fps (for latency tests).
    ``realtime=False`` decodes as fast as possible but still waits until the previous frame
    has been taken, so no frame is skipped (for throughput benchmarks).
    """

    def __init__(self, path: Path, *, loop: bool = True, realtime: bool = True) -> None:
        super().__init__()
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.fps = 30.0
        self.size = (0, 0)
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()

    def _run(self) -> None:
        import cv2

        period_s = 1.0 / self.fps
        cap = cv2.VideoCapture(str(self.path))
        try:
            next_t = time.perf_counter()
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    if not self.loop:
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.realtime:
                    next_t += period_s
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    elif delay < -0.5:  # fell far behind (debugger, suspend): resync clock
                        next_t = time.perf_counter()
                else:
                    while self.slot.has_frame and not self._stop.is_set():
                        time.sleep(0.0005)
                self._publish(np.asarray(img, dtype=np.uint8))
        finally:
            cap.release()


class CameraSource(FrameSource):
    """Webcam via OpenCV. On Windows uses Media Foundation, falls back to DirectShow."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self._cap = self._open()

    def _open(self) -> Any:
        import cv2

        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(self.index, be)
            if not cap.isOpened():
                continue
            # MJPG lets most USB webcams deliver 720p/1080p at 30 fps (raw YUY2 is often 5-10 fps).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let the driver queue stale frames
            ok, _ = cap.read()
            if ok:
                log.info(
                    "camera %d opened (backend %s): %dx%d @ %.0f fps",
                    self.index,
                    cap.getBackendName(),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    cap.get(cv2.CAP_PROP_FPS),
                )
                return cap
            cap.release()
        raise RuntimeError(f"cannot open camera {self.index}")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, img = self._cap.read()
                if not ok:
                    raise RuntimeError("camera stopped delivering frames (unplugged?)")
                self._publish(np.asarray(img, dtype=np.uint8))
        finally:
            self._cap.release()


def open_source(cfg: VideoConfig, *, realtime: bool = True) -> FrameSource:
    if cfg.source == "file":
        assert cfg.file_path is not None  # guaranteed by VideoConfig validation
        return FileSource(cfg.file_path, loop=cfg.loop_file, realtime=realtime)
    return CameraSource(cfg.camera_index, cfg.capture_width, cfg.capture_height, cfg.fps)
