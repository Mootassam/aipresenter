"""Video outputs: virtual camera (softcam), preview window, and a null sink.

softcam (MIT, https://github.com/tshino/softcam) is a DirectShow virtual camera DLL. It shows
up as "DirectShow Softcam" in OBS, Zoom, Discord, Chrome and others. It must be registered
once with admin rights (``softcam_installer.exe register softcam.dll``).
We use it instead of pyvirtualcam, which is GPL-2.0.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from engine.io.frames import Image

log = logging.getLogger(__name__)


def letterbox(img: Image, width: int, height: int) -> Image:
    """Fit ``img`` inside width x height keeping aspect ratio, padding with black."""
    import cv2

    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


class VideoSink(Protocol):
    width: int
    height: int

    def send(self, img: Image) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.frames = 0

    def send(self, img: Image) -> None:
        self.frames += 1

    def close(self) -> None:
        pass


class PreviewWindow:
    """OpenCV window for development. Pressing Q or Esc sets ``quit_requested``."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str = "AI Presenter preview",
        on_key: Callable[[str], None] | None = None,
    ) -> None:
        self.width, self.height, self.title = width, height, title
        self.on_key = on_key
        self.quit_requested = False

    def send(self, img: Image) -> None:
        import cv2

        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            self.quit_requested = True
        elif key != 255 and self.on_key is not None:
            self.on_key(chr(key))

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.title)


class RecorderSink:
    """Writes the output to an .mp4 file (for reviewing quality, e.g. from a RunPod test)."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        import cv2

        self.width, self.height, self.path = width, height, path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
        if not self._w.isOpened():
            raise RuntimeError(f"cannot write video {path}")

    def send(self, img: Image) -> None:
        self._w.write(img)

    def close(self) -> None:
        self._w.release()


class SoftcamNotAvailable(RuntimeError):
    pass


def find_softcam_dll(override: Path | None, data_dir: Path) -> Path | None:
    repo = Path(__file__).resolve().parents[2]
    candidates = [
        override,
        data_dir / "bin" / "softcam.dll",
        repo / "third_party" / "softcam" / "softcam.dll",
    ]
    return next((p for p in candidates if p is not None and p.exists()), None)


class SoftcamSink:
    """Sends frames to the softcam virtual camera. Frames must be BGR uint8, exact size."""

    def __init__(self, dll_path: Path, width: int, height: int, fps: float) -> None:
        if sys.platform != "win32":
            raise SoftcamNotAvailable("softcam is Windows-only")
        self.width, self.height = width, height
        self._dll = ctypes.CDLL(str(dll_path))
        self._dll.scCreateCamera.restype = ctypes.c_void_p
        self._dll.scCreateCamera.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self._dll.scSendFrame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._dll.scSendFrame.restype = None
        self._dll.scDeleteCamera.argtypes = [ctypes.c_void_p]
        self._dll.scDeleteCamera.restype = None
        self._dll.scIsConnected.argtypes = [ctypes.c_void_p]
        self._dll.scIsConnected.restype = ctypes.c_bool
        self._cam = self._dll.scCreateCamera(width, height, float(fps))
        if not self._cam:
            raise SoftcamNotAvailable(
                "scCreateCamera failed (another app may already be using the virtual camera)"
            )
        log.info("virtual camera started: %dx%d @ %.0f fps", width, height, fps)

    @property
    def connected(self) -> bool:
        """True while some app (OBS, Zoom...) is reading the virtual camera."""
        return bool(self._cam) and bool(self._dll.scIsConnected(self._cam))

    def send(self, img: Image) -> None:
        if img.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be {self.width}x{self.height}, got {img.shape}")
        buf = np.ascontiguousarray(img)
        self._dll.scSendFrame(self._cam, buf.ctypes.data)

    def close(self) -> None:
        if self._cam:
            self._dll.scDeleteCamera(self._cam)
            self._cam = None
