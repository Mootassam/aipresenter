"""Real-time LivePortrait renderer: camera frame + face tracking -> animated character frame.

Per frame (GPU unless noted):
  1. driving crop around the user's face from MediaPipe landmarks (CPU, 256x256)
  2. motion extractor M -> pose / expression keypoints
  3. One Euro smoothing of the ~70 motion parameters (CPU, tiny)
  4. relative motion vs. the neutral calibration frame + stitching
  5. warping module W + generator G -> 512x512 animated face
  6. paste back into the character canvas with a soft mask, optional background swap
  7. single GPU -> CPU copy of the finished frame
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from common.config import AvatarConfig
from common.timing import now_ms
from engine.avatar.character import CharacterSource
from engine.avatar.liveportrait.crop import DrivingCropper, eye_open_ratio, landmarks_px
from engine.avatar.liveportrait.motion import (
    KpInfo,
    blend_toward,
    relative_keypoints,
    transform_keypoints,
)
from engine.avatar.liveportrait.network import LivePortraitNet
from engine.io.frames import Frame, Image
from engine.tracking.one_euro import OneEuroFilter
from engine.tracking.types import TrackingResult

log = logging.getLogger(__name__)

_MASK = Path(__file__).with_name("mask_template.png")
_POSE_PARAMS = 7  # pitch, yaw, roll, tx, ty, tz, scale


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def _cover_resize(img: Image, w: int, h: int) -> Image:
    """Scale and centre-crop ``img`` to exactly fill w x h."""
    import cv2

    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (round(iw * s), round(ih * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return np.ascontiguousarray(r[y0 : y0 + h, x0 : x0 + w])


class LivePortraitRenderer:
    def __init__(
        self,
        net: LivePortraitNet,
        character: CharacterSource,
        cfg: AvatarConfig,
        *,
        background: Image | None = None,
        profile: bool = False,
    ) -> None:
        import cv2

        self.net, self.cfg, self.profile = net, cfg, profile
        dev = net.device
        self.char = character
        h, w = character.canvas.shape[:2]
        self.size = (w, h)

        # --- character (source) side, computed once ---
        m = character.crop.m_o2c[:2]
        crop512 = cv2.warpAffine(
            character.canvas, m, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        small = cv2.resize(crop512, (256, 256), interpolation=cv2.INTER_AREA)
        src256 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        x = net.to_input(torch.from_numpy(src256))
        self.f_s = net.appearance(x)
        self.src = net.motion(x)
        self.x_s = transform_keypoints(self.src)

        # Canvas (RGB float, 1x3xHxW) and the paste-back sampling grid: for every canvas pixel,
        # where it lies in the 512 crop. Fixed because the character crop never moves.
        canvas_rgb = cv2.cvtColor(character.canvas, cv2.COLOR_BGR2RGB)
        self.canvas = torch.from_numpy(canvas_rgb).to(dev).permute(2, 0, 1)[None].float() / 255
        mt = torch.from_numpy(character.crop.m_o2c).to(dev)
        ys, xs = torch.meshgrid(
            torch.arange(h, device=dev, dtype=torch.float32),
            torch.arange(w, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        cx = mt[0, 0] * xs + mt[0, 1] * ys + mt[0, 2]
        cy = mt[1, 0] * xs + mt[1, 1] * ys + mt[1, 2]
        self.grid = torch.stack([(cx + 0.5) / 256 - 1, (cy + 0.5) / 256 - 1], -1)[None]
        mask = cv2.imread(str(_MASK), cv2.IMREAD_GRAYSCALE)
        mask_t = torch.from_numpy(mask).to(dev).float()[None, None] / 255
        self.mask = F.grid_sample(mask_t, self.grid, align_corners=False, padding_mode="zeros")

        # Background: alpha from the character matte, replacement from colour or image
        self.alpha: torch.Tensor | None = None
        self.bg: torch.Tensor | None = None
        if cfg.background != "original":
            if character.alpha is None:
                raise ValueError("background replacement needs a character matte")
            self.alpha = torch.from_numpy(character.alpha).to(dev)[None, None]
            if cfg.background == "image":
                if background is None:
                    raise ValueError("background image missing")
                bg = cv2.cvtColor(_cover_resize(background, w, h), cv2.COLOR_BGR2RGB)
            else:
                bg = np.empty((h, w, 3), np.uint8)
                bg[:] = _hex_to_bgr(cfg.background_color)[::-1]
            self.bg = torch.from_numpy(bg).to(dev).permute(2, 0, 1)[None].float() / 255

        # --- driving side state ---
        self.cropper = DrivingCropper()
        self.f_pose = OneEuroFilter(cfg.pose_smoothing)
        self.f_exp = OneEuroFilter(cfg.expression_smoothing)
        self.drv0: KpInfo | None = None
        self._last_params: npt.NDArray[np.float32] | None = None
        self._last_xd: torch.Tensor | None = None
        self._lost_since: float | None = None
        self._recalibrate = True
        self.timings_ms: dict[str, float] = {}
        log.info("character ready: canvas %dx%d", w, h)

    # -- public ------------------------------------------------------------------------------

    def recalibrate(self) -> None:
        """Use the next frame as the neutral pose (user looks straight at the camera)."""
        self._recalibrate = True

    def render(self, frame: Frame, track: TrackingResult, hud: list[str] | None = None) -> Image:
        t0 = now_ms()
        self.timings_ms = {}
        x_d = self._driving_keypoints(frame, track)
        self._mark("motion", t0)

        if x_d is None:
            out = self.canvas if self.bg is None else self._swap_bg(self.canvas)
        else:
            face = self.net.warp_decode(self.f_s, self.x_s, x_d)
            self._mark("warp_decode", t0)
            pasted = F.grid_sample(face, self.grid, align_corners=False, padding_mode="zeros")
            out = self.mask * pasted + (1 - self.mask) * self.canvas
            if self.bg is not None:
                out = self._swap_bg(out)

        img = self._to_bgr_u8(out)
        self._mark("compose", t0)
        if hud:
            _draw_hud(img, hud)
        return img

    # -- internals ---------------------------------------------------------------------------

    def _mark(self, name: str, t0: float) -> None:
        if self.profile and self.net.device.type == "cuda":
            torch.cuda.synchronize(self.net.device)
        self.timings_ms[name] = now_ms() - t0 - sum(self.timings_ms.values())

    def _swap_bg(self, img: torch.Tensor) -> torch.Tensor:
        assert self.alpha is not None and self.bg is not None
        return self.alpha * img + (1 - self.alpha) * self.bg

    def _to_bgr_u8(self, img: torch.Tensor) -> Image:
        u8 = (img[0].flip(0) * 255 + 0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0)
        return np.ascontiguousarray(u8.cpu().numpy())

    def _driving_keypoints(self, frame: Frame, track: TrackingResult) -> torch.Tensor | None:
        import cv2

        if track.face is None:
            return self._face_lost(frame.t_capture_ms)
        self._lost_since = None

        pts = landmarks_px(track.face.landmarks, track.width, track.height)
        crop, moved = self.cropper.update(pts)
        img = cv2.warpAffine(frame.image, crop.m_o2c[:2], (256, 256), flags=cv2.INTER_AREA,
                             borderMode=cv2.BORDER_REPLICATE)  # fmt: skip
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        info = self.net.motion(self.net.to_input(torch.from_numpy(rgb)))
        raw = info.to_params()

        if moved and self.drv0 is not None and self._last_params is not None:
            # The crop box jumped: shift the calibration by the same amount so the character
            # doesn't jump. Pose and expression don't depend on the box; t and scale do.
            d0 = self.drv0.to_params()
            d0[3:6] += raw[3:6] - self._last_params[3:6]
            d0[6] *= raw[6] / max(self._last_params[6], 1e-6)
            self.drv0 = self.drv0.with_params(d0)
            self.f_pose.reset()
            self.f_exp.reset()
        self._last_params = raw

        t = frame.t_capture_ms
        p = raw.copy()
        p[:_POSE_PARAMS] = self.f_pose(raw[:_POSE_PARAMS], t)
        p[_POSE_PARAMS:] = self.f_exp(raw[_POSE_PARAMS:], t)
        drv = info.with_params(p)

        if self._recalibrate or self.drv0 is None:
            self.drv0 = drv
            self._recalibrate = False
            log.info("calibrated neutral pose")

        if self.cfg.expression_strength != 1.0:
            k = self.cfg.expression_strength
            exp0 = self.drv0.to_params()[_POSE_PARAMS:]
            scaled = exp0 + (p[_POSE_PARAMS:] - exp0) * k
            drv = info.with_params(np.concatenate([p[:_POSE_PARAMS], scaled]))

        x_d = relative_keypoints(self.src, drv, self.drv0, pose_strength=self.cfg.pose_strength)
        if self.cfg.stitching:
            x_d = self.net.stitch(self.x_s, x_d)
        if self.cfg.eye_retarget:
            x_d = x_d + self._eye_delta(pts)
        self._last_xd = x_d
        return x_d

    def _eye_delta(self, pts: npt.NDArray[np.float32]) -> torch.Tensor:
        """Extra eye opening/closing from MediaPipe, relative to the character's own eyes."""
        src = self.char.eye_ratio
        cur = float(eye_open_ratio(pts).mean())
        ratio = torch.tensor([[src[0], src[1], cur]], device=self.net.device)
        neutral = torch.tensor([[src[0], src[1], float(src.mean())]], device=self.net.device)
        return self.net.retarget_eye(self.x_s, ratio) - self.net.retarget_eye(self.x_s, neutral)

    def _face_lost(self, t_ms: float) -> torch.Tensor | None:
        """Ease the last pose back to neutral instead of freezing or snapping."""
        if self._last_xd is None:
            return None
        if self._lost_since is None:
            self._lost_since = t_ms
        ease = self.cfg.lost_face_ease_s * 1000
        k = 1.0 if ease <= 0 else min(1.0, (t_ms - self._lost_since) / ease)
        return blend_toward(self._last_xd, self.x_s, k)


def _draw_hud(img: Image, lines: list[str]) -> None:
    import cv2

    for i, line in enumerate(lines):
        y = 22 * (i + 1)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
