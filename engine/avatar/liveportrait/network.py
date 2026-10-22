"""Loads the five LivePortrait networks and runs them with the right precision.

F  appearance feature extractor  image(256) -> 3D feature volume      (once per character)
M  motion extractor              image(256) -> pose / exp / kp          (every frame)
W  warping module                feature + kp_src + kp_drv -> warped    (every frame)
G  SPADE generator               warped feature -> image(512)           (every frame)
S  stitching + eye/lip retargeting MLPs                                 (every frame, tiny)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from engine.avatar.liveportrait.motion import KpInfo
from engine.models import MODELS, ensure_model

log = logging.getLogger(__name__)

_CONFIG = Path(__file__).with_name("models.yaml")


def build_runs_on(major: int, minor: int, arch_list: list[str]) -> bool:
    """Can a PyTorch build compiled for `arch_list` run on a GPU of compute capability major.minor?

    CUDA binaries run on any GPU of the same major version with an equal or higher minor
    (sm_86 code runs on an RTX 4090, sm_89), and PTX (compute_XY) is JIT-compiled for any newer GPU.
    """
    for arch in arch_list:
        kind, _, num = arch.partition("_")
        if not num.isdigit() or len(num) < 2:
            continue
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def pick_device(prefer: str = "cuda") -> torch.device:
    """The requested CUDA device if this PyTorch build supports it, else the CPU."""
    if prefer.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(prefer)
        major, minor = torch.cuda.get_device_capability(dev)
        if build_runs_on(major, minor, torch.cuda.get_arch_list()):
            return dev
        log.warning(
            "%s (sm_%d%d) is not supported by this PyTorch build; using the CPU",
            torch.cuda.get_device_name(dev), major, minor,
        )  # fmt: skip
    return torch.device("cpu")


def supports_fast_fp16(device: torch.device) -> bool:
    """FP16 is only fast on Volta (7.0) and newer. Pascal (e.g. GTX 1050) runs it ~60x slower."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 7


def _strip_ddp(state: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("module."): v for k, v in state.items()}


class LivePortraitNet:
    def __init__(
        self,
        models_dir: Path,
        device: torch.device | None = None,
        fp16: bool | None = None,
        *,
        compile: bool = False,
    ) -> None:
        from engine.avatar.liveportrait.modules.appearance_feature_extractor import (
            AppearanceFeatureExtractor,
        )
        from engine.avatar.liveportrait.modules.motion_extractor import MotionExtractor
        from engine.avatar.liveportrait.modules.spade_generator import SPADEDecoder
        from engine.avatar.liveportrait.modules.stitching_retargeting_network import (
            StitchingRetargetingNetwork,
        )
        from engine.avatar.liveportrait.modules.warping_network import WarpingNetwork

        self.device = device or pick_device()
        self.fp16 = supports_fast_fp16(self.device) if fp16 is None else fp16
        # Weights stay fp32; fp16 runs through autocast like upstream LivePortrait, which keeps
        # precision-sensitive ops (softmax, gaussian heatmaps, grid_sample) in fp32.
        self.dtype = torch.float32
        params = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))["model_params"]

        def load(name: str, model: Any) -> Any:  # vendored modules are untyped
            path = ensure_model(name, models_dir)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            return model.to(self.device).eval().requires_grad_(False)

        self.F: Any = load(
            "lp_appearance_feature_extractor",
            AppearanceFeatureExtractor(**params["appearance_feature_extractor_params"]),
        )
        self.M: Any = load(
            "lp_motion_extractor", MotionExtractor(**params["motion_extractor_params"])
        )
        self.W: Any = load("lp_warping_module", WarpingNetwork(**params["warping_module_params"]))
        self.G: Any = load("lp_spade_generator", SPADEDecoder(**params["spade_generator_params"]))

        s_params = params["stitching_retargeting_module_params"]
        ckpt = torch.load(
            ensure_model("lp_stitching_retargeting", models_dir),
            map_location="cpu",
            weights_only=True,
        )
        self.S: dict[str, Any] = {}
        for key, ckpt_key in (
            ("stitching", "retarget_shoulder"),
            ("lip", "retarget_mouth"),
            ("eye", "retarget_eye"),
        ):
            net = StitchingRetargetingNetwork(**s_params[key])
            net.load_state_dict(_strip_ddp(ckpt[ckpt_key]))
            # The MLPs are tiny; keep them in fp32 for numerical stability.
            self.S[key] = net.to(self.device).eval().requires_grad_(False)

        if compile:
            # Fixed input shapes every frame, so compiled kernels are reused after warm-up.
            self.M = torch.compile(self.M, mode="max-autotune-no-cudagraphs")
            self.W = torch.compile(self.W, mode="max-autotune-no-cudagraphs")
            self.G = torch.compile(self.G, mode="max-autotune-no-cudagraphs")

        log.info(
            "LivePortrait loaded on %s (%s); weights: %s",
            self.device,
            "fp16" if self.fp16 else "fp32",
            ", ".join(MODELS[n].license for n in ("lp_motion_extractor",)),
        )

    # -- helpers -----------------------------------------------------------------------------

    def _amp(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16)

    def to_input(self, img_rgb_u8: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 RGB on any device -> (1, 3, H, W) model dtype in [0, 1]."""
        x = img_rgb_u8.to(self.device, non_blocking=True).permute(2, 0, 1)[None]
        return x.to(self.dtype) / 255.0

    @torch.inference_mode()
    def appearance(self, x: torch.Tensor) -> torch.Tensor:
        with self._amp():
            return self.F(x).float()

    @torch.inference_mode()
    def motion(self, x: torch.Tensor) -> KpInfo:
        with self._amp():
            raw = self.M(x)
        return KpInfo.from_raw({k: v.float() for k, v in raw.items()})

    @torch.inference_mode()
    def stitch(self, x_s: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        """Adjusts driving keypoints so the animated face blends into the static shoulders."""
        b, n = x_s.shape[:2]
        delta = self.S["stitching"](torch.cat([x_s.reshape(b, -1), x_d.reshape(b, -1)], 1))
        out = x_d + delta[:, : 3 * n].reshape(b, n, 3)
        out[..., :2] = out[..., :2] + delta[:, 3 * n : 3 * n + 2].reshape(b, 1, 2)
        return out

    @torch.inference_mode()
    def retarget_eye(self, x_s: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """ratio: (B, 3) = [src_left, src_right, target_open]. Returns keypoint delta."""
        b, n = x_s.shape[:2]
        return self.S["eye"](torch.cat([x_s.reshape(b, -1), ratio], 1)).reshape(b, n, 3)

    @torch.inference_mode()
    def retarget_lip(self, x_s: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """ratio: (B, 2) = [src_lip, target_lip]. Returns keypoint delta."""
        b, n = x_s.shape[:2]
        return self.S["lip"](torch.cat([x_s.reshape(b, -1), ratio], 1)).reshape(b, n, 3)

    @torch.inference_mode()
    def warp_decode(
        self, feature: torch.Tensor, x_s: torch.Tensor, x_d: torch.Tensor
    ) -> torch.Tensor:
        """Returns the animated face crop (1, 3, 512, 512), float32 in [0, 1]."""
        with self._amp():
            warped = self.W(feature, kp_source=x_s, kp_driving=x_d)
            out = self.G(feature=warped["out"])
        return out.float().clamp_(0, 1)
