"""Implicit-keypoint math from LivePortrait (MIT), adapted for live driving.

LivePortrait describes a face by 21 implicit 3D keypoints plus head pose, expression offsets,
translation and scale. The animated keypoints are (paper Eq. 2):

    x_d = s * (x_c @ R + exp) + t

In *relative* mode the character keeps its own look and only copies how much the user moved
since a neutral calibration frame d0:

    R_new   = (R_d @ R_d0^T) @ R_s
    exp_new = exp_s + (exp_d - exp_d0)
    s_new   = s_s * (s_d / s_d0)
    t_new   = t_s + (t_d - t_d0)          (tz forced to 0)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

NUM_KP = 21
# Flat parameter vector layout used for smoothing: pitch, yaw, roll, tx, ty, tz, scale, exp*63
PARAM_DIM = 3 + 3 + 1 + NUM_KP * 3


def headpose_to_degree(pred: torch.Tensor) -> torch.Tensor:
    """66-bin classification output -> degrees (LivePortrait convention), shape (B,)."""
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = torch.arange(66, device=pred.device, dtype=torch.float32)
        return (F.softmax(pred.float(), dim=1) * idx).sum(1) * 3 - 97.5
    return pred.reshape(-1).float()


def rotation_matrix(pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Degrees (B,) -> (B, 3, 3), returned transposed exactly like LivePortrait does."""
    x, y, z = (a.reshape(-1, 1) * (torch.pi / 180) for a in (pitch, yaw, roll))
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    rx = torch.cat([one, zero, zero, zero, x.cos(), -x.sin(), zero, x.sin(), x.cos()], 1)
    ry = torch.cat([y.cos(), zero, y.sin(), zero, one, zero, -y.sin(), zero, y.cos()], 1)
    rz = torch.cat([z.cos(), -z.sin(), zero, z.sin(), z.cos(), zero, zero, zero, one], 1)
    b = x.shape[0]
    rot = rz.view(b, 3, 3) @ ry.view(b, 3, 3) @ rx.view(b, 3, 3)
    return rot.permute(0, 2, 1)


@dataclass(frozen=True)
class KpInfo:
    pitch: torch.Tensor  # (B,) degrees
    yaw: torch.Tensor
    roll: torch.Tensor
    t: torch.Tensor  # (B, 3)
    exp: torch.Tensor  # (B, 21, 3)
    scale: torch.Tensor  # (B, 1)
    kp: torch.Tensor  # (B, 21, 3) canonical keypoints

    @staticmethod
    def from_raw(raw: dict[str, torch.Tensor]) -> KpInfo:
        """Convert the motion extractor's raw output dict."""
        b = raw["kp"].shape[0]
        return KpInfo(
            pitch=headpose_to_degree(raw["pitch"]),
            yaw=headpose_to_degree(raw["yaw"]),
            roll=headpose_to_degree(raw["roll"]),
            t=raw["t"].float().reshape(b, 3),
            exp=raw["exp"].float().reshape(b, NUM_KP, 3),
            scale=raw["scale"].float().reshape(b, 1),
            kp=raw["kp"].float().reshape(b, NUM_KP, 3),
        )

    @property
    def rotation(self) -> torch.Tensor:
        return rotation_matrix(self.pitch, self.yaw, self.roll)

    def to_params(self) -> npt.NDArray[np.float32]:
        """Batch-1 info -> flat float vector (for CPU-side smoothing)."""
        v = torch.cat(
            [
                torch.stack([self.pitch[0], self.yaw[0], self.roll[0]]),
                self.t[0],
                self.scale[0],
                self.exp[0].reshape(-1),
            ]
        )
        return v.detach().cpu().numpy().astype(np.float32)

    def with_params(self, p: npt.NDArray[np.float32]) -> KpInfo:
        """Replace pose/t/scale/exp from a flat vector (keeps canonical kp)."""
        d = self.t.device
        v = torch.as_tensor(p, dtype=torch.float32, device=d)
        return replace(
            self,
            pitch=v[0:1],
            yaw=v[1:2],
            roll=v[2:3],
            t=v[3:6].reshape(1, 3),
            scale=v[6:7].reshape(1, 1),
            exp=v[7:].reshape(1, NUM_KP, 3),
        )


def transform_keypoints(info: KpInfo) -> torch.Tensor:
    """Eq. 2: s * (x_c @ R + exp) + t, with tz ignored. Returns (B, 21, 3)."""
    x = info.kp @ info.rotation + info.exp
    x = x * info.scale[..., None]
    x[:, :, :2] = x[:, :, :2] + info.t[:, None, :2]
    return x


def relative_keypoints(
    src: KpInfo, drv: KpInfo, drv0: KpInfo, *, pose_strength: float = 1.0
) -> torch.Tensor:
    """Character keypoints that copy the user's motion relative to calibration frame ``drv0``.

    ``pose_strength`` scales head rotation/translation (1 = copy exactly).
    """
    if pose_strength == 1.0:
        r_rel = drv.rotation @ drv0.rotation.permute(0, 2, 1)
        t_new = src.t + (drv.t - drv0.t)
    else:
        k = pose_strength
        r_rel = rotation_matrix(
            (drv.pitch - drv0.pitch) * k, (drv.yaw - drv0.yaw) * k, (drv.roll - drv0.roll) * k
        )
        t_new = src.t + (drv.t - drv0.t) * k
    r_new = r_rel @ src.rotation
    exp_new = src.exp + (drv.exp - drv0.exp)
    scale_new = src.scale * (drv.scale / drv0.scale)
    t_new = t_new.clone()
    t_new[:, 2] = 0
    x = src.kp @ r_new + exp_new
    x = x * scale_new[..., None]
    x[:, :, :2] = x[:, :, :2] + t_new[:, None, :2]
    return x


def blend_toward(x: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Linear blend used to ease back to the neutral pose when the face is lost."""
    return x + (target - x) * alpha
