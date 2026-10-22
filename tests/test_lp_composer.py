"""The face-area-only composer gives exactly the old full-canvas result (no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from engine.avatar.liveportrait.renderer import Composer, _to_bgr_u8


def _setup(h: int = 90, w: int = 120) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    canvas = torch.rand(1, 3, h, w, generator=g)
    # the 512 crop covers the middle of the canvas: grid in [-1, 1] there, outside elsewhere
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")  # fmt: skip
    grid = torch.stack([(xs - w / 2) / (w / 4), (ys - h / 2) / (h / 4)], -1)[None]
    mask_512 = torch.zeros(1, 1, 64, 64)
    mask_512[..., 8:56, 8:56] = 1.0  # soft-edged in reality; hard edges are the harder case
    mask = F.grid_sample(mask_512, grid, align_corners=False, padding_mode="zeros")
    face = torch.rand(1, 3, 64, 64, generator=g)
    alpha = torch.rand(1, 1, h, w, generator=g)
    bg = torch.rand(1, 3, h, w, generator=g)
    return canvas, grid, mask, face, alpha, bg


def _full(canvas: torch.Tensor, grid: torch.Tensor, mask: torch.Tensor, face: torch.Tensor,
          alpha: torch.Tensor | None, bg: torch.Tensor | None) -> np.ndarray:  # fmt: skip
    """The previous renderer: composite the whole canvas."""
    pasted = F.grid_sample(face, grid, align_corners=False, padding_mode="zeros")
    out = mask * pasted + (1 - mask) * canvas
    if alpha is not None and bg is not None:
        out = alpha * out + (1 - alpha) * bg
    return _to_bgr_u8(out)


@pytest.mark.parametrize("with_bg", [False, True])
def test_same_pixels_as_full_canvas(with_bg: bool) -> None:
    canvas, grid, mask, face, alpha, bg = _setup()
    a, b = (alpha, bg) if with_bg else (None, None)
    comp = Composer(canvas, grid, mask, a, b)
    y0, y1, x0, x1 = comp.roi
    assert (y1 - y0) * (x1 - x0) < canvas.shape[2] * canvas.shape[3]  # really a smaller area
    np.testing.assert_array_equal(comp.compose(face), _full(canvas, grid, mask, face, a, b))


def test_still_is_the_unanimated_character() -> None:
    canvas, grid, mask, _, alpha, bg = _setup()
    comp = Composer(canvas, grid, mask, alpha, bg)
    np.testing.assert_array_equal(comp.still(), _to_bgr_u8(alpha * canvas + (1 - alpha) * bg))


def test_frames_are_independent_copies() -> None:
    canvas, grid, mask, face, _, _ = _setup()
    comp = Composer(canvas, grid, mask)
    first = comp.compose(face)
    first[:] = 0  # e.g. the HUD drawing on a frame must not leak into the next one
    assert comp.compose(face).any()


def test_empty_mask() -> None:
    canvas, grid, _, face, _, _ = _setup()
    comp = Composer(canvas, grid, torch.zeros(1, 1, *canvas.shape[2:]))
    np.testing.assert_array_equal(comp.compose(face), _to_bgr_u8(canvas))
