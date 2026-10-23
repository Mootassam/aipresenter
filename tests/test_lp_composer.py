"""The face-area-only composer gives exactly the old full-canvas result (no GPU needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from engine.avatar.liveportrait.renderer import KEY_BGR, Composer, _to_bgr_u8, key_out

KEY = (KEY_BGR[2] / 255, KEY_BGR[1] / 255, KEY_BGR[0] / 255)


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
          bg: torch.Tensor | None) -> np.ndarray:  # fmt: skip
    """Reference: composite (and key) the whole canvas."""
    pasted = F.grid_sample(face, grid, align_corners=False, padding_mode="zeros")
    out = mask * pasted + (1 - mask) * canvas
    if bg is not None:
        out = key_out(out, bg, KEY)
    return _to_bgr_u8(out)


@pytest.mark.parametrize("with_bg", [False, True])
def test_same_pixels_as_full_canvas(with_bg: bool) -> None:
    canvas, grid, mask, face, _, bg = _setup()
    b = bg if with_bg else None
    comp = Composer(canvas, grid, mask, bg=b, key=KEY)
    y0, y1, x0, x1 = comp.roi
    assert (y1 - y0) * (x1 - x0) < canvas.shape[2] * canvas.shape[3]  # really a smaller area
    np.testing.assert_array_equal(comp.compose(face), _full(canvas, grid, mask, face, b))


def test_still_is_the_unanimated_character() -> None:
    canvas, grid, mask, _, _, bg = _setup()
    comp = Composer(canvas, grid, mask, bg=bg, key=KEY)
    np.testing.assert_array_equal(comp.still(), _to_bgr_u8(key_out(canvas, bg, KEY)))


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


def _px(rgb: tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(rgb).view(1, 3, 1, 1)


def test_key_colour_becomes_the_background() -> None:
    bg = _px((0.2, 0.3, 0.9))
    assert torch.allclose(key_out(_px(KEY), bg, KEY), bg, atol=1e-6)


@pytest.mark.parametrize("person", [(0.25, 0.2, 0.15), (0.9, 0.7, 0.6), (0.5, 0.5, 0.5)])
def test_person_colours_stay(person: tuple[float, float, float]) -> None:
    # dark hair, skin, grey: no green excess -> untouched
    assert torch.allclose(key_out(_px(person), _px((0, 0, 1)), KEY), _px(person), atol=1e-6)


def test_half_transparent_hair_edge_gets_the_new_background() -> None:
    hair, bg = _px((0.3, 0.3, 0.3)), _px((0.9, 0.1, 0.1))
    edge = 0.5 * hair + 0.5 * _px(KEY)  # what the network draws at a hair edge on the key
    out = key_out(edge, bg, KEY)
    want = 0.5 * hair + 0.5 * bg
    assert (out - want).abs().max() < 0.05  # no green fringe left


def test_green_share_counts_only_the_person() -> None:
    from engine.avatar.liveportrait.renderer import green_share

    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (90, 120, 200)  # skin-ish (BGR)
    img[:, :5] = (40, 200, 40)  # green on the left half
    alpha = np.zeros((10, 10), np.float32)
    alpha[:, 3:] = 1.0  # the person covers columns 3-9: 2 of their 7 columns are green
    assert green_share(img, alpha) == pytest.approx(2 / 7)
