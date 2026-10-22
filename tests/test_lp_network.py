"""Which GPUs a PyTorch build can run on (no GPU needed)."""

from __future__ import annotations

from engine.avatar.liveportrait.network import build_runs_on

# torch 2.11 cu128 wheel
CU128 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def test_exact_arch() -> None:
    assert build_runs_on(8, 6, CU128)  # RTX 3090
    assert build_runs_on(12, 0, CU128)  # RTX 5050 / 5090


def test_same_major_newer_minor() -> None:
    assert build_runs_on(8, 9, CU128)  # RTX 4090: runs the sm_86 code


def test_too_old() -> None:
    assert not build_runs_on(6, 1, CU128)  # GTX 1050


def test_ptx_runs_on_newer() -> None:
    assert build_runs_on(12, 1, CU128)
    assert not build_runs_on(7, 0, ["compute_75"])


def test_ignores_odd_entries() -> None:
    assert not build_runs_on(8, 9, ["", "sm_", "gfx90a"])
