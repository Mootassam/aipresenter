from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine.avatar.liveportrait.motion import (  # noqa: E402
    PARAM_DIM,
    KpInfo,
    headpose_to_degree,
    relative_keypoints,
    rotation_matrix,
    transform_keypoints,
)


def _info(pitch: float = 0, yaw: float = 0, roll: float = 0, seed: int = 0) -> KpInfo:
    g = torch.Generator().manual_seed(seed)
    return KpInfo(
        pitch=torch.tensor([pitch]),
        yaw=torch.tensor([yaw]),
        roll=torch.tensor([roll]),
        t=torch.randn(1, 3, generator=g) * 0.05,
        exp=torch.randn(1, 21, 3, generator=g) * 0.01,
        scale=torch.tensor([[1.5]]),
        kp=torch.randn(1, 21, 3, generator=g) * 0.3,
    )


def test_rotation_is_orthonormal() -> None:
    r = rotation_matrix(torch.tensor([10.0]), torch.tensor([-25.0]), torch.tensor([5.0]))
    assert torch.allclose(r @ r.transpose(1, 2), torch.eye(3)[None], atol=1e-5)
    assert torch.det(r).item() == pytest.approx(1.0, abs=1e-5)


def test_rotation_identity_at_zero() -> None:
    z = torch.zeros(1)
    assert torch.allclose(rotation_matrix(z, z, z), torch.eye(3)[None])


def test_headpose_bins_to_degrees() -> None:
    logits = torch.full((1, 66), -1e4)
    logits[0, 40] = 0
    assert headpose_to_degree(logits).item() == pytest.approx(40 * 3 - 97.5)


def test_relative_motion_identity_when_user_is_neutral() -> None:
    src, drv0 = _info(seed=1), _info(pitch=5, yaw=-3, seed=2)
    x = relative_keypoints(src, drv0, drv0)
    expected = transform_keypoints(src)
    expected_no_tz = expected.clone()
    # relative mode zeroes tz; transform_keypoints never applies tz to xy anyway
    assert torch.allclose(x, expected_no_tz, atol=1e-5)


def test_relative_motion_copies_yaw_change() -> None:
    src = _info(seed=1)
    drv0 = _info(yaw=0, seed=2)
    drv = _info(yaw=20, seed=2)
    x = relative_keypoints(src, drv, drv0)
    src_turned = KpInfo(src.pitch, src.yaw + 20, src.roll, src.t, src.exp, src.scale, src.kp)
    assert torch.allclose(x, transform_keypoints(src_turned), atol=1e-4)


def test_pose_strength_zero_freezes_head() -> None:
    src, drv0 = _info(seed=1), _info(seed=2)
    drv = _info(yaw=30, pitch=10, seed=2)
    x = relative_keypoints(src, drv, drv0, pose_strength=0.0)
    assert torch.allclose(x, transform_keypoints(src), atol=1e-5)


def test_params_roundtrip() -> None:
    info = _info(pitch=3, yaw=-7, roll=1, seed=3)
    p = info.to_params()
    assert p.shape == (PARAM_DIM,)
    back = info.with_params(p)
    assert torch.allclose(transform_keypoints(back), transform_keypoints(info), atol=1e-6)
    p2 = p.copy()
    p2[1] += 10  # yaw
    assert back.with_params(p2).yaw.item() == pytest.approx(-7 + 10, abs=1e-5)
    assert np.allclose(info.with_params(p2).to_params(), p2)


def test_limit_brow_raise_caps_big_raises_softly() -> None:
    from engine.avatar.liveportrait.motion import limit_brow_raise

    exp0 = np.zeros(63, np.float32)
    small, big = exp0.copy(), exp0.copy()
    small[7], big[7] = -0.003, -0.04  # kp 2 y down = brows up
    big[4] = 0.01  # kp 1 y moves up with it
    out_small = limit_brow_raise(small, exp0, 0.015)
    out_big = limit_brow_raise(big, exp0, 0.015)
    assert out_small[7] == pytest.approx(-0.003, rel=0.02)  # normal talking passes through
    assert -0.015 < out_big[7] < -0.014  # strong raise approaches the cap
    assert 0 < out_big[4] < 0.01  # the side point is scaled by the same amount
    assert np.array_equal(np.delete(out_big, [4, 7]), np.delete(big, [4, 7]))


def test_limit_brow_raise_leaves_frowns_and_off_alone() -> None:
    from engine.avatar.liveportrait.motion import limit_brow_raise

    exp0 = np.full(63, 0.01, np.float32)
    frown = exp0.copy()
    frown[7] = 0.05
    assert np.array_equal(limit_brow_raise(frown, exp0, 0.015), frown)
    up = exp0.copy()
    up[7] = -0.05
    assert np.array_equal(limit_brow_raise(up, exp0, 0.0), up)
