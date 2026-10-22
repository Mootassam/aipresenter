"""The CUDA-graph face renderer draws exactly what the normal one draws (needs a GPU + models)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_graph_matches_eager() -> None:
    from common.config import Settings
    from engine.avatar.liveportrait.motion import transform_keypoints
    from engine.avatar.liveportrait.network import LivePortraitNet, build_runs_on

    major, minor = torch.cuda.get_device_capability()
    if not build_runs_on(major, minor, torch.cuda.get_arch_list()):
        pytest.skip("this PyTorch build doesn't support the GPU")
    net = LivePortraitNet(Settings().paths.models, torch.device("cuda"))
    x = net.to_input((torch.rand(256, 256, 3) * 255).to(torch.uint8))
    f_s = net.appearance(x)
    x_s = transform_keypoints(net.motion(x))
    for _ in range(3):
        x_d = x_s + 0.02 * torch.randn_like(x_s)
        graphed = net.warp_decode(f_s, x_s, x_d).clone()
        with torch.inference_mode():
            eager = net._warp_decode(f_s, x_s, x_d)
        assert net._warp_graph is not None  # really recorded, not the fallback
        assert (graphed - eager).abs().max().item() < 1e-6
