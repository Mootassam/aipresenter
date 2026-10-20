from __future__ import annotations

from common.system_info import GpuInfo, parse_nvidia_smi_csv, recommend_mode


def _gpu(name: str, vram: int, cc: tuple[int, int] | None) -> GpuInfo:
    return GpuInfo(0, name, vram, 0, "580.00", cc)


def test_parse_two_gpus() -> None:
    text = (
        "NVIDIA GeForce RTX 5050 Laptop GPU, 8151, 512, 582.78, 12.0\n"
        "NVIDIA GeForce GTX 1050, 4096, 100, 582.78, 6.1\n"
    )
    gpus = parse_nvidia_smi_csv(text)
    assert [g.name for g in gpus] == [
        "NVIDIA GeForce RTX 5050 Laptop GPU",
        "NVIDIA GeForce GTX 1050",
    ]
    assert gpus[0].vram_total_mb == 8151
    assert gpus[0].compute_cap == (12, 0)
    assert gpus[1].index == 1


def test_parse_handles_garbage_and_na() -> None:
    gpus = parse_nvidia_smi_csv("\nbroken line\nGPU X, [N/A], [N/A], 1.0, [N/A]\n")
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 0
    assert gpus[0].compute_cap is None


def test_recommend_no_gpu() -> None:
    assert recommend_mode([])[0] == "cloud"


def test_recommend_gtx1050_cloud() -> None:
    mode, reason = recommend_mode([_gpu("GTX 1050", 4096, (6, 1))])
    assert mode == "cloud"
    assert "too old" in reason


def test_recommend_rtx4050_6gb_cloud() -> None:
    assert recommend_mode([_gpu("RTX 4050", 6141, (8, 9))])[0] == "cloud"


def test_recommend_rtx5050_local() -> None:
    assert recommend_mode([_gpu("RTX 5050", 8151, (12, 0))])[0] == "local"
