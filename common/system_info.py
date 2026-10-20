"""Hardware detection used by the setup wizard, ``aip doctor`` and the benchmark.

Decides whether the machine can run Local mode or should use Cloud mode.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal

# Minimum for realistic avatar + voice in Local mode (target hardware: RTX 5050 8 GB).
LOCAL_MIN_VRAM_MB = 7500
LOCAL_MIN_COMPUTE_CAP = (7, 5)  # Turing or newer: needed for fast FP16 TensorRT

_NVSMI_FIELDS = "name,memory.total,memory.used,driver_version,compute_cap"


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    driver: str
    compute_cap: tuple[int, int] | None


def parse_nvidia_smi_csv(text: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu=<_NVSMI_FIELDS> --format=csv,noheader,nounits``."""
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, total, used, driver, cc = parts[:5]
        cap: tuple[int, int] | None = None
        if "." in cc:
            major, minor = cc.split(".", 1)
            if major.isdigit() and minor.isdigit():
                cap = (int(major), int(minor))
        gpus.append(
            GpuInfo(
                index=len(gpus),
                name=name,
                vram_total_mb=int(float(total)) if total.replace(".", "").isdigit() else 0,
                vram_used_mb=int(float(used)) if used.replace(".", "").isdigit() else 0,
                driver=driver,
                compute_cap=cap,
            )
        )
    return gpus


def query_gpus(timeout_s: float = 5.0) -> list[GpuInfo]:
    """Return NVIDIA GPUs, or an empty list if there are none / no driver."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_NVSMI_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_nvidia_smi_csv(out)


def recommend_mode(gpus: list[GpuInfo]) -> tuple[Literal["local", "cloud"], str]:
    """Pick the run mode for the setup wizard, with a human-readable reason."""
    if not gpus:
        return "cloud", "No NVIDIA GPU found."
    best = max(gpus, key=lambda g: g.vram_total_mb)
    if best.compute_cap is not None and best.compute_cap < LOCAL_MIN_COMPUTE_CAP:
        return "cloud", f"{best.name} is too old for real-time AI video (needs RTX 20xx or newer)."
    if best.vram_total_mb < LOCAL_MIN_VRAM_MB:
        return "cloud", (
            f"{best.name} has {best.vram_total_mb // 1024} GB VRAM; Local mode needs 8 GB."
        )
    return "local", f"{best.name} ({best.vram_total_mb // 1024} GB) can run Local mode."


def system_report() -> dict[str, Any]:
    gpus = query_gpus()
    mode, reason = recommend_mode(gpus)
    return {
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "gpus": [asdict(g) for g in gpus],
        "recommended_mode": mode,
        "reason": reason,
    }
