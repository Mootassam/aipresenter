"""Face-shape stability of a video: does the face keep the same proportions from frame to frame?

    uv run python tools/identity_metrics.py VIDEO [--every N] [--image STILL]

For every Nth frame, MediaPipe face landmarks (3D, in pixels) give a few shape ratios that don't
change when someone talks or blinks, all divided by the distance between the outer eye corners:
cheek width, jaw width, nose width and eye-to-mouth-corner height. A real person's ratios only
wobble by measuring noise; a much bigger spread on an avatar means its face shape drifts.
With --image, the ratios of the still character image are printed for comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MediaPipe face mesh indices
EYES = (33, 263)  # outer eye corners
RATIOS = {
    "cheek_width": (234, 454),  # face sides at the cheekbones
    "jaw_width": (172, 397),  # jaw angles
    "nose_width": (129, 358),  # nostril wings
    "eye_mouth_left": (33, 61),  # outer eye corner -> mouth corner (moves little when talking)
    "eye_mouth_right": (263, 291),
}


def ratios(pts: np.ndarray) -> dict[str, float]:
    """pts: (478, 3) landmarks in pixels."""
    base = float(np.linalg.norm(pts[EYES[0]] - pts[EYES[1]]))
    return {k: float(np.linalg.norm(pts[a] - pts[b])) / base for k, (a, b) in RATIOS.items()}


def _landmarker(models_dir: Path, video: bool):  # type: ignore[no-untyped-def]
    from mediapipe.tasks.python import BaseOptions, vision

    from engine.models import ensure_model

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model("face_landmarker", models_dir))),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _points(res, w: int, h: int) -> np.ndarray | None:  # type: ignore[no-untyped-def]
    if not res.face_landmarks:
        return None
    return np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]], np.float32)


def measure_video(path: Path, models_dir: Path, every: int = 1) -> dict[str, np.ndarray]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: dict[str, list[float]] = {k: [] for k in RATIOS}
    i = 0
    with _landmarker(models_dir, video=True) as lm:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % every == 0:
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps)
                )
                pts = _points(res, w, h)
                if pts is not None:
                    for k, v in ratios(pts).items():
                        out[k].append(v)
            i += 1
    cap.release()
    return {k: np.array(v) for k, v in out.items()}


def measure_image(path: Path, models_dir: Path) -> dict[str, float] | None:
    import cv2
    import mediapipe as mp

    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    with _landmarker(models_dir, video=False) as lm:
        res = lm.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        )
    pts = _points(res, w, h)
    return None if pts is None else ratios(pts)


def report(name: str, m: dict[str, np.ndarray], ref: dict[str, float] | None = None) -> float:
    """Prints per-ratio spread; returns the mean spread (std / mean, in %)."""
    n = len(next(iter(m.values())))
    print(f"\n{name}: {n} frames with a face")
    spreads = []
    for k, v in m.items():
        cv = 100 * float(v.std() / v.mean())
        spreads.append(cv)
        extra = (
            f"  still image {ref[k]:.3f} (avg off by {100 * (v.mean() / ref[k] - 1):+.1f}%)"
            if ref
            else ""
        )
        lo, hi = np.percentile(v, [5, 95])
        print(f"  {k:<16} mean {v.mean():.3f}  spread {cv:4.1f}%  range {lo:.3f}-{hi:.3f}{extra}")
    print(f"  => average spread {np.mean(spreads):.2f}%")
    # flicker: how much the shape jumps between neighbouring measured frames (not slow changes)
    jumps = [100 * float(np.median(np.abs(np.diff(v)) / v[:-1])) for v in m.values() if len(v) > 2]
    print(f"  => frame-to-frame jitter {np.mean(jumps):.2f}% (median jump)")
    return float(np.mean(spreads))


def main() -> int:
    from common.config import Settings

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=1, help="measure every Nth frame")
    ap.add_argument("--image", type=Path, help="the still character image, for comparison")
    a = ap.parse_args()
    models = Settings().paths.models
    ref = measure_image(a.image, models) if a.image else None
    for v in a.videos:
        report(v.name, measure_video(v, models, a.every), ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
