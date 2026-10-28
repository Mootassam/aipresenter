"""Render chosen frames of a driving video as the character (offline check, works on the CPU).

    uv run python tools/render_frames.py VIDEO CHARACTER OUT_DIR [--pick 6] [--frames 10,200]
        [--background original|color|image] [--bg-image IMAGE] [--cpu] [--mouth 0.8]

Frame 0 calibrates the neutral pose. ``--pick N`` chooses the N frames with the biggest head
turn / tilt / nod (the hardest cases for edges and background), ``--frames`` lists them by hand.
Each output image is the driving frame and the character side by side. ``--mouth X`` checks
the lip-sync hook: a third panel shows the same frame with a voice openness of X (0-1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import cv2
    import torch

    from aip.cli import _make_renderer
    from common.config import Settings
    from engine.avatar.character import _euler_from_matrix
    from engine.io.frames import Frame
    from engine.output.video import letterbox
    from engine.tracking.mediapipe_tracker import MediaPipeTracker

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("video", type=Path)
    ap.add_argument("character", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pick", type=int, default=6)
    ap.add_argument("--frames", help="comma-separated frame numbers instead of --pick")
    ap.add_argument("--background", choices=["original", "color", "image"], default="original")
    ap.add_argument("--bg-image", type=Path)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mouth", type=float, help="also render with this voice openness (0-1)")
    ap.add_argument("--lip-strength", type=float, default=1.0)
    ap.add_argument(
        "--brow-compare", action="store_true", help="also render without the brow-raise limit"
    )
    a = ap.parse_args()

    s = Settings()
    s.avatar = s.avatar.model_copy(
        update={"renderer": "liveportrait", "character_image": a.character}
    )
    s.avatar.background = a.background
    if a.bg_image:
        s.output.background_path = a.bg_image
    if a.cpu or not torch.cuda.is_available():
        s.avatar.device = "cpu"
    s.tracking.pose = s.tracking.hands = False

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()

    # with --frames only frame 0 + those are tracked (in order, so video-mode timestamps still rise)
    listed = sorted({int(x) for x in a.frames.split(",")}) if a.frames else None
    todo = [0, *listed] if listed else list(range(len(frames)))
    tracker = MediaPipeTracker(s.tracking, s.paths.models)
    tracks = {i: tracker.process(Frame(frames[i], i, i * 1000 / fps)) for i in todo}
    tracker.close()

    if listed:
        chosen = listed
    else:

        def motion(i: int) -> float:
            f = tracks[i].face
            if f is None:
                return -1.0
            pitch, yaw, roll = _euler_from_matrix(f.transform)
            return abs(yaw) + abs(roll) + 0.5 * abs(pitch)

        order = sorted(range(1, len(frames)), key=motion, reverse=True)
        chosen = []
        for i in order:  # spread out: no two picks within half a second
            if all(abs(i - j) > fps / 2 for j in chosen):
                chosen.append(i)
            if len(chosen) == a.pick:
                break
        chosen.sort()

    renderer = _make_renderer(s)
    a.out.mkdir(parents=True, exist_ok=True)
    renderer.render(Frame(frames[0], 0, 0.0), tracks[0])  # neutral calibration
    for i in chosen:
        renderer.f_pose.reset()
        renderer.f_exp.reset()
        frame = Frame(frames[i], i, i * 1000 / fps)
        imgs = [renderer.render(frame, tracks[i]).copy()]
        if a.brow_compare:  # order: driver | no limit | with limit
            renderer.f_pose.reset()
            renderer.f_exp.reset()
            renderer.cfg = s.avatar.model_copy(update={"brow_raise_max": 0.0})
            imgs.insert(0, renderer.render(frame, tracks[i]).copy())
            renderer.cfg = s.avatar
        if a.mouth is not None:
            renderer.f_pose.reset()
            renderer.f_exp.reset()
            renderer.cfg = s.avatar.model_copy(
                update={"lip_sync": "blend", "lip_sync_strength": a.lip_strength}
            )
            renderer.mouth = _Fixed(a.mouth)
            imgs.append(renderer.render(frame, tracks[i]))
            renderer.cfg, renderer.mouth = s.avatar, None
        h = 640
        left = letterbox(frames[i], round(h * frames[i].shape[1] / frames[i].shape[0]), h)
        right = np.hstack([letterbox(m, round(h * m.shape[1] / m.shape[0]), h) for m in imgs])
        f = tracks[i].face
        angles = _euler_from_matrix(f.transform) if f is not None else (0.0, 0.0, 0.0)
        out = a.out / f"frame{i:05d}.jpg"
        cv2.imwrite(str(out), np.hstack([left, right]))
        print(f"{out}  pitch {angles[0]:+.0f} yaw {angles[1]:+.0f} roll {angles[2]:+.0f}")
    return 0


class _Fixed:
    def __init__(self, value: float) -> None:
        self.value = value

    def openness(self, frame: object) -> float:
        return self.value


if __name__ == "__main__":
    raise SystemExit(main())
