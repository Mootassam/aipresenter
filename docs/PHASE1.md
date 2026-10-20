# Phase 1: tracking + virtual camera

## What it does

Webcam or video file → face (478 points + 52 expressions + head pose), upper body (shoulders,
elbows, wrists, hips) and both hands (21 points each, all fingers) → debug drawing → preview
window and/or virtual camera.

## Run it

```powershell
cd D:\ai-presenter
uv sync --extra tracking                      # once

# with your sample video (portrait output for TikTok)
uv run aip track --file samples\talk.mp4 --output-size 720x1280

# with your webcam (camera 0)
uv run aip track --camera 0

# speed test without a window
uv run aip track --file samples\talk.mp4 --no-preview --no-vcam --duration 60 --report report.json
```

Press **Q** or **Esc** in the preview window to stop. Options: `--pose-model lite|full|heavy`,
`--sequential`, `--no-hud`, `--fast`.

## Measured results (dev laptop: 4-core Intel CPU from ~2017, GTX 1050, tracking on CPU)

10-minute non-stop run on `samples/talk.mp4` (1080×1920, 30 fps), real-time pacing:

| Metric | Result | Target |
|---|---|---|
| Throughput | **23 fps** (camera gives 30; extra frames are dropped, never queued) | 25–30 |
| Delay capture → output | **p50 58 ms, p95 78 ms, max 109 ms** | < 200 ms (whole chain) |
| Tracking (3 models in parallel) | p50 28 ms | — |
| Face found | 100 % of frames | — |
| Memory | 248 → 419 MB, trend **+0.4 MB/min** (stable) | no growth |
| Crashes | none | none |

Running the three models in parallel instead of one after another: 13 → 20 fps.
`--pose-model lite`: about +3 fps for slightly less precise shoulders and elbows.

**On your Core Ultra 7 155H (16 cores), expect well above 30 fps.** That's an estimate; please run
the speed test above and send me the numbers.

Hands are detected in only 15–40 % of frames **because your hands are below the bottom of the
picture most of the time** in `talk.mp4`. When they're visible, all fingers are tracked
correctly and left/right is right. For the product: the setup wizard will check framing
("show both hands at chest height").

## Virtual camera (softcam)

softcam has no ready-made download, so it has to be compiled once. Two options:

**A. On your PC** (needs Visual Studio 2022 Build Tools, free for individuals and small
companies, about 6 GB):

1. Install: `winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
2. Build and register (one admin prompt):
   `powershell -ExecutionPolicy Bypass -File tools\build_softcam.ps1 -Register`

**B. On GitHub** (no install): once the repo is on GitHub, run the *Build softcam* workflow and
download `softcam.dll` + `softcam_installer.exe` into `third_party\softcam\`. Then register:
`third_party\softcam\softcam_installer.exe register third_party\softcam\softcam.dll` (as admin).

Then `uv run aip track --camera 0` also sends to the virtual camera. In OBS: *Sources → +
Video Capture Device → "DirectShow Softcam"*.

To check: hold up something **red**. If it shows blue in OBS, the colour order is swapped (tell me).

Until then, OBS can show the preview window with *Window Capture*.

## Known limits (honest)

- Tracking runs on the CPU. MediaPipe has no GPU support in Python on Windows. That's fine:
  it leaves the GPU free for the avatar and voice.
- softcam is a DirectShow camera. OBS, Zoom, Discord, Chrome and most desktop apps see it.
  The Windows "Camera" app and some Microsoft Store apps don't. Long-term option: the Windows 11
  virtual camera API (Phase 9).
- The delay number excludes the webcam's own delay (usually 30–60 ms) and the receiving app's
  delay. We'll measure the full chain with a flashing-screen test in Phase 3.
