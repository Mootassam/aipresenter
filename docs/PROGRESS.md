# Progress log (read this first when resuming)

Last updated: 2026-10-22

## Where we are

| Phase | Status |
|---|---|
| 0 Project setup | ✅ done (commit 50c435d) |
| 1 Tracking + virtual camera | ✅ done (commit b43ab8e). softcam DLL not built yet (no compiler on the laptop) |
| 2 Voice changer | ⏳ waiting on the owner (see below) |
| 3 Avatar v1 (head + shoulders) | 🔨 part 1 done on branch `phase3-avatar` (commit 7923553): LivePortrait engine works (checked on CPU). GPU speed test done on RunPod 2026-10-22 (10.3 fps, see below). Next: run tracking and rendering in parallel, TensorRT/FP16 for the render, identity stability, lip-sync hook, better matting |
| 4–9 | not started |

## Phase 3 notes
- The laptop's GTX 1050 is not supported by PyTorch cu128, so the avatar runs on the CPU there (8 s/frame, for checking quality only).
- SSH key for RunPod: `~/.ssh/aip_runpod` (the owner pastes the .pub into RunPod → Settings → SSH Public Keys).
- RunPod test: the owner deploys an RTX 4090 pod and gives Claude the SSH command; Claude copies the code + samples and runs `bash tools/runpod_bench.sh 60` (self-stop after 60 min). Then make sure the pod is TERMINATED.
- `samples/character.png` (1330×1182) passes the quality check; one tip: the hair touches the top edge.

### RunPod RTX 4090 speed test (2026-10-22, results in `results/runpod-2026-10-22/`, not in git)
- Bug found and fixed: `pick_device` refused the 4090 (sm_89) because the cu128 build lists sm_86, not
  sm_89, so the avatar ran on the CPU (0.1 fps). Fix: `build_runs_on()` (same major + lower minor, or
  PTX) with tests. This matters for the owner's RTX 5050 too.
- The pod image also needed `libegl1 libgles2` for MediaPipe (added to `tools/runpod_bench.sh`).
- Real-time result: **10.3 fps, delay p50 110 ms / p95 139 ms**, face found 100%, GPU memory only
  ~1.1 GB (peak 1.8 GB on the card) -> it will fit an 8 GB card easily.
- Per frame: tracking ~42 ms (MediaPipe on the pod's CPU) + render ~52 ms (motion 17, warp_decode 25,
  compose 10). They run one after the other, so ~95 ms -> ~10 fps.
- `--compile` (torch.compile) is NOT usable: Triton hits the 4090's shared-memory limit and the first
  frame took 3 minutes. Drop it; use TensorRT/FP16 instead.
- Quality (avatar.mp4): natural expressions and mouth, sharp face; the face shape drifts a little
  between frames (identity stability to improve).
- Next speed steps, expected to reach 25-30 fps: (1) track frame N+1 while rendering frame N
  (parallel), (2) FP16/TensorRT for warp_decode + motion, (3) compose on the GPU.

### Speed work 1 (2026-10-22, not measured on a GPU yet)
- Tracking and rendering now overlap (`video.overlap_stages`, default on; `--no-overlap` to compare):
  frame N+1 is tracked while frame N renders; hand-over keeps only the newest result.
- A renderer declares the tracking it needs (`needs`); the LivePortrait avatar only needs the face,
  so pose + hands are off for now (laptop: tracking 42 -> 18 ms). Phase 4 turns them back on.
- `Composer`: only the face area is composited and copied off the GPU; tests prove the pixels are
  identical to the full-canvas version.
- `tools/runpod_bench.sh`: the torch.compile step is replaced by a 30 s `--no-overlap` comparison.
- Next: re-run the RunPod bench (expect ~15-20 fps from these), then FP16/TensorRT for the render.

## Waiting on the owner before Phase 2

1. Install VB-Audio Virtual Cable (vb-audio.com/Cable) and restart the PC.
2. Provide a test RVC voice model (`.pth`, plus `.index` if available) they have rights to.
   No real-person voice clones.
3. Choose how to build softcam (virtual camera):
   A) install VS 2022 Build Tools on the laptop, then `tools\build_softcam.ps1 -Register`
   B) push the repo to GitHub and run the "Build softcam" workflow (also lets the friend's
      Claude join as a teammate)

## Decisions already made (don't re-ask)

- The owner wants the **complete product at the best quality**, all phases through the .exe
  installer. Priorities: avatar quality (hands, expressions, stable identity), then low delay.
- **Commercial-use licenses only.** Banned: InsightFace, pyvirtualcam, CodeFormer, FLUX.1-dev,
  XPose. **Check the PersonaLive + TalkingPose licenses before Phase 4.**
- softcam instead of pyvirtualcam. Nuitka + Inno Setup for packaging. 3-process model
  (GUI / video / audio). Python 3.11 via uv.
- VB-Cable: the user installs it (no bundling) for now.
- Realistic upper body + hands: cloud-first; local 8 GB is a later goal.
- The owner's laptop: GTX 1050 4 GB (too weak for the avatar). Avatar tests run on rented
  **RunPod** GPUs. Always give simple step-by-step rental instructions and make sure the pod
  shuts down (docs/RUNPOD_GUIDE.md). The owner's target PC: Core Ultra 7 155H + RTX 5050 8 GB.
- After each phase: explain in simple words what was built, how to test it, and what's next.
- The owner talks casually ("bro") and prefers short, simple explanations.
- The owner has a friend with their own Claude Code subscription who may join as a teammate
  via GitHub (issues/PRs as the communication channel, TEAM rules in CLAUDE.md).

## Key facts from Phase 1

- Sample video: `samples/talk.mp4` (60 s, 1080×1920 portrait, 30 fps, with audio; git-ignored).
  Hands are mostly out of frame in it.
- Laptop results: 23 fps, delay p95 78 ms, 10-minute soak stable (+0.4 MB/min).
- Run: `uv run aip track --file samples\talk.mp4 --output-size 720x1280`
