# Progress log (read this first when resuming)

Last updated: 2026-10-22

## Where we are

| Phase | Status |
|---|---|
| 0 Project setup | ✅ done (commit 50c435d) |
| 1 Tracking + virtual camera | ✅ done (commit b43ab8e). softcam DLL not built yet (no compiler on the laptop) |
| 2 Voice changer | ⏳ waiting on the owner (see below) |
| 3 Avatar v1 (head + shoulders) | 🔨 on branch `phase3-avatar`: LivePortrait engine works; **27 fps on an RTX 4090** (2026-10-22, see below). Background fix + lip-sync hook done and checked on a 4090; identity check done (face is fine at rest). **Next: the owner picks** (1) forehead wrinkles from strong brow raises, (2) lost resting smile, or (3) sharpness (Claude suggests 1 first), then check speed on an 8 GB card (RTX 5050 class) |
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
- RunPod re-run: overlap alone was SLOWER (12.3 vs 13.0 fps serial): the render launches thousands
  of small GPU kernels per frame, so its CPU work fought the tracking thread for Python's GIL.

### Speed work 2: CUDA graph (2026-10-22, RTX 4090)
- `LivePortraitNet.warp_decode` records warping + generator as one CUDA graph (`avatar.cuda_graphs`,
  default on; `--no-cuda-graphs` to compare). Falls back to normal mode if recording fails.
  Two vendored helpers now create their tensors on the GPU (no CPU->GPU copy, needed for recording).
- Same output: max pixel difference 0.000 vs normal mode (`tests/test_lp_cuda_graph.py`, GPU test).
- **Result: 27.2 fps, delay p50 71 ms / p95 89 ms** with overlap (17.7 fps without overlap). The
  file source is 30 fps, so this is close to every frame. Results: `results/runpod-2026-10-22c/`.
- Per frame now: tracking ~20 ms (CPU) in parallel with render ~34 ms (motion 12 + GPU work ~22).
- TensorRT is not needed for the 4090. Still to check: an 8 GB RTX 5050-class card.

### Background fix (2026-10-22, CPU-checked)
- With background replacement, the old background showed through where the head moved. Now the
  character is first put on a hidden key green, animated on it, and the green is keyed out every
  frame (`key_out` in the renderer). Warns if the character wears strong green.
- `stretch_matte`: the segmenter never reached 0 on the background; its 1st/99th percentiles are
  now mapped to 0/1, so the old background no longer shows through faintly.
- Tools: `tools/render_frames.py` (render chosen/hardest frames side by side, works on the CPU),
  `tools/identity_metrics.py` (face-shape drift of a video: ratios that don't change when talking).

### Lip-sync hook (2026-10-22, CPU-checked)
- `engine/avatar/mouth.py`: `VoiceMouthMeter` turns voice audio into mouth openness 0-1 every
  10 ms (speech-band loudness between an adaptive noise floor and the recent peak, fast open,
  slower close). `MouthMailbox` is where the voice engine (Phase 2) will publish it.
  `FileMouthTrack` uses a test video's own audio (via ffmpeg, dev only).
- The renderer moves the lips with LivePortrait's lip-retargeting network: tracked opening pulled
  toward the voice (`avatar.lip_sync`, `lip_sync_strength` 0.6, `lip_sync_max_ratio` 0.4,
  `lip_sync_offset_ms`). Silence closes an open mouth only half as strongly (a silent laugh
  stays open). Default **off** until the voice engine exists.
- Try it: `uv run aip track --file samples	alk.mp4 --character samples\character.png --lip-sync`
  (`--lip-strength 1` = voice only). Check single frames: `tools/render_frames.py ... --mouth 1.0`.
- CPU check: voice 1.0 opens a closed mouth naturally (teeth visible); silence half-closes an
  open one.
- RunPod RTX 4090 (2026-10-22, `results/runpod-2026-10-22d/`, not in git): lip-sync costs ~1 ms
  (27.0 vs 28.0 fps real time). How well the avatar's lips follow the voice (correlation of lip
  opening with voice loudness, all 1800 frames): off 0.13, strength 0.6 -> 0.48, strength 1.0 ->
  0.64. Lips trail the voice by ~50 ms (smoothing); fine live, where the voice engine adds delay.
  `compare_lipsync.mp4` (driver | off | 0.6 | 1.0, with sound) and `compare_bg.mp4` (green
  background, clean edges while the head moves) are for the owner to watch.
- Bug fixed: recording a file with overlap on (`--fast --record`) skipped about half the frames,
  so recorded videos were shorter than their sound. The hand-over now waits instead of dropping
  when the source is a file decoded as fast as possible (`_TrackedSlot(lossless=True)`).
- Finding: in `samples/talk.mp4` the lips follow the sound only weakly (correlation ~0.2, broad
  peak at +300 ms), so it is a poor clip for tuning timing. A clip of the owner talking into
  their webcam would be better.

### Identity stability check (2026-10-22, RTX 4090, `results/runpod-2026-10-22e/`, not in git)
- At rest the render matches the still within 1-2% (nose -2%): the generator itself keeps the face.
- `identity_metrics.py --driver` compares the avatar's shape change with the driver's frame by
  frame. Most of the measured change is the avatar copying the user's expressions and head tilt
  (eye-to-mouth follows the driver at 0.87-0.91). Fixed offsets while talking: eye-to-mouth +9%
  (the character's resting smile disappears: the driver's calibration frame was smiling),
  nose -4.5%.
- Tried an "identity lock" (hold back the expression change on LivePortrait's outline keypoints
  0,3,4,5,7,8,9,10, as in its expression-friendly mode): average spread 3.34% -> 3.16%, own
  drift 2.69% -> 2.76%, no visible difference in the side-by-side video. **Removed** (no value).
  `expression_strength` 0.8: spread 2.89%, but only because she moves less.
- Seen in the videos and worth fixing next instead: strong brow raises put forehead wrinkles on
  the character (looks older); the resting smile is lost; the face is softer than the still.
- This pod: real time 29.7 fps (the source's own 30 fps), delay p50 36 ms / p95 38 ms.

### Fix 1: forehead wrinkles (2026-10-28, CPU-checked)
- Owner picked fix (1). Expression keypoint 2 y (and a little kp 1 y) follows the brow height
  (r = -0.76 vs MediaPipe). `limit_brow_raise` in `motion.py` softly caps brow raises against the
  calibration expression (tanh knee, `avatar.brow_raise_max` 0.015, `--brow-max`, 0 = off).
  Frowns are not touched. Tests in `tests/test_lp_motion.py`.
- `tools/render_frames.py --brow-compare` renders driver | no limit | with limit
  (`results/brow-check/`, frames 120, 906, 1359): at 1359 the wrinkles are gone, and the rest of
  the expression is unchanged.
- Next: fix (2) resting smile, then (3) sharpness, then the 8 GB card speed check.

### Fix 2: resting smile (2026-10-28, CPU-checked)
- Cause measured: in `talk.mp4` frame 0 (the calibration frame) the user smiles (MediaPipe smile
  0.73; calm quiet frames ~0.25), so every calmer face pulled the character's smile below her
  still.
- `NeutralExpression` (`motion.py`): head pose still calibrates on frame 0, but for
  `avatar.neutral_window_s` (5 s, `--neutral-window`, 0 = off) the **mouth keypoints only**
  (13, 14, 16, 17, 19, 20, the ones LivePortrait's smile/lip editors move) of the expression
  baseline move to the calmest face seen (`calm_score` = smile + jaw open), easing over 300 ms.
- First try moved the whole expression baseline: smile came back, but she frowned/squinted on
  calm frames and her face shape changed. Mouth-only fixed both.
- `tools/render_frames.py --neutral-compare`: driver | frame-0 baseline | settled
  (`results/smile-check2/`, frames 231, 852, 1307).
- Next (done below): fix (3) sharpness, then the 8 GB card speed check. A full-video GPU run should confirm
  fixes 1 + 2 together (identity_metrics eye-to-mouth offset was +9%).

### Fix 3: sharpness (2026-10-28, CPU-checked)
- Measured (Laplacian variance inside the face, same face at rest): still 210, render 42 (5x
  softer). Causes: the generator works from a 256 copy of the face, and its 512 output is
  enlarged to ~735 canvas px for samples/character.png.
- `Composer(sharpen=...)` / `avatar.face_sharpen` 0.8 (`--sharpen`, 0 = off = old pixels
  exactly): unsharp mask (sigma 1 px in 512 space) on the face, then bicubic paste.
  Scores: bicubic alone 68, + sharpen 0.6 -> 151, 0.8 -> ~190, 1.0 -> 225.
  No halos or seams seen (`results/sharp-check/`). Cost on the GPU not measured yet (2 small
  convolutions on 512x512, expected < 1 ms).
- Next: one RunPod run on the full video to confirm fixes 1-3 together (speed + identity_metrics),
  then the 8 GB (RTX 5050 class) speed check. If more detail is wanted later: a face
  restoration model with a commercial licence (not CodeFormer).

### RunPod check of fixes 1-3 (2026-10-28, RTX 4090, `results/runpod-2026-10-28/`, not in git)
- `tools/runpod_bench.sh` step 5 now records the video with the fixes off too, runs
  identity_metrics on both and makes `compare_fixes.mp4` (driver | before | after, with sound).
- Real time: **26.1 fps**, delay p50 66 / p95 87 ms (was 27.2 fps on another 4090 pod; pods
  differ in CPU). Profile: compose 5.3 ms incl. sharpening, warp_decode 18.7, motion 15.7.
  GPU memory: PyTorch peak 1.1 GB, card peak 2.1 GB.
- Face shape vs the still (avg off by), fixes off -> on: eye-to-mouth +9.2/+9.5% -> +2.4/+2.5%
  (resting smile back), nose -4.6% -> -2.2%, cheek +1.2% -> +0.2%, jaw -2.4% -> -2.0%.
  Own drift about the same (2.59% -> 2.71%), jitter the same (0.33%).
- Pod: Claude could not stop/terminate it (blocked by the permission guard): the owner
  terminates it in the RunPod web page. Self-stop timer was set to 60 min.
- Next: the 8 GB card (RTX 5050 class) speed check, then Phase 4.

### Phase 4 research (2026-10-28)
- Owner moved the 8 GB check to later. Model + licence research is in `docs/PHASE4_RESEARCH.md`.
  No real-time + commercial-OK upper-body/hands model exists today. Next: RunPod test of
  Wan-Animate-2 distilled (Apache 2.0), watch for Wan-Animate-2-Lite weights.

## Waiting on the owner before Phase 2

1. Install VB-Audio Virtual Cable (vb-audio.com/Cable) and restart the PC.
2. Provide a test RVC voice model (`.pth`, plus `.index` if available) they have rights to.
   No real-person voice clones.
3. Choose how to build softcam (virtual camera):
   A) install VS 2022 Build Tools on the laptop, then `tools\build_softcam.ps1 -Register`
   B) push the repo to GitHub and run the "Build softcam" workflow (also lets the friend's
      Claude join as a teammate)

## Decisions already made (don't re-ask)

- **Voice (Phase 2) is done LAST** (owner, 2026-10-22). Order: finish Phase 3, then 4, 5, 6, 7, 8, 9, then 2.

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
