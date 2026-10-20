# Progress log (read this first when resuming)

Last updated: 2026-10-20

## Where we are

| Phase | Status |
|---|---|
| 0 Project setup | ✅ done (commit 50c435d) |
| 1 Tracking + virtual camera | ✅ done (commit b43ab8e). softcam DLL not built yet (no compiler on the laptop) |
| 2 Voice changer | ⏳ next, waiting on the owner (see below) |
| 3–9 | not started |

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
