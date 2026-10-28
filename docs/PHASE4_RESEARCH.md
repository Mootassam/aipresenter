# Phase 4 research: upper body + hands models (2026-10-28)

Goal: one character image + the user's webcam -> the character's upper body, arms and hands move
with the user, live (>= ~20 fps, low delay), and **commercial use allowed**.

Short answer: **no model does all of that today.** The closest commercial-OK model is
Wan-Animate-2 (Apache 2.0): full quality but offline; its real-time "Lite" variant is announced
but its weights are not public (checked on Hugging Face 2026-10-28).

## Candidates

| Model | What it does | Speed | Licence | Verdict |
|---|---|---|---|---|
| **Wan-Animate-2** (Alibaba Wan, Aug 2026) | Character image + driving video -> animated character, body/hands/face, no pose extraction | Offline. Defaults 8x A800 (720p), 2x A800 (480p). Distilled 10-step model released | **Apache 2.0** (HF: `Wan-AI/Wan2.2-Animate-2-14B`, `...-Distilled-Diffusers`) | Best commercial-OK quality. Test it. |
| **Wan-Animate-2-Lite** | Same, streaming | Claimed 24 fps at 400x720 | Apache 2.0 (expected) | **Weights not on HF yet.** Watch it: this is the one we want. |
| **LiveAvatar** (Alibaba Quark) | Image + **audio** -> talking upper body with gestures, streaming | 20 fps on 5x H800; 1 GPU needs 80 GB | Apache 2.0 (LoRA on Wan2.2-S2V-14B) | Audio-driven, does not copy the user's hands. Maybe a later "voice only" mode. |
| **GUAVA** (upper-body 3D Gaussian avatar) | One image -> 3D upper-body avatar, real-time render, >= 6 GB VRAM | Real time (no fps given), 512x512 | Code Apache 2.0, **but needs SMPL-X + FLAME: non-commercial**. Commercial SMPL-X via Meshcapade (from ~EUR 1,500/yr micro tier, prices change) | Only real-time + local-capable option. Needs a paid body-model licence, even for product R&D. Hands unclear. |
| PersonaLive (CVPR 2026) | Portrait (face) streaming animation | ~16 fps, 12 GB | Apache 2.0 code, but README says **"academic research only"** | Face only, and unclear licence. Excluded. |
| TalkingPose (WACV 2026) | Upper body + hands from pose | Not stated | **"Research-Only"** badge (MIT in sidebar: conflicting) | Excluded until clarified. |
| StreamDiffusionV2 | Real-time video-to-video restyle (Wan 1.3B) | High fps on multiple H100s | Code Apache 2.0, checkpoint licences unclear | No reference-image identity: not a character animator. |
| StreamV2V | Real-time video restyle | 4090 real time | **UT Austin research licence (non-commercial)** | Excluded. |
| StreamAvatar, ChatAnyone, LiveAnimate, SoulX-LiveAct, MotionStream, ViSA | Real-time body/avatar research | 20-30 fps, mostly on H100s | **No code/weights released** | Watch only. |

Banned as before: InsightFace, pyvirtualcam, CodeFormer, FLUX.1-dev, XPose.

## Plan

1. **Test Wan-Animate-2 (distilled) on RunPod** (1x H100 80 GB or similar): its examples + our
   character, driven by `samples/talk.mp4`. Measure seconds per frame, VRAM, quality of hands and
   face. This tells us how good "commercial-OK" can look and what the cloud mode would cost.
2. **Watch for Wan-Animate-2-Lite weights.** The pipeline is built so the body model is a
   swappable renderer (same `needs` interface as LivePortrait), so Lite can drop in.
3. Decide on GUAVA only with the owner (paid SMPL-X licence first, then test).
4. Ask the owner for pictures/video once the model is chosen (see the reply of 2026-10-28:
   waist-up, hands visible, same character/outfit/light, 1024+ px, plus a waist-up webcam clip).
