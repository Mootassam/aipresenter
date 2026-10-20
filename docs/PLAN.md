# AI Presenter — Plan, Tech Choices and License Review

Status: **waiting for owner approval before Phase 0.**
Date: 2026-10-20

---

## 1. Honest feasibility check (read first)

| Goal | Verdict | Notes |
|---|---|---|
| Tracking (face + pose + both hands) at 30 fps | Achievable | MediaPipe on CPU, or RTMW on the GPU via ONNX. |
| Voice changer < 100 ms | Achievable | RVC + RMVPE/FCPE with ONNX. Realistic target is 80–150 ms depending on block size. Voice runs in its **own process**, so a slow video frame can never make the audio stutter. |
| Head + shoulders realistic avatar, 25–30 fps on RTX 5050 8 GB | Probably achievable | LivePortrait with TensorRT FP16 runs about 20–30 fps on this class of GPU. Needs measuring. |
| Realistic **upper body + hands** in real time on 8 GB | **Not achievable today with open-source models** | Current open upper-body/hand models are diffusion-based and offline, or seconds per frame. Plan: v2 runs in **cloud mode first** (4090/5090), keeps the head realistic, and protects hand quality with a hybrid approach (see Phase 4). Local 8 GB upper body = long-term custom model. |
| Video + tracking + voice inside 8 GB VRAM | Tight but possible for head + shoulders | Budget: tracking on CPU (0 GB), RVC ~1.5 GB, LivePortrait TRT ~2.5–3.5 GB, buffers ~1 GB. |
| 35-minute streams without drift or memory growth | Achievable | Needs a soak test in CI (Phase 1 onward) with the mock input. |

**This dev machine** has a GTX 1050 (4 GB) and Python 3.12. It is fine for tracking, the app UI, tests, the voice changer (slowly) and the cloud mode. Avatar benchmarks must run on your RTX 5050 or a RunPod GPU.

---

## 2. Final tech choices (changes from your proposal are in **bold**)

| Area | Choice | Why |
|---|---|---|
| Language | Python 3.11 (engine, worker), PySide6 GUI | As proposed. 3.11 has the best wheel support for TensorRT/ONNX/MediaPipe. |
| Process model | **3 processes: GUI, video engine, audio engine, linked by shared memory + a small IPC bus** | The GIL and GPU stalls must not reach the audio. Also isolates crashes. |
| Tracking | MediaPipe Face Landmarker + Pose + Hand Landmarker, One Euro filter. **RTMW (ONNX, GPU) as a pluggable alternative** | MediaPipe has no GPU delegate in Python on Windows, so its CPU speed gets benchmarked. RTMW is the fallback if hands are not accurate enough. |
| Avatar v1 | LivePortrait (TensorRT FP16), **with InsightFace replaced by MediaPipe** | LivePortrait's default face detector (InsightFace) is non-commercial. |
| Avatar v2 | Research track: PersonaLive / TalkingPose-style pose-guided model, cloud-first; later our own model | See the feasibility check above. |
| Upscale | **Real-ESRGAN (BSD)** or a light 2x SR model | Avoid CodeFormer (non-commercial). |
| Voice | RVC v2 inference (ONNX), ContentVec, RMVPE / FCPE, `sounddevice` WASAPI | As proposed. |
| Virtual camera | **softcam (MIT) or our own DirectShow filter** instead of `pyvirtualcam` | `pyvirtualcam` is **GPL-2.0**, which is a problem for a closed-source commercial app. |
| Virtual mic | VB-Audio Cable installed **by the user** at first, **a licensed redistribution or our own driver later** | VB-Cable may not be bundled in a commercial installer without a license from VB-Audio. |
| Cloud transport | aiortc (WebRTC), optional RTMP push from the pod | As proposed. |
| Cloud provider | RunPod GraphQL API behind a `GpuProvider` interface (Vast.ai later) | As proposed. |
| License server | FastAPI + PostgreSQL, pluggable payment webhooks (Paddle / Lemon Squeezy) | As proposed. |
| Secrets | Windows DPAPI via `keyring` | As proposed. |
| Packaging | **Nuitka standalone (not onefile) + Inno Setup installer** | Better protection than PyInstaller. A non-onefile build keeps PySide6 (LGPL) relinkable, which LGPL requires. |
| Model protection | AES-GCM encrypted weights, decrypted into memory only, key released after the license check | As proposed. Note: no client-side protection is unbreakable. This raises the cost of copying; it doesn't make copying impossible. |

---

## 3. License table (commercial use)

Legend: ✅ OK · ⚠️ OK with conditions · ❌ blocks commercial use, must avoid or replace · ❓ verify before depending on it

| Component | License | Status | Action |
|---|---|---|---|
| MediaPipe (code + Face/Pose/Hand models) | Apache-2.0 | ✅ | — |
| RTMPose / RTMW (MMPose) | Apache-2.0 (code); weights trained on COCO-WholeBody etc. | ⚠️ | Check dataset terms of the exact checkpoint. |
| LivePortrait (code + weights) | MIT | ✅ | — |
| InsightFace models (used by LivePortrait for face detection) | Non-commercial research only | ❌ | Replace with MediaPipe face detection/landmarks. |
| XPose (LivePortrait animal mode) | Non-commercial | ❌ | Don't use. |
| PersonaLive | Reported Apache-2.0; base model weights may carry their own license (e.g. SD 1.5 = CreativeML OpenRAIL-M) | ❓ | Verify repo + every weight file. |
| ChatAnyone | Paper; no usable public code/weights as far as known | ❓ | Research reference only. |
| TalkingPose | Unknown | ❓ | Verify. |
| Stable Diffusion 1.5 / SDXL (if used for character fine-tune) | CreativeML OpenRAIL-M / OpenRAIL++ | ⚠️ | Commercial OK; use restrictions must be passed on in our ToS. |
| FLUX.1-dev | Non-commercial | ❌ | Don't use (FLUX.1-schnell is Apache-2.0 if needed). |
| CodeFormer | S-Lab non-commercial | ❌ | Don't use. |
| Real-ESRGAN | BSD-3 | ✅ | — |
| RVC WebUI code + pretrained v2 base models | MIT | ✅ | — |
| ContentVec / HuBERT base | MIT | ✅ | — |
| RMVPE | MIT (verify the weight source) | ⚠️ | Confirm the license of the weight file. |
| FCPE (torchfcpe) | MIT | ✅ | — |
| Customer-imported `.pth` voice models | Customer's responsibility | ⚠️ | Covered by the consent checkbox + ToS. |
| PyTorch | BSD-3 | ✅ | — |
| ONNX Runtime | MIT | ✅ | — |
| TensorRT / CUDA runtime | NVIDIA SLA (redistribution allowed for listed files) | ⚠️ | Ship only the redistributable DLLs; keep NOTICE files. |
| sounddevice / PortAudio | MIT | ✅ | — |
| pyvirtualcam | GPL-2.0 | ❌ | Replace with softcam (MIT) or our own filter. |
| OBS Virtual Camera | GPL-2.0 (separate program) | ⚠️ | OK to *use* as a separately installed program; don't bundle or link. |
| VB-Audio Virtual Cable | Donationware; commercial bundling needs a license | ⚠️ | The user installs it, or we buy a license from VB-Audio. |
| PySide6 / Qt | LGPL-3.0 | ⚠️ | Dynamic linking, relinkable build, LGPL notice in the app. |
| aiortc | BSD-3 | ✅ | — |
| LiveKit (alternative) | Apache-2.0 | ✅ | — |
| FastAPI / Uvicorn | MIT / BSD | ✅ | — |
| PostgreSQL | PostgreSQL License | ✅ | — |
| Nuitka | Apache-2.0 (output not restricted) | ✅ | — |
| Inno Setup | Free incl. commercial | ✅ | — |
| keyring | MIT | ✅ | — |

**Problems to know about:** InsightFace, pyvirtualcam, CodeFormer, FLUX.1-dev and XPose are excluded. VB-Cable needs a decision. PersonaLive and TalkingPose must be verified before Phase 4.

*This is an engineering review, not legal advice. Have a lawyer check the final list before launch.*

---

## 4. Repo structure (Phase 0)

```
ai-presenter/
  engine/           # shared real-time core
    io/             #   camera, video-file/mock input, audio I/O
    tracking/       #   face, pose, hands, One Euro filter
    avatar/         #   renderers: debug overlay, LivePortrait, (lite 3D)
    voice/          #   RVC pipeline, noise gate, pitch
    output/         #   virtual cam, virtual mic, RTMP
    compositing/    #   background
    bench/          #   fps / latency / VRAM benchmark
  app/              # PySide6 desktop app
  worker/           # cloud worker (Docker, WebRTC server)
  cloud/            # GpuProvider interface, RunPod client, cost tracker
  license_server/   # FastAPI + PostgreSQL
  training/         # character fine-tune, voice training
  common/           # config, logging, IPC, crypto
  tests/
  docs/             # ARCHITECTURE.md, PLAN.md, LICENSES.md, USER_GUIDE.md
```

---

## 5. Phase plan with "done" criteria

| Phase | Done when |
|---|---|
| 0 Setup | Repo, config (pydantic-settings + YAML), logging, CI tests green, ARCHITECTURE.md with diagram |
| 1 Tracking + vcam | Webcam/video file → landmarks → overlay → virtual camera; benchmark prints fps/latency; 35-minute soak test with no memory growth |
| 2 Voice | Mic/file → RVC → virtual mic, `.pth` → ONNX import, pitch, gate, presets, measured latency |
| 3 Avatar v1 | 1 image → head + shoulders animated by tracking, background composite, numbers on RTX 5050 |
| 4 Avatar v2 | Upper body + hands prototype (cloud GPU), multi-image reference, written weakness report |
| 5 App | All screens working against the real engine |
| 6 Cloud | Plans list, launch/stop pod, WebRTC, cost counter, auto-stop + "always stop on error" tests |
| 7 Training | Voice training + character Pro mode on the customer's GPU, encrypted results |
| 8 License | Server, activation, webhooks, model encryption, checks in app + worker |
| 9 Packaging | Installer, first-run wizard, user guide, crash reporting |

---

## 6. Decisions needed from the owner

1. Approve the tech changes: softcam instead of pyvirtualcam, Nuitka, the 3-process model.
2. VB-Cable: user installs it for now (recommended), or buy a redistribution license.
3. Accept that realistic upper body + hands is **cloud-first**, with local 8 GB support as a later goal.
4. Python 3.11 for the project (this machine has 3.12; I'll create a 3.11 virtual environment).
