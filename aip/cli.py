"""Developer command line: ``aip <command>``.

  aip doctor            hardware check + recommended run mode
  aip config show       print the fully resolved config (YAML + env vars)
  aip config validate   validate one or more YAML files
  aip paths             show and create the data directories
  aip models download   download + verify model files
  aip track             camera/file -> tracking -> overlay or AI avatar -> preview / virtual camera
  aip character check   check a character image and show what to fix

Engine imports happen inside the commands so ``doctor`` etc. work without the heavy extras.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from common import __version__
from common.config import Settings, default_config_files, load_settings
from common.system_info import system_report


def _load(args: argparse.Namespace) -> Settings:
    files = [Path(p) for p in args.config] if args.config else default_config_files()
    return load_settings(files, missing_ok=not args.config)


def cmd_doctor(_: argparse.Namespace) -> int:
    rep = system_report()
    print(f"AI Presenter {__version__}")
    print(f"Python  : {rep['python']}")
    print(f"OS      : {rep['os']}")
    print(f"CPU     : {rep['cpu']}")
    if not rep["gpus"]:
        print("GPU     : none detected")
    for g in rep["gpus"]:
        cc = ".".join(map(str, g["compute_cap"])) if g["compute_cap"] else "?"
        print(
            f"GPU {g['index']}   : {g['name']}  {g['vram_total_mb']} MB  "
            f"driver {g['driver']}  compute {cc}"
        )
    print(f"Mode    : {rep['recommended_mode'].upper()} - {rep['reason']}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    data = json.loads(_load(args).model_dump_json())
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), end="")
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        load_settings([Path(p) for p in args.files], missing_ok=False)
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print(f"INVALID:\n{e}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    paths = _load(args).paths.ensure()
    print(f"root        {paths.root}")
    for d in paths.all_dirs():
        print(f"{d.name:<11} {d}")
    return 0


def cmd_models_download(args: argparse.Namespace) -> int:
    from engine.models import MODELS, ensure_model

    names = args.names or list(MODELS)
    models_dir = _load(args).paths.models
    for name in names:
        path = ensure_model(name, models_dir)
        print(f"OK  {name:<24} {path}")
    return 0


def _parse_size(text: str) -> tuple[int, int]:
    w, _, h = text.lower().partition("x")
    try:
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 1280x720, got {text!r}") from None


def _mouth_track(s: Settings) -> Any:
    """Voice-driven mouth signal from the test video's own audio (until the voice engine)."""
    import cv2

    from engine.avatar.mouth import FileMouthTrack

    if s.video.source != "file" or s.video.file_path is None:
        raise SystemExit("--lip-sync needs --file for now (live voice comes with the voice engine)")
    cap = cv2.VideoCapture(str(s.video.file_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return FileMouthTrack.from_video(
        s.video.file_path, fps, n, offset_ms=s.avatar.lip_sync_offset_ms
    )


def _make_renderer(s: Settings, *, profile: bool = False) -> Any:
    """Debug overlay (Phase 1) or the LivePortrait avatar (Phase 3)."""
    if s.avatar.renderer == "debug":
        from engine.avatar.debug_overlay import DebugOverlayRenderer

        return DebugOverlayRenderer()

    from engine.avatar.character import prepare_character, read_image
    from engine.avatar.liveportrait.network import LivePortraitNet, pick_device
    from engine.avatar.liveportrait.renderer import LivePortraitRenderer

    assert s.avatar.character_image is not None
    char = prepare_character(
        s.avatar.character_image,
        s.paths.models,
        canvas_height=s.avatar.canvas_height,
        with_matte=s.avatar.background != "original",
    )
    bg = None
    if s.avatar.background == "image":
        if s.output.background_path is None:
            raise SystemExit("--bg-image (or output.background_path) is required")
        bg = read_image(s.output.background_path)
    net = LivePortraitNet(
        s.paths.models,
        pick_device(s.avatar.device),
        s.avatar.fp16,
        compile=s.avatar.compile,
        cuda_graphs=s.avatar.cuda_graphs,
    )
    mouth = _mouth_track(s) if s.avatar.lip_sync != "off" else None
    return LivePortraitRenderer(net, char, s.avatar, background=bg, mouth=mouth, profile=profile)


def cmd_character_check(args: argparse.Namespace) -> int:
    from engine.avatar.character import check_quality, detect_faces, read_image

    s = _load(args)
    img = read_image(Path(args.image))
    issues = check_quality(img, detect_faces(img, s.paths.models))
    h, w = img.shape[:2]
    print(f"{args.image}: {w}x{h}")
    if not issues:
        print("OK - this image is a good character image.")
    for i in issues:
        print(f"{'PROBLEM' if i.blocking else 'tip    '}  {i.message}")
    return 1 if any(i.blocking for i in issues) else 0


def cmd_track(args: argparse.Namespace) -> int:
    import logging

    from common.log import setup_logging
    from engine.io.frames import open_source
    from engine.output.video import (
        NullSink,
        PreviewWindow,
        RecorderSink,
        SoftcamNotAvailable,
        SoftcamSink,
        VideoSink,
        find_softcam_dll,
    )
    from engine.pipeline import VideoPipeline
    from engine.tracking.mediapipe_tracker import MediaPipeTracker

    s = _load(args)
    if args.file:
        s.video = s.video.model_copy(update={"source": "file", "file_path": Path(args.file)})
    if args.camera is not None:
        s.video = s.video.model_copy(update={"source": "camera", "camera_index": args.camera})
    if args.pose_model:
        s.tracking.pose_model = args.pose_model
    if args.sequential:
        s.tracking.parallel = False
    if args.character:
        s.avatar = s.avatar.model_copy(
            update={"renderer": "liveportrait", "character_image": Path(args.character)}
        )
    if args.background:
        s.avatar.background = args.background
    if args.bg_image:
        s.output.background_path = Path(args.bg_image)
        s.avatar.background = "image"
    if args.cpu:
        s.avatar.device = "cpu"
    if args.lip_sync:
        s.avatar.lip_sync = "blend"
    if args.lip_strength is not None:
        s.avatar.lip_sync_strength = args.lip_strength
    if args.lip_offset_ms is not None:
        s.avatar.lip_sync_offset_ms = args.lip_offset_ms
    if args.brow_max is not None:
        s.avatar.brow_raise_max = args.brow_max
    if args.compile:
        s.avatar.compile = True
    if args.no_cuda_graphs:
        s.avatar.cuda_graphs = False
    out_w, out_h = args.output_size or (s.video.output_width, s.video.output_height)
    setup_logging(s.logging, s.paths.logs, "track")
    log = logging.getLogger("aip.track")

    renderer = _make_renderer(s, profile=args.profile)
    needs = getattr(renderer, "needs", None)  # None = everything (debug overlay)
    if needs is not None:
        s.tracking.pose = s.tracking.pose and "pose" in needs
        s.tracking.hands = s.tracking.hands and "hands" in needs
    if args.no_overlap:
        s.video.overlap_stages = False

    def on_key(key: str) -> None:
        if key == "c" and hasattr(renderer, "recalibrate"):
            renderer.recalibrate()

    sinks: list[VideoSink] = []
    if s.output.preview and not args.no_preview:
        sinks.append(PreviewWindow(out_w, out_h, on_key=on_key))
    if s.output.virtual_camera and not args.no_vcam:
        dll = find_softcam_dll(s.output.softcam_dll, s.data_dir)
        if dll is None:
            log.warning("softcam.dll not found: virtual camera disabled (see docs/PHASE1.md)")
        else:
            try:
                sinks.append(SoftcamSink(dll, out_w, out_h, s.video.fps))
            except (SoftcamNotAvailable, OSError) as e:
                log.warning("virtual camera disabled: %s", e)
    if args.record:
        sinks.append(RecorderSink(Path(args.record), out_w, out_h, s.video.fps))
    if not sinks:
        sinks.append(NullSink(out_w, out_h))

    source = open_source(s.video, realtime=not args.fast)
    tracker = MediaPipeTracker(s.tracking, s.paths.models)
    pipeline = VideoPipeline(
        source, tracker, renderer, sinks, hud=not args.no_hud, overlap=s.video.overlap_stages
    )
    try:
        report = pipeline.run(
            duration_s=args.duration,
            on_stats=lambda r: log.info(
                "%.1f fps | latency p50 %.0f p95 %.0f ms | rss %.0f MB",
                r.fps,
                r.latency_ms.p50,
                r.latency_ms.p95,
                r.rss_peak_mb,
            ),
        )
    except KeyboardInterrupt:
        return 130
    finally:
        tracker.close()
        for sink in sinks:
            sink.close()

    print(report.pretty())
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0:
            print(f"GPU memory (PyTorch peak): {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip", description="AI Presenter developer tools")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument(
        "-c", "--config", action="append", metavar="FILE", help="YAML config file (repeatable)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check hardware").set_defaults(func=cmd_doctor)
    sub.add_parser("paths", help="show/create data dirs").set_defaults(func=cmd_paths)

    cfg = sub.add_parser("config", help="config tools")
    cfg_sub = cfg.add_subparsers(dest="config_cmd", required=True)
    cfg_sub.add_parser("show", help="print resolved config").set_defaults(func=cmd_config_show)
    val = cfg_sub.add_parser("validate", help="validate YAML files")
    val.add_argument("files", nargs="+")
    val.set_defaults(func=cmd_config_validate)

    models = sub.add_parser("models", help="model files")
    models_sub = models.add_subparsers(dest="models_cmd", required=True)
    dl = models_sub.add_parser("download", help="download + verify models")
    dl.add_argument("names", nargs="*", help="model names (default: all)")
    dl.set_defaults(func=cmd_models_download)

    tr = sub.add_parser("track", help="run tracking with debug overlay")
    src = tr.add_mutually_exclusive_group()
    src.add_argument("--file", help="video file as input (mock camera)")
    src.add_argument("--camera", type=int, help="webcam index")
    tr.add_argument("--duration", type=float, help="stop after N seconds")
    tr.add_argument("--fast", action="store_true", help="file input as fast as possible")
    tr.add_argument("--no-preview", action="store_true")
    tr.add_argument("--no-vcam", action="store_true", help="don't start the virtual camera")
    tr.add_argument("--no-hud", action="store_true")
    tr.add_argument("--output-size", type=_parse_size, metavar="WxH")
    tr.add_argument("--pose-model", choices=["lite", "full", "heavy"])
    tr.add_argument("--sequential", action="store_true", help="run models one after another")
    tr.add_argument("--no-cuda-graphs", action="store_true", help="compare: no CUDA graph")
    tr.add_argument(
        "--no-overlap", action="store_true", help="track and render one after another (compare)"
    )
    tr.add_argument("--report", metavar="JSON", help="write the final report to a file")
    tr.add_argument("--character", metavar="IMAGE", help="animate this character (Phase 3)")
    tr.add_argument("--background", choices=["original", "color", "image"])
    tr.add_argument("--bg-image", metavar="IMAGE", help="background image (implies image mode)")
    tr.add_argument("--cpu", action="store_true", help="run the avatar on the CPU (very slow)")
    tr.add_argument(
        "--lip-sync",
        action="store_true",
        help="also drive the mouth from the video file's voice (test of the lip-sync hook)",
    )
    tr.add_argument("--lip-strength", type=float, help="0 = camera only, 1 = voice only")
    tr.add_argument("--lip-offset-ms", type=float, help="shift the voice against the video")
    tr.add_argument("--brow-max", type=float, help="cap on brow raises (0 = no limit)")
    tr.add_argument("--profile", action="store_true", help="exact per-step GPU timings (slower)")
    tr.add_argument("--record", metavar="MP4", help="also save the output to a video file")
    tr.add_argument("--compile", action="store_true", help="torch.compile the avatar (Linux)")
    tr.set_defaults(func=cmd_track)

    ch = sub.add_parser("character", help="character images")
    ch_sub = ch.add_subparsers(dest="character_cmd", required=True)
    chk = ch_sub.add_parser("check", help="check a character image")
    chk.add_argument("image")
    chk.set_defaults(func=cmd_character_check)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValidationError as e:
        print(f"Config error:\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
