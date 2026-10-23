"""Typed configuration for the whole application.

Values are resolved in this order (later wins):

1. Defaults defined in the models below.
2. YAML files passed to :func:`load_settings` (e.g. ``config/default.yaml`` then
   ``config/local.yaml``), merged deeply.
3. Environment variables with the ``AIP_`` prefix, nested with ``__``,
   e.g. ``AIP_VIDEO__SOURCE=file`` or ``AIP_CLOUD__MAX_SESSION_MINUTES=20``.

Secrets (RunPod API key, license key) are deliberately NOT part of this config. They are kept
in the OS keychain (see ``app/`` in Phase 5) so they can never leak through a config file or log.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from common.paths import DataPaths, default_data_dir


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VideoConfig(_Section):
    source: Literal["camera", "file"] = "camera"
    """``file`` = mock input from a recorded video (cloud dev box has no webcam)."""
    camera_index: int = Field(0, ge=0)
    file_path: Path | None = None
    loop_file: bool = True
    capture_width: int = Field(1280, ge=160)
    capture_height: int = Field(720, ge=120)
    fps: int = Field(30, ge=5, le=60)
    render_width: int = Field(512, ge=128, multiple_of=8)
    render_height: int = Field(768, ge=128, multiple_of=8)
    output_width: int = Field(1280, ge=160)
    output_height: int = Field(720, ge=120)
    overlap_stages: bool = True
    """Track the next frame while the current one renders (frame rate = slower stage, not sum)."""

    @model_validator(mode="after")
    def _file_needs_path(self) -> Self:
        if self.source == "file" and self.file_path is None:
            raise ValueError("video.file_path is required when video.source is 'file'")
        return self


class AudioConfig(_Section):
    source: Literal["mic", "file"] = "mic"
    input_device: str | None = None
    """Device name substring; ``None`` = system default."""
    output_device: str | None = "CABLE Input"
    """Virtual mic sink. Default matches VB-Audio Virtual Cable."""
    file_path: Path | None = None
    host_api: Literal["wasapi", "asio"] = "wasapi"
    """Never MME: its buffering adds 100+ ms."""
    sample_rate: int = Field(48000, ge=16000, le=96000)
    block_ms: int = Field(20, ge=5, le=200)

    @model_validator(mode="after")
    def _file_needs_path(self) -> Self:
        if self.source == "file" and self.file_path is None:
            raise ValueError("audio.file_path is required when audio.source is 'file'")
        return self


class OneEuroConfig(_Section):
    """One Euro filter tuning: lower min_cutoff = smoother, higher beta = less lag on fast moves."""

    min_cutoff: float = Field(1.0, gt=0)
    beta: float = Field(0.05, ge=0)
    d_cutoff: float = Field(1.0, gt=0)


class TrackingConfig(_Section):
    backend: Literal["mediapipe", "rtmw"] = "mediapipe"
    face: bool = True
    pose: bool = True
    hands: bool = True
    max_hands: int = Field(2, ge=0, le=2)
    pose_model: Literal["lite", "full", "heavy"] = "full"
    min_confidence: float = Field(0.5, ge=0.05, le=0.95)
    max_input_side: int = Field(960, ge=256)
    """Frames are downscaled to this before tracking. The models run at <=256 px internally,
    so bigger inputs only cost colour conversion and copy time."""
    mirror_input: bool = False
    """True if the camera image is already mirrored (selfie view)."""
    parallel: bool = True
    """Run face / pose / hands models in parallel threads."""
    smoothing: OneEuroConfig = OneEuroConfig(min_cutoff=1.0, beta=5.0, d_cutoff=1.0)
    hand_smoothing: OneEuroConfig = OneEuroConfig(min_cutoff=1.5, beta=10.0, d_cutoff=1.0)
    """Fingers move fast; they get less smoothing so gestures don't lag."""
    blendshape_smoothing: OneEuroConfig = OneEuroConfig(min_cutoff=3.0, beta=1.0, d_cutoff=1.0)
    """Expressions (blinks!) must react fast, so only light smoothing."""


class AvatarConfig(_Section):
    renderer: Literal["debug", "liveportrait"] = "debug"
    character_image: Path | None = None
    device: str = "cuda"
    fp16: bool | None = None
    """None = automatic (on for RTX 20xx and newer)."""
    compile: bool = False
    """torch.compile the per-frame networks (needs Triton: Linux / cloud worker)."""
    cuda_graphs: bool = True
    """Record the face renderer as a CUDA graph: far less CPU work per frame (GPU only)."""
    canvas_height: int = Field(768, ge=256, le=2160, multiple_of=2)
    """The character is rendered at this height, then scaled to the output size."""
    pose_strength: float = Field(1.0, ge=0.0, le=2.0)
    expression_strength: float = Field(1.0, ge=0.0, le=2.0)
    stitching: bool = True
    """Blend the animated head into the still shoulders (strongly recommended)."""
    eye_retarget: bool = False
    """Drive blinks from MediaPipe eye openness in addition to LivePortrait's own motion."""
    background: Literal["original", "color", "image"] = "original"
    background_color: str = "#00b140"
    """Used when background is "color" (default: chroma green)."""
    pose_smoothing: OneEuroConfig = OneEuroConfig(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    """Head pose (degrees) + translation + scale."""
    expression_smoothing: OneEuroConfig = OneEuroConfig(min_cutoff=2.5, beta=2.0, d_cutoff=1.0)
    lip_sync: Literal["off", "blend"] = "off"
    """Also drive the mouth from the voice (needs a voice signal: the voice engine, or the
    audio of a test video file). Off until the voice engine exists."""
    lip_sync_strength: float = Field(0.6, ge=0.0, le=1.0)
    """0 = mouth follows the camera only, 1 = mouth follows the voice only."""
    lip_sync_max_ratio: float = Field(0.4, ge=0.1, le=0.8)
    """Lip opening (inner-lip gap / mouth width) for the loudest speech."""
    lip_sync_offset_ms: float = Field(0.0, ge=-500.0, le=500.0)
    """Shift the voice signal against the video (positive = use later audio)."""
    lost_face_ease_s: float = Field(0.5, ge=0.0, le=5.0)
    """When the face disappears, glide back to the neutral pose over this time."""

    @model_validator(mode="after")
    def _needs_image(self) -> Self:
        if self.renderer == "liveportrait" and self.character_image is None:
            raise ValueError("avatar.character_image is required for the liveportrait renderer")
        return self


class OutputConfig(_Section):
    virtual_camera: bool = True
    softcam_dll: Path | None = None
    """Override path to softcam.dll (default: <data_dir>/bin, then third_party/softcam)."""
    preview: bool = True
    """Show a local preview window (dev tool; the desktop app has its own preview)."""
    virtual_mic: bool = True
    ai_label: bool = True
    """Show a small "AI-generated" label on the output (platform disclosure rules)."""
    background_path: Path | None = None


class CloudConfig(_Section):
    provider: Literal["runpod"] = "runpod"
    max_session_minutes: int = Field(35, ge=5, le=240)
    warn_before_stop_minutes: int = Field(5, ge=1)
    connection_lost_grace_seconds: int = Field(60, ge=10, le=600)
    """If the app loses the pod for this long, the pod is terminated so nobody pays for nothing."""

    @model_validator(mode="after")
    def _warn_before_end(self) -> Self:
        if self.warn_before_stop_minutes >= self.max_session_minutes:
            raise ValueError("cloud.warn_before_stop_minutes must be < max_session_minutes")
        return self


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False
    max_file_mb: int = Field(10, ge=1)
    backup_count: int = Field(5, ge=0)


class Settings(BaseSettings):
    """Root config object. Create it with :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="AIP_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_assignment=True,
    )

    mode: Literal["local", "cloud"] = "local"
    data_dir: Path = Field(default_factory=default_data_dir)
    video: VideoConfig = VideoConfig()
    audio: AudioConfig = AudioConfig()
    tracking: TrackingConfig = TrackingConfig()
    avatar: AvatarConfig = AvatarConfig()
    output: OutputConfig = OutputConfig()
    cloud: CloudConfig = CloudConfig()
    logging: LoggingConfig = LoggingConfig()

    @property
    def paths(self) -> DataPaths:
        return DataPaths(self.data_dir)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Env vars beat YAML values (which arrive as init kwargs), so a single
        # AIP_... variable can override a file on the dev box or in the worker container.
        return (env_settings, init_settings)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_settings(files: Iterable[Path | str] = (), *, missing_ok: bool = True) -> Settings:
    """Build :class:`Settings` from YAML files (in order) plus environment variables."""
    merged: dict[str, Any] = {}
    for f in files:
        path = Path(f)
        if not path.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(path)
        merged = deep_merge(merged, read_yaml(path))
    return Settings(**merged)


def default_config_files(project_root: Path | None = None) -> list[Path]:
    """Standard lookup: repo defaults, local overrides, then the user's data-dir config."""
    root = project_root or Path(__file__).resolve().parent.parent
    return [
        root / "config" / "default.yaml",
        root / "config" / "local.yaml",
        default_data_dir() / "config.yaml",
    ]
