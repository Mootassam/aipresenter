from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from common.config import Settings, deep_merge, default_config_files, load_settings

REPO = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.mode == "local"
    assert s.cloud.max_session_minutes == 35
    assert s.audio.host_api == "wasapi"


def test_repo_default_yaml_matches_code_defaults() -> None:
    from_yaml = load_settings([REPO / "config" / "default.yaml"], missing_ok=False)
    assert from_yaml.model_dump(exclude={"data_dir"}) == Settings().model_dump(exclude={"data_dir"})


def test_yaml_files_merge_in_order(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", "video: {fps: 25, camera_index: 1}\n")
    b = _write(tmp_path / "b.yaml", "video: {fps: 30}\n")
    s = load_settings([a, b])
    assert s.video.fps == 30
    assert s.video.camera_index == 1  # kept from a.yaml


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _write(tmp_path / "c.yaml", "cloud: {max_session_minutes: 20}\n")
    monkeypatch.setenv("AIP_CLOUD__MAX_SESSION_MINUTES", "30")
    assert load_settings([f]).cloud.max_session_minutes == 30


def test_missing_file_ok_or_error(tmp_path: Path) -> None:
    assert load_settings([tmp_path / "nope.yaml"]).mode == "local"
    with pytest.raises(FileNotFoundError):
        load_settings([tmp_path / "nope.yaml"], missing_ok=False)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "video: {source: file}\n",  # file source without a path
        "audio: {source: file}\n",
        "cloud: {max_session_minutes: 10, warn_before_stop_minutes: 10}\n",
        "video: {render_width: 500}\n",  # not a multiple of 8
        "audio: {host_api: mme}\n",  # MME is not allowed
        "unknown_section: 1\n",  # typos are rejected, not silently ignored
        "video: {fpss: 30}\n",
    ],
)
def test_invalid_configs_rejected(tmp_path: Path, yaml_text: str) -> None:
    f = _write(tmp_path / "bad.yaml", yaml_text)
    with pytest.raises(ValidationError):
        load_settings([f])


def test_file_source_with_path_ok(tmp_path: Path) -> None:
    f = _write(tmp_path / "ok.yaml", "video: {source: file, file_path: samples/talk.mp4}\n")
    assert load_settings([f]).video.file_path == Path("samples/talk.mp4")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    f = _write(tmp_path / "list.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_settings([f])


def test_deep_merge_does_not_mutate() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    over = {"a": {"y": 3}, "c": 4}
    out = deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_data_dir_from_env(tmp_path: Path) -> None:
    assert Settings().data_dir == tmp_path / "data"


def test_default_config_files_order() -> None:
    files = default_config_files(REPO)
    assert files[0].name == "default.yaml"
    assert files[1].name == "local.yaml"
