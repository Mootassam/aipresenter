# AI Presenter

Desktop app that turns a creator's **webcam + microphone** into a **live AI presenter**: an AI
character that copies your face, body and hands in real time, with a real-time voice changer.
The output appears as a virtual camera + virtual microphone for OBS, TikTok Live Studio,
YouTube, Zoom, Discord and others.

Two run modes:

- **Local**: everything runs on your own NVIDIA GPU (target: RTX 5050 8 GB).
- **Cloud (bring your own GPU)**: the app rents a GPU on *your* RunPod account, streams through
  it, and always shuts it down at the end.

> Status: **Phase 1 (tracking + virtual camera) done.** How to run it: [docs/PHASE1.md](docs/PHASE1.md).
> See [docs/PLAN.md](docs/PLAN.md) for the roadmap and
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit together.

## Repository layout

| Folder | What it is |
|---|---|
| `common/` | Config, logging, paths, timing, hardware detection, `aip` CLI |
| `engine/` | Real-time core: input → tracking → avatar / voice → output |
| `app/` | PySide6 desktop app |
| `worker/` | Cloud worker that runs inside the RunPod pod |
| `cloud/` | GPU provider integration (RunPod, later Vast.ai), cost counter, auto-shutdown |
| `license_server/` | FastAPI + PostgreSQL license backend |
| `training/` | Character and voice training pipelines |
| `config/` | `default.yaml` (in git) and `local.yaml` (your overrides, git-ignored) |
| `tests/` | Tests for everything that doesn't need a GPU |
| `docs/` | Plan, architecture, licenses, guides |

## Developer setup (Windows)

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 automatically).

```powershell
git clone <repo-url> ai-presenter
cd ai-presenter
uv sync                 # create .venv with Python 3.11 + dev tools
uv run pytest           # run the tests
uv run aip doctor       # check your GPU and recommended mode
uv run aip config show  # print the resolved config
```

Feature groups are installed when a phase needs them, e.g. `uv sync --extra tracking`.

## Configuration

Values come from `config/default.yaml`, then `config/local.yaml`, then
`%LOCALAPPDATA%\AIPresenter\config.yaml`, and finally environment variables:

```powershell
$env:AIP_VIDEO__SOURCE = "file"
$env:AIP_VIDEO__FILE_PATH = "samples/talk.mp4"
uv run aip config show
```

Unknown keys are rejected, so typos fail loudly. **Secrets never go in config files.** API keys
and license keys are stored in the Windows credential store.

## Quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

GPU tests are marked `@pytest.mark.gpu` and skipped by default. Run them on a GPU machine with
`uv run pytest -m gpu`.

## License

Proprietary. Third-party components and their licenses: see [docs/PLAN.md](docs/PLAN.md#3-license-table-commercial-use).
