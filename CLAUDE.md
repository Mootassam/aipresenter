# Notes for Claude Code sessions working on this repo

Several Claude Code sessions (the owner's and a teammate's) may work on this repo. Read this first.

**When resuming work, read `docs/PROGRESS.md` first.** It has the current status, open
questions and decisions already made. Update it at the end of every phase or session.

## Project
Commercial desktop app "AI Presenter": webcam + mic → live AI character video + voice changer
→ virtual camera/mic for streaming. Roadmap and decisions: `docs/PLAN.md`. Design:
`docs/ARCHITECTURE.md`.

## Hard rules
- **Commercial licenses only.** Before adding any library or model, check its license and add it
  to the table in `docs/PLAN.md`. Banned: InsightFace models, pyvirtualcam (GPL), CodeFormer,
  FLUX.1-dev, XPose, anything "non-commercial" or "research only".
- **No secrets in code, config or logs.** API keys and license keys go in the OS keychain.
- **Cloud pods must always be stopped.** Any code that creates a pod must stop it on every exit
  path: success, error, timeout, crash recovery.
- `engine/` must not import `app/` or Qt (it runs headless in the cloud worker).
- Real-time loops: no per-frame allocations that grow, always use the newest frame (drop old
  ones), measure with `common.timing`.

## Workflow
- Python 3.11 via `uv`. Before every commit: `uv run ruff format . && uv run ruff check . &&
  uv run mypy && uv run pytest`.
- Tests for all non-GPU logic. GPU tests get `@pytest.mark.gpu`.
- Work on a branch per task (`phase1-tracking`, `phase2-voice`, …), open a PR, and don't push
  to `main` directly.
- Typed code, docstrings on public functions, comments only where the *why* isn't obvious.
