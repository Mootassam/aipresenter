#!/usr/bin/env bash
# Phase 3 avatar benchmark on a RunPod GPU pod.
#
#   bash tools/runpod_bench.sh [AUTO_STOP_MINUTES]
#
# Safety first: the pod stops itself after AUTO_STOP_MINUTES (default 60) even if nobody
# remembers. Results land in results/ (speed reports, VRAM log, avatar.mp4).
set -euo pipefail
cd "$(dirname "$0")/.."

AUTO_STOP_MIN="${1:-60}"
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  nohup bash -c "sleep $((AUTO_STOP_MIN * 60)); runpodctl stop pod $RUNPOD_POD_ID" \
    >/tmp/autostop.log 2>&1 &
  echo ">>> Safety timer: this pod stops itself in ${AUTO_STOP_MIN} minutes."
fi

echo ">>> Installing system libraries"
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 libegl1 libgles2 >/dev/null

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
export AIP_DATA_DIR=/workspace/aip-data
export UV_LINK_MODE=copy

echo ">>> Installing Python packages (first time: a few minutes)"
uv sync --extra tracking --extra avatar -q
uv run aip doctor
uv run aip models download >/dev/null

mkdir -p results
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > results/vram.csv &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

COMMON=(--file samples/talk.mp4 --character samples/character.png --no-preview --no-vcam
        --output-size 720x1280)

echo ">>> 1/4 Real-time test (like a live webcam, 60 s)"
uv run aip track "${COMMON[@]}" --duration 60 --report results/realtime.json | tee results/realtime.txt

echo ">>> 2/4 Per-step timings (30 s)"
uv run aip track "${COMMON[@]}" --fast --profile --duration 30 --report results/profile.json \
  | tee results/profile.txt

echo ">>> 3/4 Comparison: tracking and rendering one after another (the old way, 30 s)"
uv run aip track "${COMMON[@]}" --no-overlap --duration 30 --report results/serial.json \
  | tee results/serial.txt

echo ">>> 4/4 Quality video: every frame of the sample, side by side"
AIP_VIDEO__LOOP_FILE=false uv run aip track "${COMMON[@]}" --fast --no-hud \
  --record results/avatar.mp4 --report results/record.json | tail -3

kill $SMI_PID 2>/dev/null || true
echo ">>> Peak VRAM (MiB): $(cut -d, -f2 results/vram.csv | tail -n +2 | tr -dc '0-9\n' | sort -n | tail -1)"
tar czf results.tgz results
echo ">>> DONE. Results in results.tgz"
