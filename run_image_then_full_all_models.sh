#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/tyejaedon/PycharmProjects/AI_Compressor"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/models/automation_runs/$TS"
PIPELINE_SEARCH_ROOT="$RUN_ROOT/full_pipeline/search"
PIPELINE_TRAIN_ROOT="$RUN_ROOT/full_pipeline/runs"
PIPELINE_PRODUCTION_ROOT="$RUN_ROOT/full_pipeline/production_bundle"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$PIPELINE_SEARCH_ROOT" "$PIPELINE_TRAIN_ROOT" "$PIPELINE_PRODUCTION_ROOT" "$LOG_DIR"

run_cmd() {
  local cmd="$1"
  echo "\n+ $cmd"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$cmd"
  fi
}

echo "[automation] run_root=$RUN_ROOT"

# Avoid overlapping full-pipeline jobs before launching a clean run.
OLD_PIDS="$(pgrep -f "run_full_production_pipeline.py" | tr '\n' ' ' | sed 's/[[:space:]]*$//' || true)"
if [[ -n "$OLD_PIDS" ]]; then
  echo "[automation] found existing full pipeline pid(s): $OLD_PIDS"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    kill $OLD_PIDS || true
    sleep 1
  fi
fi

PIPELINE_CMD="\"$PYTHON\" \"$ROOT/run_full_production_pipeline.py\" --project-root \"$ROOT\" --python \"$PYTHON\" --resource-profile balanced --include-video --video-max-clips 12000 --trials-image-standard 12 --trials-image-lossy 12 --trials-audio 10 --trials-video 8 --minutes-image-standard 180 --minutes-image-lossy 180 --minutes-audio 180 --minutes-video 240 --search-output-root \"$PIPELINE_SEARCH_ROOT\" --train-output-root \"$PIPELINE_TRAIN_ROOT\" --production-root \"$PIPELINE_PRODUCTION_ROOT\""
run_cmd "$PIPELINE_CMD | tee \"$LOG_DIR/01_full_pipeline.log\""

echo "[automation] complete. outputs under: $RUN_ROOT"

