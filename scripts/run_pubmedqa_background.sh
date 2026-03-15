#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-nested}"
CONFIG="${2:-configs/pubmedqa_nested.json}"
PREV_RESULTS="${3:-results_pubmedqa_nested.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[pubmed-bg] config not found: $CONFIG" >&2
  exit 1
fi

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="logs/background/pubmedqa_${MODE}_${TIMESTAMP}"
LOG_PATH="${RUN_DIR}/run.log"
PID_PATH="${RUN_DIR}/run.pid"

mkdir -p "$RUN_DIR"

case "$MODE" in
  nested)
    CMD=(python3 -u scripts/run_pubmedqa.py "$CONFIG")
    ;;
  l2-only)
    if [[ ! -f "$PREV_RESULTS" ]]; then
      echo "[pubmed-bg] previous results not found: $PREV_RESULTS" >&2
      exit 1
    fi
    CMD=(python3 -u scripts/run_pubmedqa_l2_only.py "$CONFIG" "$PREV_RESULTS")
    ;;
  *)
    echo "Usage:" >&2
    echo "  ./scripts/run_pubmedqa_background.sh [nested|l2-only] [config_path] [prev_results_path]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  ./scripts/run_pubmedqa_background.sh" >&2
    echo "  ./scripts/run_pubmedqa_background.sh nested configs/pubmedqa_nested.json" >&2
    echo "  ./scripts/run_pubmedqa_background.sh l2-only configs/pubmedqa_nested.json results_pubmedqa_nested.json" >&2
    exit 1
    ;;
esac

nohup "${CMD[@]}" >"$LOG_PATH" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_PATH"

echo "[pubmed-bg] started"
echo "[pubmed-bg] mode: $MODE"
echo "[pubmed-bg] pid: $PID"
echo "[pubmed-bg] log: $LOG_PATH"
echo "[pubmed-bg] pid file: $PID_PATH"
echo "[pubmed-bg] tail -f $LOG_PATH"
