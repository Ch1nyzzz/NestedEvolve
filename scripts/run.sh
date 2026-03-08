#!/usr/bin/env bash
# NOA 统一运行脚本
# 用法: ./scripts/run.sh [config_path]
# 默认: configs/hotpotqa_nested.json

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/hotpotqa_nested.json}"
echo "[NOA] config: $CONFIG"
if [[ ! -f "$CONFIG" ]]; then
  echo "[NOA] config not found: $CONFIG" >&2
  exit 1
fi

python3 -u scripts/run_nested.py "$CONFIG"
