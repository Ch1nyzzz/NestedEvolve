#!/usr/bin/env bash
# NOA 统一运行脚本
# 用法: ./scripts/run.sh [config_path]
# 默认: configs/hotpotqa_default.json

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/hotpotqa_nested.json}"
echo "[NOA] config: $CONFIG"
python3 scripts/run_nested.py "$CONFIG"
