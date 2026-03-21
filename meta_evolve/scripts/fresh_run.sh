#!/usr/bin/env bash
# fresh_run.sh — refresh skill library 后从零开始跑 skill-evolve
#
# 用法:
#   ./scripts/fresh_run.sh --task circle_packing --iterations 20
#   ./scripts/fresh_run.sh --task circle_packing --iterations 20 --config configs/skill_based.yaml

set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

echo "=== Step 1: Refresh skill library ==="
python3 -m meta_evolve refresh

echo ""
echo "=== Step 2: skill-evolve (fresh) ==="
PYTHONUNBUFFERED=1 exec python3 -m meta_evolve skill-evolve "$@"
