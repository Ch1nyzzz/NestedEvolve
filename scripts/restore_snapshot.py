"""快照查看/导出工具。

用法:
    python scripts/restore_snapshot.py --list                                        # 列出所有 run 和快照
    python scripts/restore_snapshot.py --run <run_id> --label <label> --dest ./out   # 导出快照到指定目录
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def find_runs_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, ".noa_runs")


def list_all(runs_dir: str) -> None:
    if not os.path.isdir(runs_dir):
        print("没有找到任何 run（.noa_runs/ 不存在）")
        return
    for run_id in sorted(os.listdir(runs_dir)):
        snap_dir = os.path.join(runs_dir, run_id, "snapshots")
        labels = sorted(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else []
        print(f"  {run_id}  snapshots: {labels or '(无)'}")


def export_snapshot(runs_dir: str, run_id: str, label: str, dest: str) -> None:
    snap_path = os.path.join(runs_dir, run_id, "snapshots", label)
    if not os.path.isdir(snap_path):
        print(f"快照不存在: {snap_path}", file=sys.stderr)
        sys.exit(1)
    dest = os.path.abspath(dest)
    if os.path.exists(dest):
        print(f"目标目录已存在: {dest}", file=sys.stderr)
        sys.exit(1)
    shutil.copytree(snap_path, dest)
    print(f"已导出到: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NOA 快照查看/导出工具")
    parser.add_argument("--list", action="store_true", help="列出所有 run 和快照")
    parser.add_argument("--run", type=str, help="run_id")
    parser.add_argument("--label", type=str, help="快照标签")
    parser.add_argument("--dest", type=str, help="导出目标目录")
    args = parser.parse_args()

    runs_dir = find_runs_dir()

    if args.list:
        list_all(runs_dir)
    elif args.run and args.label and args.dest:
        export_snapshot(runs_dir, args.run, args.label, args.dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
