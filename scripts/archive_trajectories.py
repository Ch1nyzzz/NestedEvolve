"""Archive and reset trajectory cache directory."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil


def archive_and_reset_trajectory_dir(project_root: str, once: bool = True) -> dict:
    root = Path(project_root).resolve()
    cache_dir = root / ".noa_cache"
    trajectories_dir = cache_dir / "trajectories"
    archive_root = cache_dir / "trajectories_archive"
    marker_path = archive_root / ".initial_cleanup_done"

    archive_root.mkdir(parents=True, exist_ok=True)

    if once and marker_path.exists():
        return {
            "ok": True,
            "skipped": True,
            "reason": "cleanup already completed once",
            "archive_root": str(archive_root),
        }

    archived_path = None
    if trajectories_dir.exists():
        has_entries = any(trajectories_dir.iterdir())
        if has_entries:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = archive_root / f"trajectories_{ts}"
            suffix = 1
            while target.exists():
                suffix += 1
                target = archive_root / f"trajectories_{ts}_{suffix}"
            shutil.move(str(trajectories_dir), str(target))
            archived_path = str(target)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    if once:
        marker_path.write_text(datetime.now().isoformat(), encoding="utf-8")

    return {
        "ok": True,
        "skipped": False,
        "archive_root": str(archive_root),
        "archived_path": archived_path,
        "active_trajectories_dir": str(trajectories_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive and reset .noa_cache/trajectories"
    )
    parser.add_argument(
        "--project-root", type=str, default=".", help="Project root path"
    )
    parser.add_argument(
        "--once", action="store_true", default=True, help="Run cleanup only once"
    )
    parser.add_argument(
        "--no-once", action="store_true", help="Disable once-only behavior"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    once = False if args.no_once else bool(args.once)
    result = archive_and_reset_trajectory_dir(args.project_root, once=once)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
