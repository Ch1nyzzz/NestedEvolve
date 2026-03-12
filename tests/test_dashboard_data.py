from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dashboard.data import (
    RunRecord,
    classify_children,
    is_active_run,
    pick_default_run,
)


class DashboardDataTests(unittest.TestCase):
    def test_pick_default_run_prefers_latest_active_run(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        runs = [
            RunRecord(
                run_id="older-completed",
                run_dir=Path("/tmp/older-completed"),
                current_run={
                    "status": "completed",
                    "updated_at": (now - timedelta(seconds=5)).isoformat(),
                },
            ),
            RunRecord(
                run_id="newer-running",
                run_dir=Path("/tmp/newer-running"),
                current_run={
                    "status": "running",
                    "updated_at": (now - timedelta(seconds=10)).isoformat(),
                },
            ),
        ]

        self.assertEqual(
            pick_default_run(runs, now=now),
            "newer-running",
        )

    def test_is_active_run_requires_running_status_and_recent_update(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(
            is_active_run(
                {
                    "status": "running",
                    "updated_at": (now - timedelta(seconds=20)).isoformat(),
                },
                now=now,
                active_window_sec=60,
            )
        )
        self.assertFalse(
            is_active_run(
                {
                    "status": "completed",
                    "updated_at": (now - timedelta(seconds=20)).isoformat(),
                },
                now=now,
                active_window_sec=60,
            )
        )

    def test_classify_children_filters_stale_or_dead_pids(self) -> None:
        now = datetime(2026, 3, 12, 21, 0, 0, tzinfo=timezone.utc)
        children = [
            {"pid": 101, "started_at": (now - timedelta(seconds=4)).isoformat()},
            {"pid": 202, "started_at": (now - timedelta(seconds=40)).isoformat()},
            {"pid": 303, "started_at": (now - timedelta(seconds=4)).isoformat()},
        ]
        heartbeats = [
            {"pid": 101, "last_heartbeat_at": (now - timedelta(seconds=3)).isoformat()},
            {
                "pid": 202,
                "last_heartbeat_at": (now - timedelta(seconds=30)).isoformat(),
            },
            {"pid": 303, "last_heartbeat_at": (now - timedelta(seconds=3)).isoformat()},
        ]

        active, stale = classify_children(
            children,
            heartbeats,
            now=now,
            stale_after_sec=15,
            pid_exists=lambda pid: pid in {101, 202},
        )

        self.assertEqual([item["pid"] for item in active], [101])
        self.assertCountEqual([item["pid"] for item in stale], [202, 303])


if __name__ == "__main__":
    unittest.main()
