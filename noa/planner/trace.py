"""Planner step trace persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime

from noa.planner.protocol import PlannerStepRecord


class PlannerTraceWriter:
    def __init__(self, project_root: str, layer: str, run_tag: str = "default"):
        trace_dir = os.path.join(project_root, ".noa_cache", "planner_traces")
        os.makedirs(trace_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(trace_dir, f"{layer}_{run_tag}_{ts}.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: PlannerStepRecord) -> None:
        self._fh.write(
            json.dumps(record.to_dict(), ensure_ascii=False, default=_json_default)
            + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
