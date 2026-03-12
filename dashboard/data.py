"""Read-only helpers for the local NOA runtime dashboard."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ACTIVE_RUN_WINDOW_SEC = 60
HEARTBEAT_STALE_SEC = 15
LLM_RATE_WINDOW_SEC = 60


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    current_run: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.current_run.get("status", "unknown"))

    @property
    def updated_at(self) -> datetime | None:
        return parse_iso8601(self.current_run.get("updated_at"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso8601(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    except OSError:
        return []
    return rows


def list_runs(root: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    if not root.exists():
        return runs
    for child in root.iterdir():
        if not child.is_dir():
            continue
        current_run = _read_json(child / "state" / "current_run.json", {})
        if not isinstance(current_run, dict):
            current_run = {}
        run_id = str(current_run.get("run_id") or child.name)
        runs.append(RunRecord(run_id=run_id, run_dir=child, current_run=current_run))
    runs.sort(
        key=lambda item: item.updated_at
        or parse_iso8601(item.current_run.get("started_at"))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return runs


def is_active_run(
    current_run: dict[str, Any],
    *,
    now: datetime | None = None,
    active_window_sec: int = ACTIVE_RUN_WINDOW_SEC,
) -> bool:
    now = now or utc_now()
    if current_run.get("status") != "running":
        return False
    updated_at = parse_iso8601(current_run.get("updated_at"))
    if updated_at is None:
        return False
    return (now - updated_at).total_seconds() <= active_window_sec


def pick_default_run(
    runs: list[RunRecord],
    *,
    now: datetime | None = None,
    active_window_sec: int = ACTIVE_RUN_WINDOW_SEC,
) -> str | None:
    if not runs:
        return None
    now = now or utc_now()
    for run in runs:
        if is_active_run(run.current_run, now=now, active_window_sec=active_window_sec):
            return run.run_id
    return runs[0].run_id


def classify_children(
    children: list[dict[str, Any]],
    heartbeats: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    stale_after_sec: int = HEARTBEAT_STALE_SEC,
    pid_exists: Callable[[int], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now = now or utc_now()
    pid_exists = pid_exists or (lambda _pid: False)
    heartbeat_by_pid = {
        item.get("pid"): item
        for item in heartbeats
        if isinstance(item, dict) and item.get("pid") is not None
    }

    active: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        pid = child.get("pid")
        if not isinstance(pid, int):
            stale.append(child)
            continue

        heartbeat = heartbeat_by_pid.get(pid, {})
        heartbeat_at = parse_iso8601(heartbeat.get("last_heartbeat_at"))
        started_at = parse_iso8601(child.get("started_at"))
        age_anchor = heartbeat_at or started_at
        is_live = bool(pid_exists(pid))
        is_fresh = False
        if age_anchor is not None:
            is_fresh = (now - age_anchor).total_seconds() <= stale_after_sec

        enriched = dict(child)
        if heartbeat:
            enriched["heartbeat"] = heartbeat
        if is_live and is_fresh:
            active.append(enriched)
        else:
            stale.append(enriched)
    return active, stale


def load_llm_entries(run_dir: Path) -> list[dict[str, Any]]:
    llm_dir = run_dir / "logs" / "llm"
    if not llm_dir.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(llm_dir.glob("*.jsonl")):
        entries.extend(_read_jsonl(path))
    return entries


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    values = sorted(values)
    index = (len(values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return round(values[lower], 1)
    weight = index - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 1)


def summarize_llm_entries(
    entries: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    rate_window_sec: int = LLM_RATE_WINDOW_SEC,
) -> dict[str, Any]:
    now = now or utc_now()
    started: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    recent_errors: list[dict[str, Any]] = []
    latencies: list[float] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    window_starts = 0

    for entry in entries:
        request_id = str(entry.get("request_id", ""))
        event = entry.get("event")
        if event == "request_started":
            started[request_id] = entry
            ts = entry.get("timestamp")
            if isinstance(ts, (int, float)):
                entry_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                if (now - entry_time).total_seconds() <= rate_window_sec:
                    window_starts += 1
        elif event == "request_finished":
            finished[request_id] = entry
            latency = entry.get("latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
            usage = entry.get("usage", {}) or {}
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
            if entry.get("ok") is False:
                recent_errors.append(entry)

    inflight = sorted(set(started) - set(finished))
    recent_errors = recent_errors[-10:]
    recent_finished = sorted(
        finished.values(),
        key=lambda item: float(
            started.get(str(item.get("request_id", "")), {}).get("timestamp", 0.0)
        ),
    )[-20:]
    models = sorted(
        {
            str(item.get("resolved_model") or item.get("model"))
            for item in started.values()
            if item.get("resolved_model") or item.get("model")
        }
    )

    return {
        "total_requests": len(started),
        "finished_requests": len(finished),
        "inflight_requests": len(inflight),
        "recent_rate_per_min": window_starts,
        "success_count": sum(1 for item in finished.values() if item.get("ok") is True),
        "error_count": sum(1 for item in finished.values() if item.get("ok") is False),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "recent_errors": recent_errors,
        "models": models,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "recent_finished": recent_finished,
    }


def load_run_events(run_dir: Path) -> list[dict[str, Any]]:
    events = _read_jsonl(run_dir / "state" / "run_events.jsonl")
    events.sort(key=lambda item: item.get("ts", ""))
    return events


def resolve_detail_file(
    run_dir: Path, current_run: dict[str, Any], detail_file: str | None
) -> Path | None:
    if not detail_file:
        return None
    path = Path(detail_file)
    if path.is_absolute():
        return path if path.exists() else None
    source_dir = current_run.get("source_dir")
    if isinstance(source_dir, str) and source_dir:
        candidate = Path(source_dir) / detail_file
        if candidate.exists():
            return candidate
    workspace_root = run_dir / "workspace"
    if workspace_root.exists():
        matches = list(workspace_root.glob(f"**/{detail_file}"))
        if matches:
            return matches[0]
    return None


def load_detail_payload(path: Path | None) -> dict[str, Any] | list[Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_candidate_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for event in events:
        action = event.get("event")
        label = event.get("label") or event.get("candidate_label")
        if not isinstance(label, str) or not label:
            continue
        row = by_label.setdefault(label, {"label": label})
        row["last_ts"] = event.get("ts")
        if action == "checkpoint_candidate":
            row["checkpoint_ok"] = event.get("ok")
            row["ops"] = event.get("ops", [])
            row["rationale"] = event.get("rationale", "")
            row["detail_file"] = event.get("detail_file")
        elif action == "eval_candidate":
            row["before_score"] = event.get("before_score")
            row["after_score"] = event.get("after_score")
            row["accepted"] = event.get("accepted")
            row["error"] = event.get("error")
        elif action == "accept_candidate":
            row["accepted_score"] = event.get("score")
            row["accepted_at"] = event.get("ts")
        elif action == "reject_candidate":
            row["reject_reason"] = event.get("reason")
    rows = list(by_label.values())
    rows.sort(key=lambda item: item.get("last_ts", ""), reverse=True)
    return rows


def timeline_rows(
    events: list[dict[str, Any]], *, limit: int = 100
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events[-limit:]:
        rows.append(
            {
                "ts": event.get("ts"),
                "layer": event.get("layer_id"),
                "step": event.get("step"),
                "event": event.get("event"),
                "label": event.get("label") or event.get("candidate_label"),
                "score": event.get("after_score")
                or event.get("score")
                or event.get("test_score"),
                "status": event.get("ok") if "ok" in event else event.get("accepted"),
            }
        )
    return list(reversed(rows))


def patch_preview_from_ops(ops: list[dict[str, Any]]) -> str:
    previews: list[str] = []
    for op in ops[:8]:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")
        file_path = op.get("file_path", "<unknown>")
        if kind == "update":
            before = str(op.get("search", ""))
            after = str(op.get("replace", ""))
            diff = "\n".join(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=f"{file_path}:before",
                    tofile=f"{file_path}:after",
                    lineterm="",
                )
            )
            previews.append(diff or f"--- {file_path}\n(no visible diff)")
        elif kind in {"create", "overwrite"}:
            content = str(op.get("content", "")) or str(op.get("replace", ""))
            previews.append(f"+++ {file_path}\n{content[:800]}")
        elif kind in {"insert_after", "insert_before"}:
            anchor = str(op.get("anchor", ""))
            new_lines = str(op.get("new_lines", ""))
            previews.append(
                f"*** {file_path}\nanchor:\n{anchor[:400]}\n\ninsert:\n{new_lines[:800]}"
            )
        elif kind == "delete":
            previews.append(f"--- {file_path}\nfile deleted")
        else:
            previews.append(f"*** {file_path}\nunsupported op={kind}")
    return "\n\n".join(previews).strip()


def format_elapsed(started_at: Any, *, now: datetime | None = None) -> str:
    started = parse_iso8601(started_at)
    if started is None:
        return "n/a"
    now = now or utc_now()
    seconds = max(0, int((now - started).total_seconds()))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def build_run_snapshot(
    run_dir: Path,
    *,
    now: datetime | None = None,
    stale_after_sec: int = HEARTBEAT_STALE_SEC,
    active_window_sec: int = ACTIVE_RUN_WINDOW_SEC,
    pid_exists: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    now = now or utc_now()
    current_run = _read_json(run_dir / "state" / "current_run.json", {})
    children = _read_json(run_dir / "state" / "children.json", [])
    heartbeats = _read_json(run_dir / "state" / "heartbeats.json", [])
    events = load_run_events(run_dir)
    llm_entries = load_llm_entries(run_dir)
    active_children, stale_children = classify_children(
        children if isinstance(children, list) else [],
        heartbeats if isinstance(heartbeats, list) else [],
        now=now,
        stale_after_sec=stale_after_sec,
        pid_exists=pid_exists,
    )
    return {
        "run_dir": run_dir,
        "run_id": current_run.get("run_id", run_dir.name),
        "current_run": current_run,
        "is_active": is_active_run(
            current_run,
            now=now,
            active_window_sec=active_window_sec,
        ),
        "active_children": active_children,
        "stale_children": stale_children,
        "events": events,
        "timeline": timeline_rows(events),
        "candidate_rows": build_candidate_rows(events),
        "llm_entries": llm_entries,
        "llm_summary": summarize_llm_entries(llm_entries, now=now),
    }
