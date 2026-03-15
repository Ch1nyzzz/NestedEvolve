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
DEFAULT_RPM_BY_PROVIDER = {
    "anthropic": 3800,
    "openai": 3800,
    "vt": 55,
    "together": 55,
}


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


def infer_provider_from_model(model: str) -> str | None:
    normalized = (model or "").lower()
    if not normalized:
        return None
    if normalized.startswith("together_ai/") or "minimaxai/" in normalized:
        return "together"
    if normalized.startswith("openai/"):
        return "vt" if "minimax-m2.5" in normalized else "openai"
    if normalized.startswith("gpt-"):
        return "openai"
    if "claude" in normalized or "anthropic" in normalized:
        return "anthropic"
    if normalized == "minimax-m2.5":
        return "vt"
    return None


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
    provider_counts: dict[str, int] = {}
    for item in started.values():
        provider = item.get("provider") or infer_provider_from_model(
            str(item.get("resolved_model") or item.get("model") or "")
        )
        if not provider:
            continue
        provider_counts[str(provider)] = provider_counts.get(str(provider), 0) + 1
    observed_providers = sorted(provider_counts)
    dominant_provider = None
    if provider_counts:
        dominant_provider = max(
            provider_counts.items(), key=lambda pair: (pair[1], pair[0])
        )[0]

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
        "provider_counts": provider_counts,
        "observed_providers": observed_providers,
        "dominant_provider": dominant_provider,
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


def _normalize_eval_split(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"val", "validation", "valid"}:
        return "validation"
    if normalized in {"test", "human_generated_eval"}:
        return "test"
    if normalized == "train":
        return "train"
    return normalized


def _infer_split_from_sample_id(sample_id: Any) -> str | None:
    if not isinstance(sample_id, str):
        return None
    normalized = sample_id.lower()
    if "_train_" in normalized or normalized.startswith("train_"):
        return "train"
    if (
        "_val_" in normalized
        or "_valid_" in normalized
        or normalized.startswith("val_")
    ):
        return "validation"
    if "_test_" in normalized or normalized.startswith("test_"):
        return "test"
    return None


def _infer_eval_split(event: dict[str, Any]) -> str:
    explicit = _normalize_eval_split(event.get("dataset_split") or event.get("split"))
    if explicit:
        return explicit
    event_name = str(event.get("event") or "")
    if event_name == "validation_eval":
        return "validation"
    if event_name in {
        "test_eval_candidate",
        "final_eval_commit",
        "final_eval_no_commit",
    }:
        return "test"
    if event_name in {"eval_candidate", "eval"}:
        return "train"
    return "unknown"


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _trim_text(value: Any, limit: int = 240) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _resolve_eval_detail_file(
    run_dir: Path, current_run: dict[str, Any], event: dict[str, Any]
) -> Path | None:
    detail_file = event.get("detail_file")
    if isinstance(detail_file, str) and detail_file:
        return resolve_detail_file(run_dir, current_run, detail_file)

    if event.get("event") == "eval_candidate":
        label = event.get("label") or event.get("candidate_label")
        if isinstance(label, str) and label:
            source_dir = current_run.get("source_dir")
            if isinstance(source_dir, str) and source_dir:
                candidate = Path(source_dir) / ".noa_meta" / f"eval_{label}.json"
                if candidate.exists():
                    return candidate
    return None


def _build_eval_detail_rows(
    payload: dict[str, Any] | list[Any] | None,
    *,
    default_split: str,
) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        details = payload.get("details", [])
    elif isinstance(payload, list):
        details = payload
    else:
        details = []

    rows: list[dict[str, Any]] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("id") or item.get("index") or "")
        split = (
            _normalize_eval_split(item.get("split"))
            or _infer_split_from_sample_id(sample_id)
            or default_split
        )
        score = _first_number(
            item.get("accuracy"),
            item.get("f1"),
            item.get("raw_score"),
            item.get("score"),
        )
        rows.append(
            {
                "split": split,
                "sample_id": sample_id or "-",
                "question": str(item.get("question") or item.get("prompt") or ""),
                "ground_truth": item.get("ground_truth", item.get("answer")),
                "prediction": str(item.get("prediction") or item.get("output") or ""),
                "prediction_preview": _trim_text(
                    item.get("prediction") or item.get("output") or "", limit=280
                ),
                "extracted": item.get("extracted"),
                "score": round(score, 3) if score is not None else None,
                "accepted": item.get("accepted"),
                "steps": item.get("steps"),
            }
        )
    return rows


def _summarize_eval_details(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in detail_rows:
        split = str(row.get("split") or "unknown")
        score = row.get("score")
        grouped.setdefault(split, [])
        if isinstance(score, (int, float)):
            grouped[split].append(float(score))

    summary: list[dict[str, Any]] = []
    for split, scores in grouped.items():
        avg_score = round(sum(scores) / len(scores), 3) if scores else None
        summary.append(
            {
                "split": split,
                "samples": sum(1 for row in detail_rows if row.get("split") == split),
                "avg_score": avg_score,
            }
        )
    summary.sort(key=lambda item: item["split"])
    return summary


def build_evaluation_rows(
    events: list[dict[str, Any]],
    run_dir: Path,
    current_run: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        event_name = str(event.get("event") or "")
        if event_name not in {
            "baseline_eval",
            "eval_candidate",
            "validation_eval",
            "test_eval_candidate",
            "final_eval_commit",
        }:
            continue

        label = event.get("label") or event.get("candidate_label")
        if not isinstance(label, str) or not label:
            continue

        split = _infer_eval_split(event)
        detail_path = _resolve_eval_detail_file(run_dir, current_run, event)
        detail_payload = load_detail_payload(detail_path)
        detail_rows = _build_eval_detail_rows(detail_payload, default_split=split)
        detail_summary = _summarize_eval_details(detail_rows)
        score = _first_number(
            event.get("after_score"),
            event.get("val_score"),
            event.get("test_score"),
            event.get("score"),
        )
        baseline_score = _first_number(
            event.get("before_score"),
            event.get("baseline_score"),
        )
        n_samples = event.get("n_samples")
        if not isinstance(n_samples, int):
            n_samples = len(detail_rows) or event.get("details_count")

        rows.append(
            {
                "ts": event.get("ts"),
                "event": event_name,
                "split": split,
                "label": label,
                "score": round(score, 2) if score is not None else None,
                "baseline_score": round(baseline_score, 2)
                if baseline_score is not None
                else None,
                "delta": round(score - baseline_score, 2)
                if score is not None and baseline_score is not None
                else None,
                "accepted": event.get("accepted"),
                "error": event.get("error"),
                "rationale": str(event.get("rationale") or ""),
                "ops": event.get("ops", []),
                "n_samples": n_samples,
                "detail_file": event.get("detail_file"),
                "detail_path": detail_path,
                "detail_rows": detail_rows,
                "detail_summary": detail_summary,
                "train_score": _first_number(event.get("train_score")),
                "detail_count": len(detail_rows),
            }
        )

    rows.sort(key=lambda item: item.get("ts", ""), reverse=True)
    return rows


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
    llm_summary = summarize_llm_entries(llm_entries, now=now)
    configured_provider = current_run.get("llm_provider") or llm_summary.get(
        "dominant_provider"
    )
    configured_rpm_limit = current_run.get("llm_rpm_limit")
    if configured_provider and not configured_rpm_limit:
        configured_rpm_limit = DEFAULT_RPM_BY_PROVIDER.get(str(configured_provider))
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
        "evaluation_rows": build_evaluation_rows(events, run_dir, current_run),
        "llm_entries": llm_entries,
        "llm_summary": llm_summary,
        "configured_provider": configured_provider,
        "configured_rpm_limit": configured_rpm_limit,
    }
