"""Shared runtime context, logs, and state files."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fcntl

_SCOPE = contextvars.ContextVar("noa_runtime_scope", default=None)
_LOGGER_CACHE: set[tuple[str, str]] = set()
_LOCK = threading.Lock()

_RUN_ID_ENV = "NOA_RUN_ID"
_RUN_DIR_ENV = "NOA_RUN_DIR"


@dataclass(frozen=True)
class RuntimeScope:
    run_id: str = ""
    run_dir: str = ""
    layer_id: str = ""
    spawn_id: str = ""
    candidate_label: str = ""
    request_id: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str, default: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return default
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)


def current_scope() -> RuntimeScope:
    base = {
        "run_id": os.getenv(_RUN_ID_ENV, ""),
        "run_dir": os.getenv(_RUN_DIR_ENV, ""),
        "layer_id": "",
        "spawn_id": "",
        "candidate_label": "",
        "request_id": "",
    }
    override = _SCOPE.get()
    if override:
        base.update({k: v for k, v in override.items() if v is not None})
    return RuntimeScope(**base)


def _runtime_root(explicit_run_dir: str | None = None) -> Path | None:
    run_dir = explicit_run_dir or current_scope().run_dir
    if not run_dir:
        return None
    return Path(run_dir)


def ensure_runtime_layout(run_dir: str | None = None) -> Path | None:
    root = _runtime_root(run_dir)
    if root is None:
        return None

    for rel in (
        "logs",
        "logs/layers",
        "logs/llm",
        "logs/mini_l1",
        "state",
        "state/llm",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)

    _ensure_json_file(root / "state" / "current_run.json", {})
    _ensure_json_file(root / "state" / "children.json", [])
    _ensure_json_file(root / "state" / "heartbeats.json", [])
    _ensure_text_file(root / "state" / "run_events.jsonl")
    return root


def install_run_context(run_id: str, run_dir: str) -> None:
    os.environ[_RUN_ID_ENV] = run_id
    os.environ[_RUN_DIR_ENV] = run_dir
    ensure_runtime_layout(run_dir)
    update_current_run(
        run_id=run_id,
        run_dir=run_dir,
        pid=os.getpid(),
        status="running",
        started_at=_utc_now(),
        active_layer=None,
        active_spawn_id=None,
    )


def _ensure_json_file(path: Path, default: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_text_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def state_path(name: str) -> str:
    root = ensure_runtime_layout()
    if root is None:
        return os.path.join(tempfile.gettempdir(), name)
    return str(root / "state" / name)


def heartbeat_path() -> str:
    return state_path("heartbeats.json")


def run_events_path() -> str:
    return state_path("run_events.jsonl")


def layer_log_path(layer_id: str) -> str:
    root = ensure_runtime_layout()
    if root is None:
        return os.path.join(
            tempfile.gettempdir(), f"{_safe_name(layer_id, 'layer')}.log"
        )
    return str(root / "logs" / "layers" / f"{_safe_name(layer_id, 'layer')}.log")


def llm_log_path(request_id: str) -> str:
    root = ensure_runtime_layout()
    if root is None:
        return os.path.join(
            tempfile.gettempdir(), f"{_safe_name(request_id, 'llm')}.jsonl"
        )
    return str(root / "logs" / "llm" / f"{_safe_name(request_id, 'llm')}.jsonl")


def mini_l1_log_path(spawn_id: str, attempt: int) -> str:
    root = ensure_runtime_layout()
    safe_spawn = _safe_name(spawn_id, "spawn")
    if root is None:
        return os.path.join(tempfile.gettempdir(), f"{safe_spawn}_{attempt}.log")
    folder = root / "logs" / "mini_l1" / safe_spawn
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / f"attempt_{attempt}.log")


def _lock_path(path: str) -> str:
    return f"{path}.lock"


def _json_update(path: str, default_factory, update_fn) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = _lock_path(path)
    with _LOCK, open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = default_factory()
        else:
            data = default_factory()
        update_fn(data)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp_runtime_", suffix=".json", dir=os.path.dirname(path)
        )
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_current_run(**fields) -> None:
    path = state_path("current_run.json")

    def _mutate(data: dict) -> None:
        data.update({k: v for k, v in fields.items() if v is not None})
        data.setdefault("updated_at", _utc_now())
        data["updated_at"] = _utc_now()

    _json_update(path, dict, _mutate)


def register_child(
    *,
    pid: int,
    kind: str,
    layer_id: str = "",
    spawn_id: str = "",
    request_id: str = "",
    candidate_label: str = "",
    log_path: str = "",
    deadline_at: str = "",
    message: str = "",
) -> None:
    path = state_path("children.json")

    def _mutate(data: list[dict]) -> None:
        data[:] = [item for item in data if item.get("pid") != pid]
        data.append(
            {
                "pid": pid,
                "kind": kind,
                "layer_id": layer_id,
                "spawn_id": spawn_id,
                "request_id": request_id,
                "candidate_label": candidate_label,
                "log_path": log_path,
                "deadline_at": deadline_at,
                "message": message,
                "started_at": _utc_now(),
            }
        )

    _json_update(path, list, _mutate)


def unregister_child(pid: int) -> None:
    path = state_path("children.json")

    def _mutate(data: list[dict]) -> None:
        data[:] = [item for item in data if item.get("pid") != pid]

    _json_update(path, list, _mutate)


def update_heartbeat(pid: int, *, kind: str, phase: str, message: str = "") -> None:
    path = heartbeat_path()

    def _mutate(data: list[dict]) -> None:
        data[:] = [item for item in data if item.get("pid") != pid]
        data.append(
            {
                "pid": pid,
                "kind": kind,
                "phase": phase,
                "message": message,
                "last_heartbeat_at": _utc_now(),
            }
        )

    _json_update(path, list, _mutate)


def clear_heartbeat(pid: int) -> None:
    path = heartbeat_path()

    def _mutate(data: list[dict]) -> None:
        data[:] = [item for item in data if item.get("pid") != pid]

    _json_update(path, list, _mutate)


def append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def record_run_event(event: str, **fields) -> None:
    scope = current_scope()
    payload = {
        "ts": _utc_now(),
        "event": event,
        "run_id": scope.run_id,
        "layer_id": scope.layer_id,
        "spawn_id": scope.spawn_id,
        "candidate_label": scope.candidate_label,
        "request_id": scope.request_id,
    }
    payload.update({k: v for k, v in fields.items() if v is not None})
    append_jsonl(run_events_path(), payload)
    update_current_run(last_event=event)


def scope_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    scope = current_scope()
    env = os.environ.copy()
    if scope.run_id:
        env[_RUN_ID_ENV] = scope.run_id
    if scope.run_dir:
        env[_RUN_DIR_ENV] = scope.run_dir
    if scope.layer_id:
        env["NOA_LAYER_ID"] = scope.layer_id
    if scope.spawn_id:
        env["NOA_SPAWN_ID"] = scope.spawn_id
    if scope.candidate_label:
        env["NOA_CANDIDATE_LABEL"] = scope.candidate_label
    if scope.request_id:
        env["NOA_REQUEST_ID"] = scope.request_id
    if extra:
        env.update(extra)
    return env


@contextlib.contextmanager
def bind_scope(
    *,
    layer_id: str | None = None,
    spawn_id: str | None = None,
    candidate_label: str | None = None,
    request_id: str | None = None,
):
    prev = current_scope()
    payload = asdict(prev)
    if layer_id is not None:
        payload["layer_id"] = layer_id
    if spawn_id is not None:
        payload["spawn_id"] = spawn_id
    if candidate_label is not None:
        payload["candidate_label"] = candidate_label
    if request_id is not None:
        payload["request_id"] = request_id
    token = _SCOPE.set(payload)
    update_current_run(
        active_layer=payload.get("layer_id") or None,
        active_spawn_id=payload.get("spawn_id") or None,
    )
    try:
        yield RuntimeScope(**payload)
    finally:
        _SCOPE.reset(token)
        restored = current_scope()
        update_current_run(
            active_layer=restored.layer_id or None,
            active_spawn_id=restored.spawn_id or None,
        )


def configure_layer_logging(layer_id: str) -> None:
    scope = current_scope()
    if not scope.run_dir:
        return
    log_path = layer_log_path(layer_id)
    key = (layer_id, log_path)
    if key in _LOGGER_CACHE:
        return

    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    for name in ("noa", "noa.unified_agent", "noa.stages", "noa.engine", "utils.llm"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    _LOGGER_CACHE.add(key)
