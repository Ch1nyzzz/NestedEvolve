"""Process supervision helpers with hard timeouts and state tracking."""

from __future__ import annotations

import collections
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .context import (
    clear_heartbeat,
    current_scope,
    register_child,
    scope_env,
    unregister_child,
    update_heartbeat,
)

_HEARTBEAT_INTERVAL_SEC = 5.0
_TERM_GRACE_SEC = float(os.getenv("NOA_TERM_GRACE_SEC", "5"))


@dataclass
class ManagedProcessResult:
    ok: bool
    returncode: int | None
    error_type: str = ""
    error: str = ""
    timed_out: bool = False
    duration_sec: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    log_path: str = ""
    result_path: str = ""
    payload: Any = None


def _deadline_iso(timeout_sec: float) -> str:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_sec)
    return deadline.isoformat()


def _terminate_pid(pid: int, grace_sec: float = _TERM_GRACE_SEC) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
    end = time.time() + grace_sec
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            return


def _pump_stream(
    stream, sink, tail: collections.deque[str], label: str, enabled: bool
) -> None:
    try:
        for line in iter(stream.readline, ""):
            tail.append(line)
            if sink is not None and enabled:
                sink.write(f"[{label}] {line}")
                sink.flush()
    finally:
        stream.close()


def run_subprocess(
    command: list[str],
    *,
    timeout_sec: float,
    kind: str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    log_path: str = "",
    result_path: str = "",
    stream_output: bool = True,
    heartbeat_message: str = "",
) -> ManagedProcessResult:
    scope = current_scope()
    started = time.time()
    stdout_tail: collections.deque[str] = collections.deque(maxlen=120)
    stderr_tail: collections.deque[str] = collections.deque(maxlen=120)

    sink = (
        open(log_path, "a", encoding="utf-8") if (log_path and stream_output) else None
    )
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env or scope_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register_child(
        pid=proc.pid,
        kind=kind,
        layer_id=scope.layer_id,
        spawn_id=scope.spawn_id,
        request_id=scope.request_id,
        candidate_label=scope.candidate_label,
        log_path=log_path,
        deadline_at=_deadline_iso(timeout_sec),
        message=heartbeat_message,
    )

    threads = [
        threading.Thread(
            target=_pump_stream,
            args=(proc.stdout, sink, stdout_tail, "stdout", stream_output),
            daemon=True,
        ),
        threading.Thread(
            target=_pump_stream,
            args=(proc.stderr, sink, stderr_tail, "stderr", stream_output),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    next_heartbeat = 0.0
    try:
        while True:
            returncode = proc.poll()
            now = time.time()
            if returncode is not None:
                break
            if now - started >= timeout_sec:
                timed_out = True
                _terminate_pid(proc.pid)
                try:
                    proc.wait(timeout=max(_TERM_GRACE_SEC, 1.0))
                except subprocess.TimeoutExpired:
                    pass
                break
            if now >= next_heartbeat:
                update_heartbeat(
                    proc.pid,
                    kind=kind,
                    phase="running",
                    message=heartbeat_message or "subprocess alive",
                )
                next_heartbeat = now + _HEARTBEAT_INTERVAL_SEC
            time.sleep(0.1)
    finally:
        for thread in threads:
            thread.join(timeout=1.0)
        if sink is not None:
            sink.close()
        unregister_child(proc.pid)
        clear_heartbeat(proc.pid)

    duration = round(time.time() - started, 3)
    returncode = proc.returncode
    return ManagedProcessResult(
        ok=(returncode == 0 and not timed_out),
        returncode=returncode,
        error_type="infra_timeout" if timed_out else "",
        error=f"Timed out after {timeout_sec}s" if timed_out else "",
        timed_out=timed_out,
        duration_sec=duration,
        stdout_tail="".join(stdout_tail)[-2000:],
        stderr_tail="".join(stderr_tail)[-2000:],
        log_path=log_path,
        result_path=result_path,
    )


def _fork_entry(
    conn,
    fn: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    env_map: dict[str, str],
    kind: str,
) -> None:
    try:
        os.setsid()
    except Exception:
        pass
    os.environ.update(env_map)
    pid = os.getpid()
    stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            update_heartbeat(
                pid, kind=kind, phase="running", message="forked task alive"
            )

    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()
    try:
        result = fn(*args, **kwargs)
        conn.send({"ok": True, "payload": result})
    except Exception as e:
        conn.send(
            {
                "ok": False,
                "error_type": "runtime_crash",
                "error": str(e)[:500],
                "traceback": traceback.format_exc()[-2000:],
            }
        )
    finally:
        stop.set()
        clear_heartbeat(pid)
        conn.close()


def run_forked(
    fn: Callable,
    *,
    timeout_sec: float,
    kind: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    heartbeat_message: str = "",
) -> ManagedProcessResult:
    kwargs = kwargs or {}
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    env_map = scope_env()
    proc = ctx.Process(
        target=_fork_entry,
        args=(child_conn, fn, args, kwargs, env_map, kind),
        daemon=True,
    )

    scope = current_scope()
    started = time.time()
    proc.start()
    child_conn.close()
    register_child(
        pid=proc.pid,
        kind=kind,
        layer_id=scope.layer_id,
        spawn_id=scope.spawn_id,
        request_id=scope.request_id,
        candidate_label=scope.candidate_label,
        deadline_at=_deadline_iso(timeout_sec),
        message=heartbeat_message,
    )

    timed_out = False
    next_heartbeat = 0.0
    message: dict[str, Any] | None = None
    try:
        while True:
            now = time.time()
            if parent_conn.poll(0.1):
                message = parent_conn.recv()
                break
            if not proc.is_alive():
                break
            if now - started >= timeout_sec:
                timed_out = True
                _terminate_pid(proc.pid)
                proc.join(timeout=max(_TERM_GRACE_SEC, 1.0))
                break
            if now >= next_heartbeat:
                update_heartbeat(
                    proc.pid,
                    kind=kind,
                    phase="running",
                    message=heartbeat_message or "forked task alive",
                )
                next_heartbeat = now + _HEARTBEAT_INTERVAL_SEC
    finally:
        parent_conn.close()
        proc.join(timeout=1.0)
        unregister_child(proc.pid)
        clear_heartbeat(proc.pid)

    duration = round(time.time() - started, 3)
    if timed_out:
        return ManagedProcessResult(
            ok=False,
            returncode=None,
            error_type="infra_timeout",
            error=f"Timed out after {timeout_sec}s",
            timed_out=True,
            duration_sec=duration,
        )
    if message is None:
        return ManagedProcessResult(
            ok=False,
            returncode=proc.exitcode,
            error_type="runtime_crash",
            error=f"Forked task exited unexpectedly (exitcode={proc.exitcode})",
            duration_sec=duration,
        )
    return ManagedProcessResult(
        ok=bool(message.get("ok")),
        returncode=proc.exitcode,
        error_type=message.get("error_type", ""),
        error=message.get("error", ""),
        duration_sec=duration,
        payload=message.get("payload"),
        stderr_tail=message.get("traceback", ""),
    )
