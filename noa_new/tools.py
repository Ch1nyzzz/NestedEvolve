"""Two-tool runtime for noa_new."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MINIMAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code directly in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>"


def _payload(proc: subprocess.CompletedProcess[str], *, max_output_chars: int) -> str:
    return json.dumps(
        {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": _trim(proc.stdout or "", max_output_chars),
            "stderr": _trim(proc.stderr or "", max_output_chars),
        },
        ensure_ascii=False,
    )


def run_bash(
    command: str,
    *,
    cwd: str | Path,
    timeout_sec: int = 300,
    max_output_chars: int = 20000,
) -> str:
    """Execute a shell command and return a JSON payload."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/zsh",
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return json.dumps(
            {
                "ok": False,
                "exit_code": None,
                "stdout": _trim(exc.stdout or "", max_output_chars),
                "stderr": _trim(exc.stderr or "", max_output_chars),
                "error": f"Timeout after {timeout_sec}s",
            },
            ensure_ascii=False,
        )
    return _payload(proc, max_output_chars=max_output_chars)


def run_python(
    code: str,
    *,
    cwd: str | Path,
    timeout_sec: int = 600,
    max_output_chars: int = 20000,
) -> str:
    """Execute Python code and return a JSON payload."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return json.dumps(
            {
                "ok": False,
                "exit_code": None,
                "stdout": _trim(exc.stdout or "", max_output_chars),
                "stderr": _trim(exc.stderr or "", max_output_chars),
                "error": f"Timeout after {timeout_sec}s",
            },
            ensure_ascii=False,
        )
    return _payload(proc, max_output_chars=max_output_chars)
