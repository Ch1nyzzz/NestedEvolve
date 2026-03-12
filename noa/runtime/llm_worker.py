"""One-shot LiteLLM worker process."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

from noa.runtime.context import append_jsonl

litellm.drop_params = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_error(message: str) -> str:
    lower = message.lower()
    if "timeout" in lower:
        return "timeout"
    if any(
        token in lower
        for token in (
            "connection error",
            "api connection",
            "connection reset",
            "server disconnected",
            "remoteprotocolerror",
            "temporarily unavailable",
            "rate limit",
            "429",
        )
    ):
        return "llm_connection_error"
    return "runtime_crash"


def _serialize_response(resp) -> dict:
    msg = resp.choices[0].message
    tool_calls = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )
    headers = getattr(resp, "_response_headers", None) or {}
    return {
        "ok": True,
        "text": msg.content.strip() if msg.content else "",
        "tool_calls": tool_calls,
        "usage": dict(resp.usage) if resp.usage else {},
        "finish_reason": getattr(resp.choices[0], "finish_reason", "stop") or "stop",
        "additional_headers": headers,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        raise SystemExit(
            "usage: python -m noa.runtime.llm_worker <request.json> <response.json> <log.jsonl>"
        )
    request_path, response_path, log_path = argv[1:]
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    kwargs = request["kwargs"]
    meta = request.get("meta", {})
    request_id = request.get("request_id", "")
    append_jsonl(
        log_path,
        {
            "event": "request_started",
            "request_id": request_id,
            "model": kwargs.get("model"),
            "provider": meta.get("provider"),
            "resolved_model": meta.get("resolved_model"),
            "message_count": meta.get("message_count"),
            "tool_count": meta.get("tool_count"),
            "input_chars": meta.get("input_chars"),
            "max_tokens": meta.get("max_tokens"),
            "temperature": meta.get("temperature"),
            "ts": _utc_now(),
            "timestamp": time.time(),
        },
    )
    started = time.time()
    try:
        resp = litellm.completion(**kwargs)
        payload = _serialize_response(resp)
        payload["latency_ms"] = round((time.time() - started) * 1000, 1)
        append_jsonl(
            log_path,
            {
                "event": "request_finished",
                "request_id": request_id,
                "ok": True,
                "latency_ms": payload["latency_ms"],
                "usage": payload.get("usage", {}),
                "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
                "total_tokens": payload.get("usage", {}).get("total_tokens"),
                "finish_reason": payload.get("finish_reason", "stop"),
                "ts": _utc_now(),
                "timestamp": time.time(),
            },
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e)[:4000],
            "error_type": _classify_error(str(e)),
            "latency_ms": round((time.time() - started) * 1000, 1),
        }
        append_jsonl(
            log_path,
            {
                "event": "request_finished",
                "request_id": request_id,
                "ok": False,
                "latency_ms": payload["latency_ms"],
                "usage": {},
                "error_type": payload["error_type"],
                "error": payload["error"][:500],
                "ts": _utc_now(),
                "timestamp": time.time(),
            },
        )
    Path(response_path).write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
