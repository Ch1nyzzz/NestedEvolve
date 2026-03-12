"""LLM 调用封装 — 基于 litellm 的薄封装，支持指数退避重试 + 进程级硬超时。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace

import litellm

from noa.runtime import (
    bind_scope,
    current_scope,
    llm_log_path,
    run_subprocess,
    scope_env,
)

litellm.drop_params = True

_CONSECUTIVE_FAIL_THRESHOLD = 5


class _CircuitBreaker:
    """记录连续失败次数，避免把 provider 抖动误判为正常请求。"""

    def __init__(self, threshold: int = _CONSECUTIVE_FAIL_THRESHOLD):
        self._threshold = threshold
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._consecutive_failures = 0
                return True
            return False


_circuit_breaker = _CircuitBreaker()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

_VT_API_BASE = "https://llm-api.arc.vt.edu/api/v1"
_VT_API_KEY = os.getenv("VT_API_KEY")
_VT_DEFAULT_MODEL = "MiniMax-M2.5"
_TOGETHER_DEFAULT_MODEL = os.getenv("TOGETHER_MODEL", "MiniMaxAI/MiniMax-M2.5")

_ANTHROPIC_MODEL_MAP = {
    "gpt-5-nano": "claude-haiku-4-5-20251001",
    "gpt-4.1-mini": "claude-haiku-4-5-20251001",
    "gpt-4.1": "claude-haiku-4-5-20251001",
}

if LLM_PROVIDER == "openai":
    DEFAULT_MODEL = "gpt-5-nano"
elif LLM_PROVIDER == "vt":
    DEFAULT_MODEL = _VT_DEFAULT_MODEL
elif LLM_PROVIDER == "together":
    DEFAULT_MODEL = _TOGETHER_DEFAULT_MODEL
else:
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


LLM_TIMEOUT_SEC = _env_float("LLM_TIMEOUT_SEC", 120.0)
_HARD_TIMEOUT_SEC = _env_float("LLM_HARD_TIMEOUT_SEC", 300.0)

_DEFAULT_RPM = {"anthropic": 3800, "openai": 3800, "vt": 55, "together": 55}
RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", str(_DEFAULT_RPM.get(LLM_PROVIDER, 55))))


class _RateLimiter:
    """自适应速率限制器：从响应头读取 RPM 并动态更新间隔。"""

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def update_from_response(self, resp) -> None:
        headers = getattr(resp, "_response_headers", None) or {}
        # 优先读动态限制，fallback 到静态限制
        raw = headers.get("x-ratelimit-limit-dynamic") or headers.get(
            "x-ratelimit-limit"
        )
        if not raw:
            return
        try:
            limit = int(raw)
        except (ValueError, TypeError):
            return
        # Together AI 用 reset 窗口（秒）计限制，换算为 RPM
        try:
            reset_sec = int(headers.get("x-ratelimit-reset", 1))
        except (ValueError, TypeError):
            reset_sec = 1
        new_rpm = limit * 60 // max(reset_sec, 1)
        if new_rpm <= 0 or new_rpm == self._rpm:
            return
        with self._lock:
            self._rpm = new_rpm
            self._interval = 60.0 / (new_rpm * 0.9)
        print(f"[LLM] 速率限制自动调整: {new_rpm} RPM (raw={limit}/{reset_sec}s)")


_rate_limiter = _RateLimiter(RPM_LIMIT)


def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
            else:
                total += len(str(item))
        return total
    return len(str(content or ""))


def resolve_model(model: str) -> str:
    """根据 LLM_PROVIDER 将模型名映射到对应的提供商模型。"""
    if LLM_PROVIDER == "vt":
        return f"openai/{_VT_DEFAULT_MODEL}"
    if LLM_PROVIDER == "together":
        return f"together_ai/{_TOGETHER_DEFAULT_MODEL}"
    if LLM_PROVIDER == "anthropic" and model.startswith("gpt"):
        return _ANTHROPIC_MODEL_MAP.get(model, "claude-haiku-4-5-20251001")
    return model


def _vt_kwargs() -> dict:
    """VT ARC API 的额外参数。"""
    if LLM_PROVIDER == "vt":
        key = os.getenv("VT_API_KEY") or _VT_API_KEY
        os.environ["OPENAI_API_KEY"] = key or ""
        os.environ["OPENAI_API_BASE"] = _VT_API_BASE
        return {"api_base": _VT_API_BASE, "api_key": key}
    return {}


@dataclass
class LLMResponse:
    text: str
    usage: dict
    latency_ms: float
    finish_reason: str = "stop"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponseWithTools:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0


def _temp_json_path(prefix: str) -> str:
    scope = current_scope()
    base_dir = (
        os.path.join(scope.run_dir, "state", "llm")
        if scope.run_dir
        else tempfile.gettempdir()
    )
    os.makedirs(base_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=base_dir)
    os.close(fd)
    return path


def _load_worker_payload(response_path: str) -> dict:
    if not os.path.exists(response_path):
        return {}
    try:
        with open(response_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cleanup_temp_paths(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)


def _raise_from_worker_payload(payload: dict, *, model: str) -> None:
    error = payload.get("error", "Unknown LLM worker failure")
    error_type = payload.get("error_type", "runtime_crash")
    if error_type == "timeout":
        raise litellm.Timeout(
            message=error,
            model=model,
            llm_provider=model.split("/")[0],
        )
    raise RuntimeError(f"{error_type}: {error}")


def _compat_response_from_worker(payload: dict):
    tool_calls = []
    for tc in payload.get("tool_calls", []):
        tool_calls.append(
            SimpleNamespace(
                id=tc.get("id", ""),
                function=SimpleNamespace(
                    name=tc.get("name", ""),
                    arguments=json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                ),
            )
        )
    msg = SimpleNamespace(content=payload.get("text"), tool_calls=tool_calls)
    choice = SimpleNamespace(
        message=msg,
        finish_reason=payload.get("finish_reason", "stop"),
    )
    return SimpleNamespace(
        choices=[choice],
        usage=payload.get("usage", {}),
        _hidden_params={
            "additional_headers": payload.get("additional_headers", {}) or {}
        },
        latency_ms=payload.get("latency_ms", 0.0),
    )


def _completion_with_hard_timeout(**kwargs):
    """将单次 litellm.completion 放进独立 worker 子进程。"""
    request_id = uuid.uuid4().hex[:12]
    request_path = _temp_json_path(f"llm_req_{request_id}_")
    response_path = _temp_json_path(f"llm_resp_{request_id}_")
    log_path = llm_log_path(request_id)
    messages = kwargs.get("messages", [])
    request = {
        "request_id": request_id,
        "kwargs": kwargs,
        "meta": {
            "provider": LLM_PROVIDER,
            "resolved_model": kwargs.get("model"),
            "message_count": len(messages),
            "tool_count": len(kwargs.get("tools", []) or []),
            "input_chars": sum(_content_chars(msg.get("content")) for msg in messages),
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        },
    }
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f, ensure_ascii=False, default=str)

    command = [
        os.sys.executable,
        "-m",
        "noa.runtime.llm_worker",
        request_path,
        response_path,
        log_path,
    ]

    try:
        with bind_scope(request_id=request_id):
            proc = run_subprocess(
                command,
                timeout_sec=_HARD_TIMEOUT_SEC,
                kind="llm_worker",
                env=scope_env({"NOA_REQUEST_ID": request_id}),
                log_path=log_path,
                result_path=response_path,
                stream_output=False,
                heartbeat_message="llm request running",
            )
        payload = _load_worker_payload(response_path)
        if proc.timed_out:
            raise litellm.Timeout(
                message=f"Hard timeout ({_HARD_TIMEOUT_SEC}s) - worker killed",
                model=kwargs.get("model", "unknown"),
                llm_provider=kwargs.get("model", "unknown").split("/")[0],
            )
        if not proc.ok and not payload:
            raise RuntimeError(
                proc.error
                or proc.stderr_tail
                or proc.stdout_tail
                or f"llm worker exited with code {proc.returncode}"
            )
        if payload.get("ok") is not True:
            _raise_from_worker_payload(payload, model=kwargs.get("model", "unknown"))
        return _compat_response_from_worker(payload)
    finally:
        _cleanup_temp_paths(request_path, response_path)


def llm_call(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.7,
    system: str | None = None,
) -> LLMResponse:
    """调用 LLM，返回结构化响应。指数退避重试最多 MAX_RETRIES 次。"""
    model = resolve_model(model)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            _rate_limiter.acquire()
            t0 = time.time()
            resp = _completion_with_hard_timeout(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SEC,
                **_vt_kwargs(),
            )
            latency = (time.time() - t0) * 1000
            _rate_limiter.update_from_response(resp)
            _circuit_breaker.record_success()
            return LLMResponse(
                text=(resp.choices[0].message.content or "").strip(),
                usage=dict(resp.usage) if resp.usage else {},
                latency_ms=round(latency, 1),
                finish_reason=getattr(resp.choices[0], "finish_reason", "stop")
                or "stop",
            )
        except Exception as e:
            _circuit_breaker.record_failure()
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"[LLM] retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)


def llm_call_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0,
) -> LLMResponseWithTools:
    """调用 LLM（带 tool calling），返回文本或工具调用列表。"""
    model = resolve_model(model)
    if not tools:
        cleaned = []
        for m in messages:
            if m.get("role") == "tool":
                continue
            if "tool_calls" in m:
                m = {k: v for k, v in m.items() if k != "tool_calls"}
            cleaned.append(m)
        messages = cleaned
    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=LLM_TIMEOUT_SEC,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(MAX_RETRIES):
        try:
            _rate_limiter.acquire()
            t0 = time.time()
            resp = _completion_with_hard_timeout(**kwargs, **_vt_kwargs())
            latency = (time.time() - t0) * 1000
            _rate_limiter.update_from_response(resp)
            _circuit_breaker.record_success()
            msg = resp.choices[0].message

            parsed_tools: list[ToolCall] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    parsed_tools.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        )
                    )

            return LLMResponseWithTools(
                text=msg.content.strip() if msg.content else None,
                tool_calls=parsed_tools,
                usage=dict(resp.usage) if resp.usage else {},
                latency_ms=round(latency, 1),
            )
        except Exception as e:
            _circuit_breaker.record_failure()
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"[LLM] retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
