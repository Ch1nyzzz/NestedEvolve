"""LLM 调用封装 — 基于 litellm 的薄封装，支持指数退避重试。"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field

import litellm

litellm.drop_params = True

# 从环境变量读取 provider，默认 anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()

# VT ARC API 配置
_VT_API_BASE = "https://llm-api.arc.vt.edu/api/v1"
_VT_API_KEY = os.getenv("VT_API_KEY")
_VT_DEFAULT_MODEL = "Kimi-K2.5"

# Together AI 配置（litellm 原生支持，只需 TOGETHER_API_KEY 环境变量）
_TOGETHER_DEFAULT_MODEL = os.getenv("TOGETHER_MODEL", "moonshotai/Kimi-K2.5")

# 模型映射：openai 模型名 → anthropic 模型名
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

# 全局速率限制器 — 令牌桶算法，线程安全
# 按 provider 自动选择默认 RPM（可通过 LLM_RPM_LIMIT 覆盖）
_DEFAULT_RPM = {"anthropic": 3800, "openai": 3800, "vt": 55, "together": 55}
RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", str(_DEFAULT_RPM.get(LLM_PROVIDER, 55))))


class _RateLimiter:
    """自适应速率限制器：从响应头读取实际 RPM 上限，动态调整间隔。"""

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def update_from_response(self, resp):
        """从 litellm 响应头中提取动态 RPM 并更新间隔。"""
        headers = getattr(resp, "_hidden_params", {}).get("additional_headers", {})
        # 优先动态限制，回退到基础限制
        raw = headers.get("x-ratelimit-limit-dynamic") or headers.get(
            "x-ratelimit-limit"
        )
        if not raw:
            return
        try:
            new_rpm = int(raw)
        except (ValueError, TypeError):
            return
        if new_rpm <= 0 or new_rpm == self._rpm:
            return
        with self._lock:
            self._rpm = new_rpm
            self._interval = 60.0 / (new_rpm * 0.9)  # 留 10% 余量
        print(f"[LLM] 速率限制自动调整: {new_rpm} RPM")


_rate_limiter = _RateLimiter(RPM_LIMIT)


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
    """VT ARC API 的额外参数。延迟读取 key 以确保 dotenv 已加载。"""
    if LLM_PROVIDER == "vt":
        key = os.getenv("VT_API_KEY") or _VT_API_KEY
        # litellm 对 openai/ 前缀会读 OPENAI_API_KEY 环境变量，需覆盖
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
            resp = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SEC,
                **_vt_kwargs(),
            )
            latency = (time.time() - t0) * 1000
            _rate_limiter.update_from_response(resp)
            return LLMResponse(
                text=resp.choices[0].message.content.strip(),
                usage=dict(resp.usage) if resp.usage else {},
                latency_ms=round(latency, 1),
                finish_reason=getattr(resp.choices[0], "finish_reason", "stop")
                or "stop",
            )
        except Exception as e:
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
    # 无 tools 时，清理 messages 中的 tool_calls / tool role（Anthropic 不允许无 tools 定义时出现）
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
            resp = litellm.completion(**kwargs, **_vt_kwargs())
            latency = (time.time() - t0) * 1000
            _rate_limiter.update_from_response(resp)
            msg = resp.choices[0].message

            # 解析 tool_calls
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
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2**attempt
            print(f"[LLM] retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
