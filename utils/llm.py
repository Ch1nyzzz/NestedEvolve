"""LLM 调用封装 — 基于 litellm 的薄封装，支持指数退避重试。"""

from __future__ import annotations

import json
import os
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


def resolve_model(model: str) -> str:
    """根据 LLM_PROVIDER 将模型名映射到对应的提供商模型。"""
    if LLM_PROVIDER == "vt":
        return f"openai/{_VT_DEFAULT_MODEL}"
    if LLM_PROVIDER == "anthropic" and model.startswith("gpt"):
        return _ANTHROPIC_MODEL_MAP.get(model, "claude-haiku-4-5-20251001")
    return model


def _vt_kwargs() -> dict:
    """VT ARC API 的额外参数。"""
    if LLM_PROVIDER == "vt":
        return {"api_base": _VT_API_BASE, "api_key": _VT_API_KEY}
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
            t0 = time.time()
            resp = litellm.completion(**kwargs, **_vt_kwargs())
            latency = (time.time() - t0) * 1000
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
