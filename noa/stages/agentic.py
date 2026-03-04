"""通用 agentic_loop — 所有需要 tool calling 的组件共用。"""

from __future__ import annotations

import json
import logging
from typing import Callable

from utils.llm import llm_call_with_tools

log = logging.getLogger(__name__)


def agentic_loop(
    *,
    messages: list[dict],
    tools: list[dict],
    tool_executor: Callable[[str, dict], str],
    model: str,
    max_tool_calls: int = 5,
    max_tokens: int = 4096,
    temperature: float = 0,
    json_retries: int = 2,
    budget_exhausted_prompt: str = "Tool call budget exhausted. Output your final answer now as JSON.",
    invalid_json_prompt: str = "Your last response was not valid JSON. Output ONLY valid JSON now.",
    parse_fn: Callable[[str], object | None] | None = None,
    early_stop_fn: Callable[[], bool] | None = None,
    no_tool_call_prompt: str | None = None,
) -> str:
    """通用多轮 tool-calling 循环。

    max_tool_calls=0 时退化为单次 llm_call()（无工具）。
    返回 LLM 的最终文本输出。

    early_stop_fn: 每次循环开头 + 每次 tool 执行后检查，返回 True 则立即退出。
    no_tool_call_prompt: 当 LLM 不调用工具且 parse_fn 失败时，追加此提示继续循环。
                         为 None 时保持现有行为（直接返回）。
    """
    if max_tool_calls <= 0:
        resp = llm_call_with_tools(
            messages,
            tools=[],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.text or ""

    calls_remaining = max_tool_calls
    retries_left = json_retries

    while calls_remaining > 0:
        if early_stop_fn is not None and early_stop_fn():
            break

        resp = llm_call_with_tools(
            messages, tools, model=model, max_tokens=max_tokens, temperature=temperature
        )

        if resp.tool_calls:
            tc_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.text or "",
                    "tool_calls": tc_dicts,
                }
            )

            for tc in resp.tool_calls:
                result_text = tool_executor(tc.name, tc.arguments)
                log.info("[agentic_loop] tool=%s result=%s", tc.name, result_text[:200])
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                )
                calls_remaining -= 1
                if early_stop_fn is not None and early_stop_fn():
                    return resp.text or ""
                if calls_remaining <= 0:
                    break
        else:
            text = resp.text or ""
            if parse_fn is not None:
                parsed = parse_fn(text)
                if parsed is not None:
                    return text
                # parse 失败，尝试重试
                if retries_left <= 0:
                    return text
                retries_left -= 1
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": invalid_json_prompt})
                continue
            # 无 parse_fn 时：如果有 no_tool_call_prompt，追加提示继续循环
            if no_tool_call_prompt is not None:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": no_tool_call_prompt})
                continue
            return text

    # 预算耗尽，强制要求结论
    messages.append({"role": "user", "content": budget_exhausted_prompt})
    resp = llm_call_with_tools(
        messages, tools=[], model=model, max_tokens=max_tokens, temperature=temperature
    )
    text = resp.text or ""

    if parse_fn is not None:
        parsed = parse_fn(text)
        if parsed is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": invalid_json_prompt})
            resp = llm_call_with_tools(
                messages,
                tools=[],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.text or ""

    return text
