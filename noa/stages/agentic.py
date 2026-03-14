"""通用 agentic_loop — 所有需要 tool calling 的组件共用，内置墙钟超时 + 连续错误熔断。"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from utils.llm import llm_call_with_tools

log = logging.getLogger(__name__)

# 默认墙钟超时 (秒)，0 表示不限制。改为 4h (从 600s)
_DEFAULT_WALL_TIMEOUT = 14400
# 连续 LLM 调用失败 N 次后放弃当前 loop
_MAX_CONSECUTIVE_ERRORS = 5


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
    stats: dict | None = None,
    wall_timeout_sec: float = _DEFAULT_WALL_TIMEOUT,
    compact_fn: Callable[[list[dict]], list[dict]] | None = None,
    compact_threshold: int = 0,
) -> str:
    """通用多轮 tool-calling 循环。

    max_tool_calls=0 时退化为单次 llm_call()（无工具）。
    返回 LLM 的最终文本输出。

    early_stop_fn: 每次循环开头 + 每次 tool 执行后检查，返回 True 则立即退出。
    no_tool_call_prompt: 当 LLM 不调用工具且 parse_fn 失败时，追加此提示继续循环。
                         为 None 时保持现有行为（直接返回）。
    wall_timeout_sec: 墙钟超时（秒），超时后强制退出。0 表示不限制。
    """
    llm_call_count = 0
    consecutive_errors = 0
    t_start = time.monotonic()

    def _wall_expired() -> bool:
        if wall_timeout_sec <= 0:
            return False
        return (time.monotonic() - t_start) >= wall_timeout_sec

    if max_tool_calls <= 0:
        resp = llm_call_with_tools(
            messages,
            tools=[],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        llm_call_count += 1
        if stats is not None:
            stats["llm_calls"] = llm_call_count
        return resp.text or ""

    calls_remaining = max_tool_calls
    retries_left = json_retries
    last_text = ""

    while calls_remaining > 0:
        if early_stop_fn is not None and early_stop_fn():
            break
        if _wall_expired():
            log.warning("[agentic_loop] 墙钟超时 (%.0fs)，强制退出", wall_timeout_sec)
            if stats is not None:
                stats["llm_calls"] = llm_call_count
                stats["exit_reason"] = "wall_timeout"
            break

        # 在 LLM 调用前记录剩余时间，帮助诊断
        _remaining = (
            (wall_timeout_sec - (time.monotonic() - t_start))
            if wall_timeout_sec > 0
            else float("inf")
        )
        if _remaining < 600 and wall_timeout_sec > 0:
            log.warning("[agentic_loop] 剩余时间不足: %.0fs", _remaining)

        try:
            resp = llm_call_with_tools(
                messages,
                tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            backoff = min(2**consecutive_errors, 30)  # 2, 4, 8, 16, 30s
            log.warning(
                "[agentic_loop] LLM 调用失败 (%d/%d), %.0fs 后重试: %s",
                consecutive_errors,
                _MAX_CONSECUTIVE_ERRORS,
                backoff,
                str(e)[:200],
            )
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                log.error(
                    "[agentic_loop] 连续 %d 次 LLM 错误，熔断退出", consecutive_errors
                )
                if stats is not None:
                    stats["llm_calls"] = llm_call_count
                    stats["exit_reason"] = "consecutive_errors"
                return last_text
            time.sleep(backoff)
            continue
        llm_call_count += 1

        last_text = resp.text or ""

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
                # 工具执行前再次检查超时
                if _wall_expired():
                    log.warning(
                        "[agentic_loop] 墙钟超时 (tool exec 前)，跳过 tool=%s", tc.name
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(
                                {
                                    "error": "Wall timeout reached before tool execution",
                                    "error_type": "wall_timeout",
                                }
                            ),
                        }
                    )
                    if stats is not None:
                        stats["llm_calls"] = llm_call_count
                        stats["exit_reason"] = "wall_timeout"
                    return resp.text or ""

                result_text = tool_executor(tc.name, tc.arguments)
                log.info("[agentic_loop] tool=%s result=%s", tc.name, result_text[:200])
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                )
                # compact 检查: threshold 触发 或 tool 返回 __force_compact__
                _force = '"__force_compact__"' in result_text
                if compact_fn is not None and (
                    _force
                    or (compact_threshold > 0 and len(messages) > compact_threshold)
                ):
                    before_len = len(messages)
                    messages[:] = compact_fn(messages)
                    if len(messages) < before_len:
                        log.info(
                            "[agentic_loop] compact: %d -> %d messages",
                            before_len,
                            len(messages),
                        )
                calls_remaining -= 1
                if early_stop_fn is not None and early_stop_fn():
                    return resp.text or ""
                if _wall_expired():
                    log.warning("[agentic_loop] 墙钟超时 (tool exec 后)，强制退出")
                    if stats is not None:
                        stats["llm_calls"] = llm_call_count
                        stats["exit_reason"] = "wall_timeout"
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
    llm_call_count += 1
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
            llm_call_count += 1
            text = resp.text or ""

    if stats is not None:
        stats["llm_calls"] = llm_call_count
    return text
