"""LLM 调用封装 — 基于 litellm 的薄封装，支持指数退避重试。"""

import time
from dataclasses import dataclass

import litellm

litellm.drop_params = True

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3


@dataclass
class LLMResponse:
    text: str
    usage: dict
    latency_ms: float


def llm_call(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.7,
    system: str | None = None,
) -> LLMResponse:
    """调用 LLM，返回结构化响应。指数退避重试最多 MAX_RETRIES 次。"""
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
            )
            latency = (time.time() - t0) * 1000
            return LLMResponse(
                text=resp.choices[0].message.content.strip(),
                usage=dict(resp.usage) if resp.usage else {},
                latency_ms=round(latency, 1),
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"[LLM] retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
