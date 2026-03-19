"""LLM 薄封装：基于 litellm。"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import litellm

load_dotenv()

# litellm 需要 TOGETHER_AI_API_KEY，.env 里可能是 TOGETHER_API_KEY
if os.getenv("TOGETHER_API_KEY") and not os.getenv("TOGETHER_AI_API_KEY"):
    os.environ["TOGETHER_AI_API_KEY"] = os.environ["TOGETHER_API_KEY"]


class LLMClient:
    def __init__(
        self,
        model: str = "together_ai/MiniMaxAI/MiniMax-M2.5",
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens

    async def generate(
        self,
        system_msg: str,
        user_msg: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ) -> str:
        """调用 LLM 生成回复（带重试）。"""
        import asyncio

        tokens = max_tokens or self.max_tokens
        last_err = None
        for attempt in range(max_retries):
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_err
