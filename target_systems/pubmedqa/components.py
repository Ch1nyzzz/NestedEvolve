"""PubMedQA Pipeline 组件 — 4 个组件的统一接口。

架构（来自 OPTIMAS 论文）：
  ContextModelSelector → ContextAnalyst → SolverModelSelector → ProblemSolver

异构配置：
  - ModelSelector: 离散模型选择（可优化 selected_model）
  - Analyst/Solver: prompt 优化 + 模型参数
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.llm import llm_call
from .config import ComponentConfig, ModelSelectorConfig
from . import prompts


def _make_artifacts(resp, config: ComponentConfig) -> dict:
    """构造 LLM 调用的执行元数据。"""
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "finish_reason": resp.finish_reason,
        "tokens_used": resp.usage.get("completion_tokens", 0),
        "latency_ms": resp.latency_ms,
    }


class BaseComponent(ABC):
    """组件基类，统一 forward(**inputs) -> dict 接口。"""

    @abstractmethod
    def forward(self, **inputs) -> dict: ...


class ContextModelSelector(BaseComponent):
    """选择 ContextAnalyst 使用的模型。"""

    def __init__(self, config: ModelSelectorConfig):
        self.config = config

    def forward(self, *, context: str, question: str, **_) -> dict:
        return {"context_analyst_model": self.config.selected_model}


class ContextAnalyst(BaseComponent):
    """提取并总结上下文中回答问题所需的关键信息。"""

    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(
        self, *, context: str, question: str, context_analyst_model: str, **_
    ) -> dict:
        prompt = (
            f"{self.config.prompt_template}\n\n"
            f'Here is the given context:\n"{context}"\n\n'
            f'Problem:\n"{question}"\n\n'
            f"Please summarize the relevant information from the context "
            f"related to the question."
        )
        resp = llm_call(
            prompt,
            model=context_analyst_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=prompts.SYSTEM_PROMPT,
        )
        return {"summary": resp.text, "_artifacts": _make_artifacts(resp, self.config)}


class SolverModelSelector(BaseComponent):
    """选择 ProblemSolver 使用的模型。"""

    def __init__(self, config: ModelSelectorConfig):
        self.config = config

    def forward(self, *, question: str, summary: str, **_) -> dict:
        return {"problem_solver_model": self.config.selected_model}


class ProblemSolver(BaseComponent):
    """基于摘要判断 yes/no/maybe 答案。"""

    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(
        self, *, question: str, summary: str, problem_solver_model: str, **_
    ) -> dict:
        prompt = (
            f"{self.config.prompt_template}\n\n"
            f'Problem:\n"{question}"\n\n'
            f'Here is a summary of relevant information:\n"{summary}"\n\n'
            f"Please provide yes, no or maybe to the given problem. "
            f"{prompts.FORMAT_YESNO}"
        )
        resp = llm_call(
            prompt,
            model=problem_solver_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=prompts.SYSTEM_PROMPT,
        )
        return {"answer": resp.text, "_artifacts": _make_artifacts(resp, self.config)}
