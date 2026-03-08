"""PubMedQA Pipeline 编排 — 顺序执行 4 个组件。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SystemConfig
from .components import (
    ContextModelSelector,
    ContextAnalyst,
    SolverModelSelector,
    ProblemSolver,
)


@dataclass
class PipelineResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class PubMedQAPipeline:
    """PubMedQA Pipeline — 4 组件顺序执行。"""

    def __init__(self, config: SystemConfig | None = None):
        self.config = config or SystemConfig()
        self._build_components()

    def _build_components(self):
        self.components = [
            (
                "context_model_selector",
                ContextModelSelector(self.config.context_model_selector),
            ),
            ("context_analyst", ContextAnalyst(self.config.context_analyst)),
            (
                "solver_model_selector",
                SolverModelSelector(self.config.solver_model_selector),
            ),
            ("problem_solver", ProblemSolver(self.config.problem_solver)),
        ]

    def __call__(self, question: str, context: str) -> PipelineResult:
        ctx = {"question": question, "context": context}
        intermediate = {}

        for name, component in self.components:
            output = component.forward(**ctx)
            ctx.update(output)
            intermediate[name] = output

        return PipelineResult(
            answer=ctx.get("answer", ""),
            intermediate=intermediate,
        )

    def update_config(self, patch: dict):
        """增量更新配置并重建组件。"""
        cfg_dict = self.config.to_dict()
        for key, val in patch.items():
            if (
                key in cfg_dict
                and isinstance(cfg_dict[key], dict)
                and isinstance(val, dict)
            ):
                cfg_dict[key].update(val)
            else:
                cfg_dict[key] = val
        self.config = SystemConfig.from_dict(cfg_dict)
        self._build_components()

    def state_dict(self) -> dict:
        return self.config.to_dict()

    def load_state_dict(self, state: dict):
        self.config = SystemConfig.from_dict(state)
        self._build_components()
