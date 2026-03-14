"""PubMedQA Pipeline 编排 — 顺序执行 4 个组件。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .config import SystemConfig
from .components import (
    ContextModelSelector,
    ContextAnalyst,
    SolverModelSelector,
    ProblemSolver,
)

COMPONENT_REGISTRY = {
    "ContextModelSelector": ContextModelSelector,
    "ContextAnalyst": ContextAnalyst,
    "SolverModelSelector": SolverModelSelector,
    "ProblemSolver": ProblemSolver,
}

_CONFIG_FIELD_MAP = {
    "context_model_selector": "context_model_selector",
    "context_analyst": "context_analyst",
    "solver_model_selector": "solver_model_selector",
    "problem_solver": "problem_solver",
}


@dataclass
class PipelineResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class PubMedQAPipeline:
    """PubMedQA Pipeline — 4 组件顺序执行。"""

    def __init__(
        self, config: SystemConfig | None = None, source_dir: str | None = None
    ):
        self.config = config or SystemConfig()
        self._source_dir = source_dir or os.path.dirname(__file__)
        self._build_components()

    def _build_components(self):
        config_path = os.path.join(self._source_dir, "pipeline_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                pipeline_def = json.load(f)["pipeline"]
            self.components = []
            for entry in pipeline_def:
                if not entry.get("enabled", True):
                    continue
                cls = COMPONENT_REGISTRY[entry["class"]]
                cfg_field = _CONFIG_FIELD_MAP.get(entry["name"])
                if cfg_field:
                    self.components.append(
                        (entry["name"], cls(getattr(self.config, cfg_field)))
                    )
        else:
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
