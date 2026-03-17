"""PubMedQA 系统配置 — dataclass + JSON 序列化。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from . import prompts
from utils.llm import resolve_model


# 可选模型列表 — 仅包含默认输出模型，优化集中在算法/prompt 层面
MODELS_LIST = [
    "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
]


@dataclass
class ComponentConfig:
    name: str
    prompt_template: str
    model: str = resolve_model("together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo")
    max_tokens: int = 2048
    temperature: float = 0.0


@dataclass
class ModelSelectorConfig:
    name: str
    selected_model: str = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"
    models_list: list[str] = field(default_factory=lambda: list(MODELS_LIST))


@dataclass
class SystemConfig:
    context_model_selector: ModelSelectorConfig = field(default=None)
    context_analyst: ComponentConfig = field(default=None)
    solver_model_selector: ModelSelectorConfig = field(default=None)
    problem_solver: ComponentConfig = field(default=None)

    def __post_init__(self):
        if self.context_model_selector is None:
            self.context_model_selector = ModelSelectorConfig("context_model_selector")
        if self.context_analyst is None:
            self.context_analyst = ComponentConfig(
                "context_analyst", prompts.CONTEXT_ANALYST
            )
        if self.solver_model_selector is None:
            self.solver_model_selector = ModelSelectorConfig("solver_model_selector")
        if self.problem_solver is None:
            self.problem_solver = ComponentConfig(
                "problem_solver", prompts.PROBLEM_SOLVER
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SystemConfig:
        cfg = cls()
        for comp_name in ("context_analyst", "problem_solver"):
            if comp_name in d:
                comp_d = dict(d[comp_name])
                if "model" in comp_d:
                    comp_d["model"] = resolve_model(comp_d["model"])
                setattr(cfg, comp_name, ComponentConfig(**comp_d))
        for sel_name in ("context_model_selector", "solver_model_selector"):
            if sel_name in d:
                setattr(cfg, sel_name, ModelSelectorConfig(**d[sel_name]))
        return cfg

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> SystemConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
