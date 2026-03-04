"""系统配置 — dataclass + JSON 序列化。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from . import prompts
from utils.llm import resolve_model


@dataclass
class ComponentConfig:
    name: str
    prompt_template: str
    model: str = resolve_model("claude-haiku-4-5-20251001")
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class RetrieverConfig:
    backend: str = (
        "wiki_semantic"  # "wiki_semantic" | "local" | "colbert" | "wikipedia"
    )
    k: int = 3
    k_search_space: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    colbert_url: str = "http://20.102.90.50:2017/wiki17_abstracts"
    embed_model: str = "all-MiniLM-L6-v2"
    n_search: int = 10


@dataclass
class SystemConfig:
    question_rewriter: ComponentConfig = field(default=None)
    info_extractor: ComponentConfig = field(default=None)
    retriever: RetrieverConfig = field(default=None)
    hint_generator: ComponentConfig = field(default=None)
    answer_generator: ComponentConfig = field(default=None)

    def __post_init__(self):
        if self.question_rewriter is None:
            self.question_rewriter = ComponentConfig(
                "question_rewriter", prompts.QUESTION_REWRITER
            )
        if self.info_extractor is None:
            self.info_extractor = ComponentConfig(
                "info_extractor", prompts.INFO_EXTRACTOR
            )
        if self.retriever is None:
            self.retriever = RetrieverConfig()
        if self.hint_generator is None:
            self.hint_generator = ComponentConfig(
                "hint_generator", prompts.HINT_GENERATOR
            )
        if self.answer_generator is None:
            self.answer_generator = ComponentConfig(
                "answer_generator", prompts.ANSWER_GENERATOR
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SystemConfig:
        cfg = cls()
        for comp_name in (
            "question_rewriter",
            "info_extractor",
            "hint_generator",
            "answer_generator",
        ):
            if comp_name in d:
                comp_d = dict(d[comp_name])
                if "model" in comp_d:
                    comp_d["model"] = resolve_model(comp_d["model"])
                setattr(cfg, comp_name, ComponentConfig(**comp_d))
        if "retriever" in d:
            cfg.retriever = RetrieverConfig(**d["retriever"])
        return cfg

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> SystemConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
