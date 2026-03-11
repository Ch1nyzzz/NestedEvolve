"""System configuration for STaRK-Prime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from . import prompts


@dataclass
class ComponentConfig:
    name: str
    prompt_template: str
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 2048
    temperature: float = 0.0


@dataclass
class AggregatorConfig:
    text_weight: float = 0.5
    relation_weight: float = 0.5
    weight_search_space: list[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )


@dataclass
class SystemConfig:
    text_scorer: ComponentConfig = field(default=None)
    relation_scorer: ComponentConfig = field(default=None)
    aggregator: AggregatorConfig = field(default=None)
    candidate_top_k: int = 20
    max_doc_chars: int = 1200

    def __post_init__(self):
        if self.text_scorer is None:
            self.text_scorer = ComponentConfig("text_scorer", prompts.TEXT_SCORER)
        if self.relation_scorer is None:
            self.relation_scorer = ComponentConfig(
                "relation_scorer", prompts.RELATION_SCORER
            )
        if self.aggregator is None:
            self.aggregator = AggregatorConfig()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SystemConfig":
        cfg = cls()
        for comp_name in ("text_scorer", "relation_scorer"):
            if comp_name in d:
                setattr(cfg, comp_name, ComponentConfig(**dict(d[comp_name])))
        if "aggregator" in d:
            cfg.aggregator = AggregatorConfig(**d["aggregator"])
        if "candidate_top_k" in d:
            cfg.candidate_top_k = int(d["candidate_top_k"])
        if "max_doc_chars" in d:
            cfg.max_doc_chars = int(d["max_doc_chars"])
        return cfg

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "SystemConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
