"""RAG Pipeline 编排 — 顺序执行 5 个组件。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .config import SystemConfig
from .components import (
    QuestionRewriter,
    InfoExtractor,
    Retriever,
    HintGenerator,
    AnswerGenerator,
)

COMPONENT_REGISTRY = {
    "QuestionRewriter": QuestionRewriter,
    "InfoExtractor": InfoExtractor,
    "Retriever": Retriever,
    "HintGenerator": HintGenerator,
    "AnswerGenerator": AnswerGenerator,
}

_CONFIG_FIELD_MAP = {
    "question_rewriter": "question_rewriter",
    "info_extractor": "info_extractor",
    "retriever": "retriever",
    "hint_generator": "hint_generator",
    "answer_generator": "answer_generator",
}


@dataclass
class PipelineResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class RAGPipeline:
    """HotpotQA RAG Pipeline — 5 组件顺序执行。"""

    def __init__(
        self,
        config: SystemConfig | None = None,
        corpus: list[str] | None = None,
        source_dir: str | None = None,
    ):
        self.config = config or SystemConfig()
        self.corpus = corpus
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
                    cfg = getattr(self.config, cfg_field)
                    if cls is Retriever:
                        self.components.append(
                            (entry["name"], cls(cfg, corpus=self.corpus))
                        )
                    else:
                        self.components.append((entry["name"], cls(cfg)))
        else:
            self.components = [
                ("question_rewriter", QuestionRewriter(self.config.question_rewriter)),
                ("info_extractor", InfoExtractor(self.config.info_extractor)),
                ("retriever", Retriever(self.config.retriever, corpus=self.corpus)),
                ("hint_generator", HintGenerator(self.config.hint_generator)),
                ("answer_generator", AnswerGenerator(self.config.answer_generator)),
            ]

    def __call__(self, question: str) -> PipelineResult:
        ctx = {"question": question}
        intermediate = {}

        for name, component in self.components:
            output = component.forward(**ctx)
            ctx.update(output)
            intermediate[name] = output

        # Propagate ambiguity and retrieval flags through context
        if "is_ambiguous" not in ctx:
            ctx["is_ambiguous"] = False
        if "retrieval_empty" not in ctx:
            ctx["retrieval_empty"] = False

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
