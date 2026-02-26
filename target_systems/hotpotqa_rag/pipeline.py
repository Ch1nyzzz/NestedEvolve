"""RAG Pipeline 编排 — 顺序执行 5 个组件。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .config import SystemConfig
from .components import (
    QuestionRewriter,
    InfoExtractor,
    Retriever,
    HintGenerator,
    AnswerGenerator,
)


@dataclass
class PipelineResult:
    answer: str
    intermediate: dict = field(default_factory=dict)


class RAGPipeline:
    """HotpotQA RAG Pipeline — 5 组件顺序执行。"""

    def __init__(self, config: SystemConfig | None = None, corpus: list[str] | None = None):
        self.config = config or SystemConfig()
        self.corpus = corpus
        self._build_components()

    def _build_components(self):
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

        return PipelineResult(
            answer=ctx.get("answer", ""),
            intermediate=intermediate,
        )

    def update_config(self, patch: dict):
        """增量更新配置并重建组件。"""
        cfg_dict = self.config.to_dict()
        for key, val in patch.items():
            if key in cfg_dict and isinstance(cfg_dict[key], dict) and isinstance(val, dict):
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
