"""STaRK-Prime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .benchmark import StarkQaBackend
from .components import Aggregator, RelationScorer, TextScorer
from .config import SystemConfig


@dataclass
class PipelineResult:
    answer: str
    prediction: dict[int, float] = field(default_factory=dict)
    intermediate: dict = field(default_factory=dict)


class STaRKPrimePipeline:
    """STaRK-Prime uses two scorer modules and a weighted aggregator.

    Candidate generation is handled by a fixed BM25 retriever from the official
    STaRK benchmark package so the optimized modules stay aligned with the paper:
    TextScorer, RelationScorer, Aggregator.
    """

    def __init__(
        self,
        config: SystemConfig | None = None,
        backend: StarkQaBackend | None = None,
    ):
        self.config = config or SystemConfig()
        self.backend = backend
        self._build_components()

    def _ensure_backend(self) -> StarkQaBackend:
        if self.backend is None:
            self.backend = StarkQaBackend(candidate_top_k=self.config.candidate_top_k)
        return self.backend

    def _build_components(self):
        self.components = [
            ("text_scorer", TextScorer(self.config.text_scorer)),
            ("relation_scorer", RelationScorer(self.config.relation_scorer)),
            ("aggregator", Aggregator(self.config.aggregator)),
        ]

    def _truncate(self, text: str) -> str:
        if len(text) <= self.config.max_doc_chars:
            return text
        return text[: self.config.max_doc_chars].rstrip() + " ..."

    def __call__(self, question: str, query_id: int | None = None) -> PipelineResult:
        backend = self._ensure_backend()
        bm25_scores = backend.retrieve_candidates(
            question, top_k=self.config.candidate_top_k
        )
        candidate_ids = list(bm25_scores)
        candidate_texts = {
            cid: self._truncate(backend.get_doc_text(cid, add_rel=False))
            for cid in candidate_ids
        }
        relation_texts = {
            cid: self._truncate(backend.get_doc_text(cid, add_rel=True))
            for cid in candidate_ids
        }

        ctx = {
            "question": question,
            "query_id": query_id,
            "candidate_ids": candidate_ids,
            "candidate_texts": candidate_texts,
            "relation_texts": relation_texts,
            "bm25_scores": bm25_scores,
        }
        intermediate = {"bm25": bm25_scores}

        for name, component in self.components:
            output = component.forward(**ctx)
            ctx.update(output)
            intermediate[name] = output

        ranking = ctx.get("ranking", {})
        return PipelineResult(
            answer=ctx.get("answer", ""),
            prediction=ranking,
            intermediate=intermediate,
        )

    def update_config(self, patch: dict):
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
