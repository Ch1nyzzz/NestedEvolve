"""Components for the STaRK-Prime retrieval system."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from . import prompts
from .config import AggregatorConfig, ComponentConfig


def _make_artifacts(resp, config: ComponentConfig) -> dict:
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "finish_reason": resp.finish_reason,
        "tokens_used": resp.usage.get("completion_tokens", 0),
        "latency_ms": resp.latency_ms,
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end].split("\n", 1)[-1]
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and left < right:
        try:
            return json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_scores(text: str, candidate_ids: list[int]) -> dict[int, float]:
    parsed = _extract_json(text)
    raw_scores = parsed.get("scores", parsed if isinstance(parsed, dict) else {})
    scores = {cid: 0.0 for cid in candidate_ids}
    if not isinstance(raw_scores, dict):
        return scores

    for key, value in raw_scores.items():
        try:
            cid = int(key)
        except (TypeError, ValueError):
            continue
        if cid not in scores:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        scores[cid] = max(0.0, min(1.0, score))
    return scores


def _render_candidates(candidate_ids: list[int], docs: dict[int, str]) -> str:
    blocks = []
    for cid in candidate_ids:
        blocks.append(f"[Candidate {cid}]\n{docs.get(cid, '').strip()}")
    return "\n\n".join(blocks)


class BaseComponent(ABC):
    @abstractmethod
    def forward(self, **inputs) -> dict: ...


class _BaseScorer(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def _score(self, *, question: str, candidate_ids: list[int], docs: dict[int, str]):
        from utils.llm import llm_call

        prompt = (
            f"{self.config.prompt_template}\n\n"
            f"Query:\n{question}\n\n"
            f"Candidates:\n{_render_candidates(candidate_ids, docs)}\n\n"
            f"{prompts.OUTPUT_FORMAT}"
        )
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=prompts.SYSTEM_PROMPT,
        )
        return _parse_scores(resp.text, candidate_ids), _make_artifacts(
            resp, self.config
        )


class TextScorer(_BaseScorer):
    def forward(
        self, *, question: str, candidate_ids: list[int], candidate_texts: dict, **_
    ):
        scores, artifacts = self._score(
            question=question,
            candidate_ids=candidate_ids,
            docs=candidate_texts,
        )
        return {"text_scores": scores, "_artifacts": artifacts}


class RelationScorer(_BaseScorer):
    def forward(
        self,
        *,
        question: str,
        candidate_ids: list[int],
        relation_texts: dict,
        **_,
    ):
        scores, artifacts = self._score(
            question=question,
            candidate_ids=candidate_ids,
            docs=relation_texts,
        )
        return {"relation_scores": scores, "_artifacts": artifacts}


class Aggregator(BaseComponent):
    def __init__(self, config: AggregatorConfig):
        self.config = config

    def forward(
        self,
        *,
        candidate_ids: list[int],
        text_scores: dict[int, float],
        relation_scores: dict[int, float],
        **_,
    ) -> dict:
        ranking = {}
        for cid in candidate_ids:
            text_score = float(text_scores.get(cid, 0.0))
            relation_score = float(relation_scores.get(cid, 0.0))
            ranking[cid] = (
                self.config.text_weight * text_score
                + self.config.relation_weight * relation_score
            )

        ordered = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True))
        top_id = next(iter(ordered), "")
        return {"ranking": ordered, "answer": str(top_id)}
