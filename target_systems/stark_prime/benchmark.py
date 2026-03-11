"""Thin wrapper around the official STaRK benchmark package."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DATASET = "prime"
DEFAULT_CANDIDATE_TOP_K = 20


@dataclass
class STaRKExample:
    question: str
    answer_ids: list[int]
    id: str
    query_id: int
    meta_info: dict = field(default_factory=dict)


def _normalize_doc_info(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, list):
        return "\n".join(_normalize_doc_info(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _require_stark_qa():
    try:
        import stark_qa  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "stark_qa is required for STaRK-Prime. Install it with "
            "`python3 -m pip install stark-qa` and configure the benchmark data root."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "stark_qa is present but its dependencies are incomplete or incompatible. "
            "Reinstall the official STaRK benchmark stack before running STaRK-Prime."
        ) from exc
    return stark_qa


class StarkQaBackend:
    """Official STaRK benchmark adapter used by the STaRK-Prime target system."""

    def __init__(
        self,
        dataset_name: str | None = None,
        *,
        root: str | None = None,
        candidate_top_k: int | None = None,
    ):
        stark_qa = _require_stark_qa()
        self.dataset_name = dataset_name or os.getenv(
            "STARK_QA_DATASET", DEFAULT_DATASET
        )
        self.root = root or os.getenv("STARK_QA_ROOT")
        self.candidate_top_k = int(
            candidate_top_k
            or os.getenv("STARK_QA_BM25_TOP_K", str(DEFAULT_CANDIDATE_TOP_K))
        )

        self.qa_dataset = stark_qa.load_qa(self.dataset_name, root=self.root)
        self.skb = stark_qa.load_skb(self.dataset_name, root=self.root)
        self.retriever = stark_qa.BM25Retriever(skb=self.skb)
        self.evaluator = stark_qa.Evaluator(
            candidate_ids=getattr(self.skb, "candidate_ids", None)
        )
        self._splits = self._load_splits()

    def _load_splits(self) -> dict:
        try:
            return self.qa_dataset.get_idx_split(test_ratio=1.0)
        except TypeError:
            return self.qa_dataset.get_idx_split()

    def available_splits(self) -> list[str]:
        return list(self._splits.keys())

    def resolve_split(self, split: str) -> str:
        aliases = {
            "train": ["train", "training"],
            "val": ["val", "valid", "validation", "dev"],
            "test": ["test", "human_generated_eval", "test-0.1", "test-1.0"],
        }
        candidates = aliases.get(split, [split])
        for name in candidates:
            if name in self._splits:
                return name
        available = ", ".join(sorted(self._splits))
        raise KeyError(f"Unknown STaRK split '{split}'. Available: {available}")

    def load_examples(
        self,
        split: str,
        *,
        n: int | None = None,
        seed: int = 42,
    ) -> list[STaRKExample]:
        split_name = self.resolve_split(split)
        indices = list(self._splits[split_name])
        if n is not None and n < len(indices):
            rng = random.Random(seed)
            indices = rng.sample(indices, n)

        examples = []
        for idx in indices:
            item = self.qa_dataset[idx]
            qid = int(item.get("q_id", item.get("query_id", idx)))
            question = item.get("query", item.get("question", "")).strip()
            answer_ids = [int(v) for v in item.get("answer_ids", [])]
            examples.append(
                STaRKExample(
                    question=question,
                    answer_ids=answer_ids,
                    id=str(qid),
                    query_id=qid,
                    meta_info=item.get("meta_info", {}),
                )
            )
        return examples

    def retrieve_candidates(self, query: str, *, top_k: int | None = None) -> dict:
        top_k = int(top_k or self.candidate_top_k)

        if callable(self.retriever):
            pred = self.retriever(query, topk=top_k)
        elif hasattr(self.retriever, "retrieve"):
            pred = self.retriever.retrieve(query, topk=top_k)
        else:
            raise RuntimeError("BM25 retriever is neither callable nor has retrieve()")

        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, list):
            pred = {node_id: float(top_k - i) for i, node_id in enumerate(pred)}
        if not isinstance(pred, dict):
            raise TypeError(f"Unexpected retriever output type: {type(pred).__name__}")

        normalized = {}
        for key, value in pred.items():
            try:
                norm_key = int(key)
            except (TypeError, ValueError):
                norm_key = key
            normalized[norm_key] = float(value)
        return normalized

    def get_doc_text(self, node_id: int, *, add_rel: bool) -> str:
        getter = getattr(self.skb, "get_doc_info", None) or getattr(
            self.skb, "get_node_info", None
        )
        if getter is None:
            raise RuntimeError("STaRK SKB has neither get_doc_info nor get_node_info")

        try:
            value = getter(node_id, add_rel=add_rel, compact=True)
        except TypeError:
            try:
                value = getter(node_id, add_rel=add_rel)
            except TypeError:
                value = getter(node_id)
        return _normalize_doc_info(value)
