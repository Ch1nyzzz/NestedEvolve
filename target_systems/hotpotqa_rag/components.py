"""RAG Pipeline 组件 — 5 个组件的统一接口。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from utils.llm import llm_call
from .config import ComponentConfig, RetrieverConfig
from .retriever import (
    HybridRetriever,
    LocalRetriever,
    WikipediaRetriever,
    WikiSemanticRetriever,
)


def _clean_keywords(text: str) -> str:
    """Strip markdown artifacts and descriptive suffixes from LLM keyword output."""
    text = re.sub(r"^[\s]*[#*\->]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(
        r"^(?:keywords?|search\s*keywords?)\s*[:：]\s*", "", text, flags=re.IGNORECASE
    )
    parts = re.split(r"[,\n]+", text)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Remove " - description" suffixes (e.g. "Scott Derrickson - the director")
        p = re.sub(r"\s*[-–—]\s+.*$", "", p)
        # Remove parenthetical explanations (e.g. "keyword (explanation)")
        p = re.sub(r"\s*\(.*?\)\s*", "", p)
        p = p.strip()
        if len(p) > 1:
            cleaned.append(p)
    return ", ".join(cleaned)


def _make_artifacts(resp, config) -> dict:
    """构造 LLM 调用的执行元数据。"""
    return {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "finish_reason": resp.finish_reason,
        "tokens_used": resp.usage.get("completion_tokens", 0),
        "latency_ms": resp.latency_ms,
    }


class BaseComponent(ABC):
    """组件基类，统一 forward(**inputs) -> dict 接口。"""

    @abstractmethod
    def forward(self, **inputs) -> dict: ...


class QuestionRewriter(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(self, *, question: str, **_) -> dict:
        prompt = f"{self.config.prompt_template}\n\nQuestion: {question}\n\nRewritten question:"
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        rewritten = resp.text.strip()
        is_ambiguous = rewritten.startswith("[AMBIGUOUS]")
        return {
            "rewritten_query": rewritten,
            "is_ambiguous": is_ambiguous,
            "_artifacts": _make_artifacts(resp, self.config),
        }


class InfoExtractor(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(self, *, rewritten_query: str, **_) -> dict:
        prompt = (
            f"{self.config.prompt_template}\n\nQuery: {rewritten_query}\n\nKeywords:"
        )
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        keywords = _clean_keywords(resp.text)
        return {
            "search_keywords": keywords,
            "_artifacts": _make_artifacts(resp, self.config),
        }


class Retriever(BaseComponent):
    def __init__(self, config: RetrieverConfig, corpus: list[str] | None = None):
        self.config = config
        if config.backend == "wiki_semantic":
            self.engine = WikiSemanticRetriever(
                model_name=config.embed_model,
                n_search=config.n_search,
            )
        elif config.backend == "local":
            if corpus is None:
                raise ValueError("backend='local' 需要提供 corpus")
            self.engine = LocalRetriever(corpus)
        elif config.backend == "wikipedia":
            self.engine = WikipediaRetriever()
        else:
            self.engine = HybridRetriever(config.colbert_url)

    def forward(self, *, search_keywords: str, is_ambiguous: bool = False, **_) -> dict:
        keywords = [kw.strip() for kw in search_keywords.split(",") if kw.strip()]
        passages = []
        seen = set()
        for kw in keywords:
            for p in self.engine.retrieve(kw, k=self.config.k):
                if p not in seen:
                    seen.add(p)
                    passages.append(p)
        passages = passages[: self.config.k]

        content = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return {"retrieve_content": content, "retrieval_empty": len(passages) == 0}


class HintGenerator(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(
        self,
        *,
        rewritten_query: str,
        retrieve_content: str,
        retrieval_empty: bool = False,
        **_,
    ) -> dict:
        if retrieval_empty:
            hint_prefix = "No relevant content was retrieved. The question may be malformed or refer to non-existent entities. "
        else:
            hint_prefix = ""

        prompt = (
            f"{self.config.prompt_template}\n\n"
            f"Query: {rewritten_query}\n\n"
            f"Retrieved content:\n{retrieve_content}\n\n"
            f"Hints:{hint_prefix}"
        )
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return {"hints": resp.text, "_artifacts": _make_artifacts(resp, self.config)}


class AnswerGenerator(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(self, *, rewritten_query: str, hints: str, **_) -> dict:
        prompt = (
            f"{self.config.prompt_template}\n\n"
            f"Query: {rewritten_query}\n\n"
            f"Hints:\n{hints}\n\n"
            f"Answer:"
        )
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return {"answer": resp.text, "_artifacts": _make_artifacts(resp, self.config)}
