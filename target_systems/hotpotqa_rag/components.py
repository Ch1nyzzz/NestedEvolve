"""RAG Pipeline 组件 — 5 个组件的统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.llm import llm_call
from .config import ComponentConfig, RetrieverConfig
from .retriever import HybridRetriever, LocalRetriever, WikipediaRetriever, WikiSemanticRetriever


class BaseComponent(ABC):
    """组件基类，统一 forward(**inputs) -> dict 接口。"""

    @abstractmethod
    def forward(self, **inputs) -> dict:
        ...


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
        return {"rewritten_query": resp.text}


class InfoExtractor(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(self, *, rewritten_query: str, **_) -> dict:
        prompt = f"{self.config.prompt_template}\n\nQuery: {rewritten_query}\n\nKeywords:"
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return {"search_keywords": resp.text}


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

    def forward(self, *, search_keywords: str, **_) -> dict:
        passages = self.engine.retrieve(search_keywords, k=self.config.k)
        content = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        return {"retrieve_content": content}


class HintGenerator(BaseComponent):
    def __init__(self, config: ComponentConfig):
        self.config = config

    def forward(self, *, rewritten_query: str, retrieve_content: str, **_) -> dict:
        prompt = (
            f"{self.config.prompt_template}\n\n"
            f"Query: {rewritten_query}\n\n"
            f"Retrieved content:\n{retrieve_content}\n\n"
            f"Hints:"
        )
        resp = llm_call(
            prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return {"hints": resp.text}


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
        return {"answer": resp.text}
