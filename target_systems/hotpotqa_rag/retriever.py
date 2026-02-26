"""检索器 — ColBERTv2 / Wikipedia / 本地 TF-IDF / WikiSemantic。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 持久化磁盘缓存 + 内存缓存，保证跨运行可复现
# 统一用 ~/.cache/noa_retriever，避免 sandbox 临时目录导致缓存路径不一致
_CACHE_DIR = Path.home() / ".cache" / "noa_retriever"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "http_cache.json"

# 内存缓存：key = str(cache_key), value = (status_code, json_body)
_mem_cache: dict[str, tuple[int, Any]] = {}


def _load_disk_cache() -> None:
    """启动时从磁盘加载缓存到内存。"""
    global _mem_cache
    if _CACHE_FILE.exists():
        try:
            _mem_cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _mem_cache = {}


def _save_disk_cache() -> None:
    """将内存缓存写回磁盘。"""
    try:
        _CACHE_FILE.write_text(json.dumps(_mem_cache, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print(f"[Cache] 写入磁盘缓存失败: {e}")


# 模块加载时读取磁盘缓存
_load_disk_cache()


class _FakeResponse:
    """轻量 Response 替身，只保留 status_code / json() / raise_for_status()。"""

    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"Cached response {self.status_code}")


def _cached_get(cache_key: tuple, url: str, *, params: dict | None = None,
                headers: dict | None = None, timeout: int = 10) -> _FakeResponse:
    """带持久化缓存的 requests.get，命中则直接返回，未命中则请求并存入。"""
    key = str(cache_key)
    if key in _mem_cache:
        sc, body = _mem_cache[key]
        return _FakeResponse(sc, body)
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    _mem_cache[key] = (resp.status_code, resp.json())
    _save_disk_cache()
    return _FakeResponse(resp.status_code, resp.json())


class ColBERTv2Retriever:
    """调用 Stanford ColBERTv2 服务检索 Wikipedia 段落。"""

    def __init__(self, url: str = "http://20.102.90.50:2017/wiki17_abstracts"):
        self.url = url

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        try:
            cache_key = ("colbert", self.url, query, k)
            resp = _cached_get(cache_key, self.url,
                               params={"query": query, "k": k})
            resp.raise_for_status()
            results = resp.json().get("topk", [])
            return [r.get("text", "") for r in results[:k]]
        except Exception as e:
            print(f"[ColBERT] error: {e}")
            return []


class LocalRetriever:
    """本地 TF-IDF 检索器 — 基于数据集自带的 context 段落建索引。"""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"[LocalRetriever] 索引建立完成，共 {len(corpus)} 个段落")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.corpus[i] for i in top_indices if scores[i] > 0]


class WikipediaRetriever:
    """Wikipedia API fallback 检索器。"""

    API_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "NOA-HotpotQA-Research/1.0 (research project)"}

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        try:
            cache_key = ("wiki_search", query, k)
            resp = _cached_get(cache_key, self.API_URL,
                               params={
                                   "action": "query",
                                   "list": "search",
                                   "srsearch": query,
                                   "srlimit": k,
                                   "format": "json",
                               },
                               headers=self.HEADERS)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            return [r.get("snippet", "") for r in results[:k]]
        except Exception as e:
            print(f"[Wikipedia] error: {e}")
            return []


class HybridRetriever:
    """ColBERTv2 优先，失败自动降级为 Wikipedia。"""

    def __init__(self, colbert_url: str = "http://20.102.90.50:2017/wiki17_abstracts"):
        self.colbert = ColBERTv2Retriever(colbert_url)
        self.wikipedia = WikipediaRetriever()

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        results = self.colbert.retrieve(query, k)
        if results:
            return results
        print("[Retriever] ColBERT failed, falling back to Wikipedia")
        return self.wikipedia.retrieve(query, k)


class WikiSemanticRetriever:
    """Wikipedia API 召回 + sentence-transformers 语义重排。

    三步检索：
    1. Wikipedia Search API → 文章标题（召回）
    2. Wikipedia Extracts API (exintro=true) → 每篇文章的引言摘要
    3. sentence-transformers 向量余弦相似度 → top-k（语义重排）
    """

    API_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "NOA-HotpotQA-Research/1.0 (research project)"}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_search: int = 10):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.n_search = n_search

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        titles = self._search_titles(query, limit=self.n_search)
        if not titles:
            return []
        abstracts = self._fetch_abstracts(titles)
        if not abstracts:
            return []
        return self._rerank(query, abstracts, k)

    def _search_titles(self, query: str, limit: int) -> list[str]:
        try:
            cache_key = ("wikisem_search", query, limit)
            resp = _cached_get(cache_key, self.API_URL,
                               params={
                                   "action": "query",
                                   "list": "search",
                                   "srsearch": query,
                                   "srlimit": limit,
                                   "format": "json",
                               },
                               headers=self.HEADERS)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except Exception as e:
            print(f"[WikiSemantic] search error: {e}")
            return []

    def _fetch_abstracts(self, titles: list[str]) -> list[str]:
        try:
            titles_key = "|".join(titles)
            cache_key = ("wikisem_extracts", titles_key)
            resp = _cached_get(cache_key, self.API_URL,
                               params={
                                   "action": "query",
                                   "prop": "extracts",
                                   "exintro": True,
                                   "explaintext": True,
                                   "titles": titles_key,
                                   "format": "json",
                               },
                               headers=self.HEADERS)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            abstracts = []
            for page in pages.values():
                text = page.get("extract", "").strip()
                if text:
                    abstracts.append(f"{page.get('title', '')}: {text}")
            return abstracts
        except Exception as e:
            print(f"[WikiSemantic] extracts error: {e}")
            return []

    def _rerank(self, query: str, abstracts: list[str], k: int) -> list[str]:
        embeddings = self.model.encode([query] + abstracts)
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]
        scores = cosine_similarity([query_emb], doc_embs).flatten()
        top_indices = np.argsort(scores)[-k:][::-1]
        return [abstracts[i] for i in top_indices]
