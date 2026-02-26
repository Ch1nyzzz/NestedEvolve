"""HotpotQA 数据加载 — 从 HuggingFace 加载并固定 seed 采样。"""

import random
from dataclasses import dataclass

from datasets import load_dataset

SEED = 42


@dataclass
class QAExample:
    question: str
    answer: str
    id: str


def _to_examples(split_data) -> list[QAExample]:
    return [
        QAExample(question=item["question"], answer=item["answer"], id=item["id"])
        for item in split_data
    ]


def load_hotpotqa(
    split: str = "validation",
    n: int = 100,
) -> list[QAExample]:
    """加载 HotpotQA distractor 数据集，固定 seed 采样返回 n 条。"""
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    items = list(ds)
    rng = random.Random(SEED)
    sampled = rng.sample(items, min(n, len(items)))
    return _to_examples(sampled)


def build_corpus(split: str = "validation") -> list[str]:
    """从 HotpotQA 数据集收集所有去重段落，用于本地 TF-IDF 检索。

    每个文档合并为一个段落："{title}: {sentence1} {sentence2} ..."
    """
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    seen = set()
    corpus = []
    for item in ds:
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        for title, sentences in zip(titles, sentences_list):
            if title in seen:
                continue
            seen.add(title)
            text = f"{title}: {' '.join(sentences)}"
            corpus.append(text)
    return corpus
