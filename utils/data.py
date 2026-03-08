"""数据加载 — HotpotQA (HuggingFace) + PubMedQA (本地 JSONL)。"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

SEED = 42


@dataclass
class QAExample:
    question: str
    answer: str
    id: str
    context: str = ""


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


def load_pubmedqa(
    split: str = "train",
    n: int = 50,
    data_dir: str | None = None,
) -> list[QAExample]:
    """加载 PubMedQA 数据集（本地 JSONL），固定 seed 采样返回 n 条。

    Args:
        split: "train" 或 "test"
        n: 采样数量
        data_dir: JSONL 文件所在目录，默认 target_systems/pubmedqa/
    """
    if data_dir is None:
        data_dir = str(
            Path(__file__).resolve().parent.parent / "target_systems" / "pubmedqa"
        )
    fname = f"combined_PubMedQA_{split}.jsonl"
    fpath = Path(data_dir) / fname
    with open(fpath, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    rng = random.Random(SEED)
    sampled = rng.sample(items, min(n, len(items)))

    examples = []
    for item in sampled:
        ctx = item.get("context", "")
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        examples.append(
            QAExample(
                question=item["question"],
                answer=item["groundtruth"],
                id=item.get("index", item.get("key", "")),
                context=ctx,
            )
        )
    return examples


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
