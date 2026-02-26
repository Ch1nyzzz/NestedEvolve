"""F1 评估指标 — 与 OPTIMAS 实现一致的 token-level F1。"""

import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def normalize_answer(s: str) -> str:
    """Lower → remove articles → remove punctuation → whitespace fix."""
    s = s.lower()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # whitespace fix
    s = " ".join(s.split())
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1，yes/no/noanswer 特殊处理。"""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # 特殊 token 精确匹配
    if gt_norm in ("yes", "no", "noanswer"):
        return float(pred_norm == gt_norm)

    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_batch(
    pipeline,
    dataset: list,
    max_workers: int = 8,
) -> dict:
    """并行评估 pipeline，返回 mean_f1 + per_example 详情。"""

    def _eval_one(example):
        result = pipeline(example.question)
        score = f1_score(result.answer, example.answer)
        return {
            "id": example.id,
            "question": example.question,
            "ground_truth": example.answer,
            "prediction": result.answer,
            "f1": score,
        }

    details = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in dataset}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            details.append(future.result())

    mean_f1 = sum(d["f1"] for d in details) / len(details) if details else 0.0
    return {"score": round(mean_f1 * 100, 2), "details": details}
