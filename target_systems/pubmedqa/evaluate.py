"""Exact Match 评估指标 — PubMedQA yes/no/maybe 三分类。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def extract_answer_yesno(text: str) -> str:
    """从模型回复中提取 yes/no/maybe 答案。"""
    pattern = r"(?i)\b(yes|no|maybe)\b"
    match = re.search(pattern, text)
    return match.group(1).lower() if match else text.strip().lower()


def exact_match(prediction: str, ground_truth: str) -> float:
    """精确匹配 — 提取 yes/no/maybe 后比较。"""
    pred = extract_answer_yesno(prediction)
    gt = ground_truth.strip().lower()
    return 1.0 if pred == gt else 0.0


def evaluate_batch(
    pipeline,
    dataset: list,
    max_workers: int = 10,
) -> dict:
    """并行评估 pipeline，返回 accuracy + per_example 详情。"""

    def _eval_one(example):
        result = pipeline(question=example.question, context=example.context)
        score = exact_match(result.answer, example.answer)
        return {
            "id": example.id,
            "question": example.question,
            "ground_truth": example.answer,
            "prediction": result.answer,
            "extracted": extract_answer_yesno(result.answer),
            "accuracy": score,
        }

    details = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in dataset}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating"
        ):
            details.append(future.result())

    accuracy = sum(d["accuracy"] for d in details) / len(details) if details else 0.0
    return {"score": round(accuracy * 100, 2), "details": details}
