"""Ranking metrics for STaRK-Prime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def reciprocal_rank(prediction: dict[int, float], ground_truth: list[int]) -> float:
    if not prediction or not ground_truth:
        return 0.0

    ranked_ids = [
        doc_id
        for doc_id, _ in sorted(
            prediction.items(), key=lambda item: item[1], reverse=True
        )
    ]
    ground_truth_set = set(int(v) for v in ground_truth)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if int(doc_id) in ground_truth_set:
            return 1.0 / rank
    return 0.0


def mrr_score(prediction: dict[int, float], ground_truth: list[int]) -> float:
    return reciprocal_rank(prediction, ground_truth)


def score_example(prediction: dict[int, float], ground_truth: list[int]) -> float:
    return mrr_score(prediction, ground_truth)


def evaluate_batch(
    pipeline,
    dataset: list,
    max_workers: int = 5,
) -> dict:
    """Evaluate the ranked prediction dict with MRR."""

    def _eval_one(example):
        result = pipeline(question=example.question, query_id=example.query_id)
        score = score_example(result.prediction, example.answer_ids)
        top_id = next(iter(result.prediction), None)
        return {
            "id": example.id,
            "query_id": example.query_id,
            "question": example.question,
            "answer_ids": example.answer_ids,
            "top_prediction": top_id,
            "mrr": score,
        }

    details = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, ex): ex for ex in dataset}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating"
        ):
            details.append(future.result())

    mean_mrr = sum(d["mrr"] for d in details) / len(details) if details else 0.0
    return {"score": round(mean_mrr * 100, 2), "details": details}
