"""Evaluator — 级联沙盒验证 patch，决定接受或回滚。

支持 progressive sampling: Stage 3 分批采样，提前决策减少不必要的 pipeline 执行。
"""

from __future__ import annotations

import os
import random
import shutil
import traceback
from typing import Callable

from noa.core.protocol import DeltaPatch, EvalResult, SourceFile
from noa.diff_utils import (
    apply_diffs_in_memory,
    write_to_temp_dir,
    commit_to_source,
)
from noa.eval_guard import (
    should_stop_early,
    _PROGRESSIVE_FIRST_BATCH,
)

_SMOKE_SAMPLES = 3
_SMOKE_THRESHOLD = 0.8  # 烟雾测试得分 < baseline * 0.8 直接拒绝


def evaluate(
    source_files: list[SourceFile],
    source_dir: str,
    patch: DeltaPatch,
    dataset: list,
    eval_fn,
    target_factory: Callable[[str], object],
    baseline_score: float,
    n_samples: int = 20,
    seed: int = 42,
    layer_context=None,
    progressive: bool = True,
) -> EvalResult:
    """级联评估：Stage1 语法检查 → Stage2 烟雾测试 → Stage3 全量评估(progressive)。"""
    rng = random.Random(seed)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    # --- Stage 1: 语法检查 — diff 能否正确应用 ---
    modified_files = apply_diffs_in_memory(source_files, patch.diffs)
    if not modified_files:
        print("[Evaluator] Stage 1 FAIL: Diff 未产生任何修改")
        return EvalResult(
            before_score=baseline_score,
            after_score=baseline_score,
            accepted=False,
            patch=patch,
            artifacts={
                "stage": 1,
                "reason": "Diff application produced no changes (SEARCH block mismatch)",
            },
            delta=0.0,
            failure_reason="search_mismatch",
        )

    # --- Stage 2: 烟雾测试 — 用少量样本快速验证 ---
    temp_dir = write_to_temp_dir(modified_files, source_dir)
    try:
        target = target_factory(temp_dir)

        smoke_n = min(_SMOKE_SAMPLES, len(sampled))
        smoke_samples = sampled[:smoke_n]
        try:
            smoke_result = eval_fn(target, smoke_samples)
            smoke_score = smoke_result["score"]
        except Exception:
            print("[Evaluator] Stage 2 FAIL: 烟雾测试异常")
            return EvalResult(
                before_score=baseline_score,
                after_score=0.0,
                accepted=False,
                patch=patch,
                error=traceback.format_exc(),
                artifacts={
                    "stage": 2,
                    "reason": "Smoke test crashed",
                    "error": traceback.format_exc()[-500:],
                },
                delta=-baseline_score,
                failure_reason="smoke_crash",
            )

        if smoke_score < baseline_score * _SMOKE_THRESHOLD:
            print(
                f"[Evaluator] Stage 2 FAIL: smoke={smoke_score:.2f} < baseline*{_SMOKE_THRESHOLD}={baseline_score * _SMOKE_THRESHOLD:.2f}"
            )
            return EvalResult(
                before_score=baseline_score,
                after_score=smoke_score,
                accepted=False,
                patch=patch,
                artifacts={
                    "stage": 2,
                    "reason": f"Smoke test failed: {smoke_score:.2f} < {baseline_score * _SMOKE_THRESHOLD:.2f}",
                    "smoke_score": smoke_score,
                },
                delta=smoke_score - baseline_score,
                failure_reason="smoke_regression",
            )

        # --- Stage 3: 全量评估 (with progressive sampling) ---
        remaining_samples = sampled[smoke_n:]  # 排除已在 smoke 中用过的
        total_n = len(remaining_samples)

        if progressive and total_n > _PROGRESSIVE_FIRST_BATCH:
            # Progressive: 先评一小批，看能否提前决策
            first_batch = remaining_samples[:_PROGRESSIVE_FIRST_BATCH]
            first_result = eval_fn(target, first_batch)
            first_score = first_result["score"]
            first_details = first_result.get("details", [])

            early = should_stop_early(
                first_score, baseline_score, len(first_batch), total_n
            )

            if early == "reject":
                print(
                    f"[Evaluator] Stage 3 EARLY REJECT: "
                    f"batch_score={first_score:.2f} vs baseline={baseline_score:.2f} "
                    f"(saved {total_n - len(first_batch)} samples)"
                )
                return EvalResult(
                    before_score=baseline_score,
                    after_score=first_score,
                    accepted=False,
                    patch=patch,
                    details=first_details,
                    artifacts={
                        "stage": 3,
                        "reason": "Progressive early reject",
                        "smoke_score": smoke_score,
                        "progressive_batch_score": first_score,
                        "samples_saved": total_n - len(first_batch),
                    },
                    delta=first_score - baseline_score,
                    failure_reason="no_improvement",
                )

            if early == "accept":
                # 明确优于 baseline，直接接受
                print(
                    f"[Evaluator] Stage 3 EARLY ACCEPT: "
                    f"batch_score={first_score:.2f} vs baseline={baseline_score:.2f} "
                    f"(saved {total_n - len(first_batch)} samples)"
                )
                commit_to_source(modified_files, source_dir, layer_context=layer_context)
                return EvalResult(
                    before_score=baseline_score,
                    after_score=first_score,
                    accepted=True,
                    patch=patch,
                    details=first_details,
                    artifacts={
                        "stage": 3,
                        "reason": "Progressive early accept",
                        "smoke_score": smoke_score,
                        "progressive_batch_score": first_score,
                        "samples_saved": total_n - len(first_batch),
                    },
                    delta=first_score - baseline_score,
                )

            # 结果模糊 → 跑全量
            print(
                f"[Evaluator] Stage 3 progressive inconclusive "
                f"(batch={first_score:.2f}), running full evaluation"
            )

        # Full evaluation — 用完整 sampled 集合
        result = eval_fn(target, sampled)
        after_score = result["score"]
        details = result.get("details", [])

        accepted = after_score > baseline_score
        if accepted:
            commit_to_source(modified_files, source_dir, layer_context=layer_context)

        # 构建结构化 artifacts
        improved = [d for d in details if d.get("f1", 0) > 0.8]
        degraded = [d for d in details if d.get("f1", 1) < 0.3]
        artifacts = {
            "stage": 3,
            "smoke_score": smoke_score,
            "improved_samples": [
                {"question": d.get("question", "?")[:80], "f1": d.get("f1")}
                for d in improved[:5]
            ],
            "degraded_samples": [
                {"question": d.get("question", "?")[:80], "f1": d.get("f1")}
                for d in degraded[:5]
            ],
        }

        return EvalResult(
            before_score=baseline_score,
            after_score=after_score,
            accepted=accepted,
            patch=patch,
            details=details,
            artifacts=artifacts,
            delta=after_score - baseline_score,
            failure_reason=None if accepted else "no_improvement",
        )
    except Exception:
        return EvalResult(
            before_score=baseline_score,
            after_score=0.0,
            accepted=False,
            patch=patch,
            error=traceback.format_exc(),
            artifacts={
                "stage": 3,
                "reason": "Full evaluation crashed",
                "error": traceback.format_exc()[-500:],
            },
            delta=-baseline_score,
            failure_reason="eval_crash",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        parent = temp_dir.rstrip("/")
        parent_dir = os.path.dirname(parent)
        if os.path.basename(parent_dir).startswith("noa_eval_"):
            shutil.rmtree(parent_dir, ignore_errors=True)
