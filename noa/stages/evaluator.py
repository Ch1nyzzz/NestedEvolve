"""Evaluator — 级联沙盒验证 patch，决定接受或回滚。"""

from __future__ import annotations

import json
import os
import random
import shutil
import traceback
from typing import Callable

from utils.llm import llm_call, DEFAULT_MODEL
from noa.core.protocol import DeltaPatch, EvalResult, SourceFile, StructuredPatch
from noa.diff_utils import (
    apply_diffs_in_memory,
    write_to_temp_dir,
    commit_to_source,
)
from noa.patch_protocol import apply_patch_ops

_SMOKE_SAMPLES = 3
_SMOKE_THRESHOLD = 0.8  # 烟雾测试得分 < baseline * 0.8 直接拒绝


def evaluate(
    source_files: list[SourceFile],
    source_dir: str,
    patch: DeltaPatch | StructuredPatch,
    dataset: list,
    eval_fn,
    target_factory: Callable[[str], object],
    baseline_score: float,
    n_samples: int = 20,
    seed: int = 42,
    layer_context=None,
    auto_commit: bool = True,
) -> EvalResult:
    """级联评估：Stage1 语法检查 → Stage2 烟雾测试 → Stage3 全量评估。"""
    rng = random.Random(seed)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    # --- Stage 1: 语法检查 — diff 能否正确应用 ---
    if isinstance(patch, StructuredPatch):
        modified_files, patch_errors, _ = apply_patch_ops(source_files, patch.ops)
        if patch_errors:
            print(
                f"[Evaluator] Stage 1 FAIL: StructuredPatch 验证失败 ({len(patch_errors)} errors)"
            )
            return EvalResult(
                before_score=baseline_score,
                after_score=baseline_score,
                accepted=False,
                patch=patch,
                artifacts={
                    "stage": 1,
                    "reason": "StructuredPatch validation failed",
                    "errors": [
                        {"op_index": e.op_index, "code": e.code, "message": e.message}
                        for e in patch_errors
                    ],
                },
                delta=0.0,
                failure_reason="patch_validation_error",
            )
    else:
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

        # --- Stage 3: 全量评估 ---
        result = eval_fn(target, sampled)
        after_score = result["score"]
        details = result.get("details", [])

        accepted = after_score > baseline_score
        if accepted and auto_commit:
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


def generate_eval_feedback(
    eval_result: EvalResult,
    baseline_details: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """用 LLM 分析 patch 评估前后的样本级变化，生成结构化反馈。"""
    if not eval_result.details or not baseline_details:
        return {}

    # 构建 before/after 对比表
    before_map = {d.get("question", ""): d.get("f1", 0) for d in baseline_details}
    comparison = []
    for d in eval_result.details:
        q = d.get("question", "")
        after_f1 = d.get("f1", 0)
        before_f1 = before_map.get(q, 0)
        delta = after_f1 - before_f1
        if abs(delta) > 0.05:
            comparison.append(
                {
                    "question": q[:100],
                    "before_f1": round(before_f1, 3),
                    "after_f1": round(after_f1, 3),
                    "delta": round(delta, 3),
                }
            )

    if not comparison:
        return {
            "summary": "No significant sample-level changes.",
            "improved": [],
            "degraded": [],
            "insights": [],
        }

    comparison.sort(key=lambda x: x["delta"])
    prompt = (
        "Analyze the per-sample F1 changes after a code patch was applied.\n\n"
        f"Patch rationale: {eval_result.patch.rationale if eval_result.patch else 'N/A'}\n"
        f"Overall: before={eval_result.before_score:.2f}, after={eval_result.after_score:.2f}, "
        f"accepted={eval_result.accepted}\n\n"
        f"Sample-level changes (sorted by delta):\n"
        f"{json.dumps(comparison[:20], ensure_ascii=False, indent=1)}\n\n"
        "Output JSON with:\n"
        '1. "improved_types": list of question/task types that improved and why\n'
        '2. "degraded_types": list of question/task types that degraded and why\n'
        '3. "insights": specific observations about what the patch actually changed\n'
        '4. "next_focus": what the next optimization round should focus on\n'
        '5. "causal_links": any suspected causal relationships between patterns\n'
    )
    try:
        resp = llm_call(
            prompt,
            model=model,
            max_tokens=2048,
            temperature=0,
            system="You are an evaluation analyst. Output strict JSON.",
        )
        text = resp.text
        # 提取 JSON
        left, right = text.find("{"), text.rfind("}")
        if left != -1 and right != -1:
            return json.loads(text[left : right + 1])
    except Exception:
        pass
    return {}


def extract_patch_regions(
    patch: DeltaPatch, source_files: list[SourceFile]
) -> list[tuple[str, int, int]]:
    """提取 patch 修改的文件区域 [(file_path, start_line, end_line), ...]。"""
    file_map = {sf.path: sf.content for sf in source_files}
    regions = []
    for diff in patch.diffs:
        content = file_map.get(diff.file_path, "")
        if not content:
            continue
        idx = content.find(diff.search)
        if idx == -1:
            continue
        start_line = content[:idx].count("\n")
        end_line = start_line + diff.search.count("\n")
        regions.append((diff.file_path, start_line, end_line))
    return regions


def merge_patches(
    source_files: list[SourceFile], patches: list[DeltaPatch]
) -> list[SourceFile] | None:
    """将多个不冲突的 patch 顺序应用到 source_files，返回合并后的文件列表。"""
    current = list(source_files)
    for patch in patches:
        modified = apply_diffs_in_memory(current, patch.diffs)
        if not modified:
            return None
        # 更新 current 中被修改的文件
        mod_map = {sf.path: sf for sf in modified}
        current = [mod_map.get(sf.path, sf) for sf in current]
    return current
