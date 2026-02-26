"""Evaluator — 临时目录沙盒验证 patch，决定接受或回滚。"""

from __future__ import annotations

import random
import shutil
from typing import Callable

from noa.core.protocol import DeltaPatch, EvalResult, SourceFile
from noa.diff_utils import (
    apply_diffs_in_memory,
    write_to_temp_dir,
    commit_to_source,
)


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
) -> EvalResult:
    """内存 diff → 临时目录 → 全新加载评估 → 接受则写回。"""
    # 采样评估集（固定 seed，所有轮次在同一子集上评估）
    rng = random.Random(seed)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    # 1. 在内存中应用 diff
    modified_files = apply_diffs_in_memory(source_files, patch.diffs)
    if not modified_files:
        print("[Evaluator] Diff 未产生任何修改")
        return EvalResult(
            before_score=baseline_score,
            after_score=baseline_score,
            accepted=False,
            patch=patch,
        )

    # 2. 写入临时目录
    temp_dir = write_to_temp_dir(modified_files, source_dir)

    try:
        # 3. 从临时目录全新加载 target
        target = target_factory(temp_dir)

        # 4. 评估
        result = eval_fn(target, sampled)
        after_score = result["score"]

        # 5. 决策
        accepted = after_score > baseline_score
        if accepted:
            commit_to_source(modified_files, source_dir)

        return EvalResult(
            before_score=baseline_score,
            after_score=after_score,
            accepted=accepted,
            patch=patch,
            details=result.get("details", []),
        )
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        # temp_dir 是 copytree 创建的目标目录，其父目录是 mkdtemp 创建的
        parent = temp_dir.rstrip("/")
        import os
        parent_dir = os.path.dirname(parent)
        if os.path.basename(parent_dir).startswith("noa_eval_"):
            shutil.rmtree(parent_dir, ignore_errors=True)
