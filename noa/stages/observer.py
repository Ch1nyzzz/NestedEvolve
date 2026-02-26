"""Observer — 运行目标系统，收集执行轨迹。"""

from __future__ import annotations

import random

from noa.core.protocol import Trajectory


def observe(
    target,
    dataset: list,
    n_samples: int = 20,
    seed: int = 42,
    *,
    score_fn,
) -> list[Trajectory]:
    """对采样数据运行 pipeline，收集完整执行轨迹。

    Args:
        score_fn: 单样本评分函数 (prediction, ground_truth) -> float，必传。
    """

    rng = random.Random(seed)
    sampled = rng.sample(dataset, min(n_samples, len(dataset)))

    trajectories = []
    for ex in sampled:
        result = target(ex.question)
        score = score_fn(result.answer, ex.answer)
        trajectories.append(Trajectory(
            question=ex.question,
            ground_truth=ex.answer,
            prediction=result.answer,
            f1=score,
            intermediate=result.intermediate,
        ))
    return trajectories
