"""种群状态描述符：数值特征提取。"""

from __future__ import annotations

import numpy as np

from .population import Population

NUMERIC_STATE_DIM = 8


def extract_numeric_state(population: Population, recent_k: int = 5) -> np.ndarray:
    """提取 8 维数值状态向量。

    [best_fitness, mean_fitness, std_fitness,
     improvement_rate, stagnation_count, generation_depth,
     population_coverage, fitness_entropy]
    """
    if population.size() == 0:
        return np.zeros(NUMERIC_STATE_DIM, dtype=np.float32)

    scores = population.scores()
    inds = population.all_sorted()

    best = scores.max()
    mean = scores.mean()
    std = scores.std() if len(scores) > 1 else 0.0

    # improvement_rate: 最近 recent_k 个 vs 之前的平均提升
    gens = [ind.generation for ind in inds]
    max_gen = max(gens) if gens else 0
    recent = [ind.score for ind in inds if ind.generation >= max_gen - recent_k]
    older = [ind.score for ind in inds if ind.generation < max_gen - recent_k]
    improvement_rate = (np.mean(recent) - np.mean(older)) if older else 0.0

    # stagnation: 最近 recent_k 代中 best 没有提升的代数
    gen_bests: dict[int, float] = {}
    for ind in inds:
        g = ind.generation
        gen_bests[g] = max(gen_bests.get(g, -np.inf), ind.score)
    sorted_gens = sorted(gen_bests.keys())
    stagnation = 0
    if len(sorted_gens) >= 2:
        running_best = gen_bests[sorted_gens[0]]
        for g in sorted_gens[1:]:
            if gen_bests[g] > running_best + 1e-8:
                running_best = gen_bests[g]
                stagnation = 0
            else:
                stagnation += 1

    generation_depth = float(max_gen)

    # population_coverage: 分数分布的范围归一化
    score_range = scores.max() - scores.min() if len(scores) > 1 else 0.0
    coverage = score_range / max(abs(scores.max()), 1e-6)

    # fitness_entropy: 离散化 score 分布的熵
    if len(scores) > 1:
        n_bins = min(10, len(scores))
        hist, _ = np.histogram(scores, bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
    else:
        entropy = 0.0

    return np.array(
        [
            best,
            mean,
            std,
            improvement_rate,
            float(stagnation),
            generation_depth,
            coverage,
            entropy,
        ],
        dtype=np.float32,
    )
