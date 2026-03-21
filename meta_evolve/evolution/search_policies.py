"""可插拔搜索策略：parent selection + context building 的多种实现。

search_policy dict 结构:
{
    "name": str,                         # 策略名
    "parent_selection": str,             # softmax | ucb | elite | tournament | random
    "context_mode": str,                 # default | diverse | top_only | recent_effective
    "temperature": float | None,         # 覆盖 LLM 温度
    "mutation_strength": float | None,   # 覆盖变异强度描述
    "context_size": int | None,          # 覆盖 context 数量
    "crossover_rate": float | None,      # 覆盖交叉率
    "exploration_rate": float | None,    # 覆盖探索率
    "max_retries": int,                  # retry 次数
    "prompt_supplement": str,            # 额外注入 mutation prompt 的文本
}
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .population import Individual, Population
from .strategy import StrategyParams


def select_parent(
    population: Population,
    policy: dict,
    params: StrategyParams,
    island_id: int,
    rng=None,
) -> Individual:
    _rng = rng if rng is not None else np.random
    mode = policy.get("parent_selection", "softmax")
    expl = policy.get("exploration_rate", params.exploration_rate)
    temp = policy.get("selection_temperature", params.selection_temperature)

    if mode == "ucb":
        return _ucb_select(population, island_id, _rng)
    if mode == "elite":
        return _elite_select(population, island_id, _rng)
    if mode == "tournament":
        return _tournament_select(population, island_id, _rng)
    if mode == "random":
        return population.select_parent(1.0, 1.0, island_id, _rng)
    # default: softmax
    return population.select_parent(temp, expl, island_id, _rng)


def select_context(
    population: Population,
    policy: dict,
    params: StrategyParams,
    island_id: int,
) -> list[Individual]:
    mode = policy.get("context_mode", "default")
    n = policy.get("context_size", params.context_size)
    div_w = params.diversity_weight

    if mode == "diverse":
        return _diverse_context(population, n, island_id)
    if mode == "top_only":
        return population.top_k(n)
    if mode == "recent_effective":
        return _recent_effective_context(population, n, island_id)
    # default
    return population.select_context(n, div_w, island_id)


def apply_policy_overrides(params: StrategyParams, policy: dict) -> StrategyParams:
    """从 search_policy 提取数值覆盖，返回新的 params（不修改原始）。"""
    import dataclasses
    new_params = dataclasses.replace(params)
    for key in ("temperature", "mutation_strength", "crossover_rate",
                "exploration_rate", "context_size"):
        val = policy.get(key)
        if val is not None:
            target_key = "llm_temperature" if key == "temperature" else key
            if hasattr(new_params, target_key):
                setattr(new_params, target_key, val)
    return new_params


# ============================================================
# Parent selection 策略
# ============================================================

# 简单的全局 visit counter（按 individual id 计数）
_visit_counts: Counter = Counter()
_VISIT_MAX = 5000


def _ucb_select(
    population: Population,
    island_id: int,
    rng,
    c: float = 1.4,
) -> Individual:
    """UCB1 选择：平衡利用和探索。"""
    inds = population.island_members(island_id)
    if not inds:
        inds = list(population.individuals.values())
    if len(inds) == 1:
        return inds[0]

    total = sum(_visit_counts[ind.id] for ind in inds) + 1
    scores = []
    for ind in inds:
        visits = _visit_counts[ind.id] + 1
        exploit = ind.score
        explore = c * math.sqrt(math.log(total) / visits)
        scores.append(exploit + explore)

    best_idx = int(np.argmax(scores))
    chosen = inds[best_idx]
    _visit_counts[chosen.id] += 1
    if len(_visit_counts) > _VISIT_MAX:
        _visit_counts.clear()
    return chosen


def _elite_select(
    population: Population,
    island_id: int,
    rng,
    top_k: int = 3,
) -> Individual:
    """从 top-k 精英中随机选。"""
    inds = population.island_members(island_id)
    if not inds:
        inds = list(population.individuals.values())
    inds.sort(key=lambda x: x.score, reverse=True)
    elite = inds[:min(top_k, len(inds))]
    return elite[rng.randint(len(elite))]


def _tournament_select(
    population: Population,
    island_id: int,
    rng,
    k: int = 3,
) -> Individual:
    """锦标赛选择：随机取 k 个，选最好的。"""
    inds = population.island_members(island_id)
    if not inds:
        inds = list(population.individuals.values())
    if len(inds) <= k:
        return max(inds, key=lambda x: x.score)
    idxs = rng.choice(len(inds), size=min(k, len(inds)), replace=False)
    candidates = [inds[i] for i in idxs]
    return max(candidates, key=lambda x: x.score)


# ============================================================
# Context building 策略
# ============================================================


def _diverse_context(
    population: Population,
    n: int,
    island_id: int,
) -> list[Individual]:
    """多样性优先：先选最优，然后贪心选和已选集合最不同的。"""
    all_inds = population.all_sorted()
    if len(all_inds) <= n:
        return all_inds

    selected = [all_inds[0]]  # 最优
    remaining = all_inds[1:]

    while len(selected) < n and remaining:
        best_dist = -1
        best_idx = 0
        for i, cand in enumerate(remaining):
            min_dist = min(_code_distance(cand, s) for s in selected)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def _recent_effective_context(
    population: Population,
    n: int,
    island_id: int,
) -> list[Individual]:
    """选最近加入且分数高的个体（generation 高 + score 高）。"""
    inds = list(population.individuals.values())
    # 按 (generation, score) 降序
    inds.sort(key=lambda x: (x.generation, x.score), reverse=True)
    return inds[:n]


def _code_distance(a: Individual, b: Individual) -> float:
    """简单的代码距离：基于行集合的 Jaccard 距离。"""
    lines_a = set(a.code.strip().splitlines())
    lines_b = set(b.code.strip().splitlines())
    if not lines_a and not lines_b:
        return 0.0
    intersection = len(lines_a & lines_b)
    union = len(lines_a | lines_b)
    return 1.0 - intersection / max(union, 1)
