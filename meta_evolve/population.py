"""种群管理：island model + 精英保留 + softmax 选择 + 多样性采样。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Individual:
    id: str
    code: str  # EVOLVE-BLOCK 部分
    score: float  # combined_score
    generation: int
    parent_id: Optional[str] = None
    island_id: int = 0


class Population:
    def __init__(self):
        self.individuals: dict[str, Individual] = {}

    def add(self, ind: Individual):
        self.individuals[ind.id] = ind

    def size(self) -> int:
        return len(self.individuals)

    def all_sorted(self) -> list[Individual]:
        return sorted(self.individuals.values(), key=lambda x: x.score, reverse=True)

    def best(self) -> Optional[Individual]:
        if not self.individuals:
            return None
        return max(self.individuals.values(), key=lambda x: x.score)

    def scores(self) -> np.ndarray:
        return np.array([ind.score for ind in self.individuals.values()])

    def island_members(self, island_id: int) -> list[Individual]:
        return [ind for ind in self.individuals.values() if ind.island_id == island_id]

    def active_islands(self) -> set[int]:
        return {ind.island_id for ind in self.individuals.values()}

    def select_parent(
        self, temperature: float, exploration_rate: float, island_id: int | None = None
    ) -> Individual:
        if island_id is not None:
            inds = self.island_members(island_id)
            if not inds:
                inds = list(self.individuals.values())
        else:
            inds = list(self.individuals.values())
        if len(inds) == 1:
            return inds[0]

        if np.random.random() < exploration_rate:
            return inds[np.random.randint(len(inds))]

        scores = np.array([ind.score for ind in inds])
        logits = scores / max(temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = np.random.choice(len(inds), p=probs)
        return inds[idx]

    def select_context(
        self, n: int, diversity_weight: float, island_id: int | None = None
    ) -> list[Individual]:
        if island_id is not None:
            local = sorted(
                self.island_members(island_id), key=lambda x: x.score, reverse=True
            )
            global_sorted = self.all_sorted()
            n_local = max(1, int(n * 0.6))
            candidates = local[:n_local]
            for ind in global_sorted:
                if len(candidates) >= n:
                    break
                if ind.id not in {c.id for c in candidates}:
                    candidates.append(ind)
            inds = candidates[:n]
        else:
            inds = self.all_sorted()

        if len(inds) <= n:
            return inds

        n_top = max(1, int(n * (1 - diversity_weight)))
        n_random = n - n_top
        selected = inds[:n_top]
        remaining = inds[n_top:]
        if remaining and n_random > 0:
            idxs = np.random.choice(
                len(remaining), size=min(n_random, len(remaining)), replace=False
            )
            selected.extend(remaining[i] for i in idxs)
        return selected[:n]

    def migrate(self, migration_rate: float, num_islands: int):
        """岛屿间迁移：每个岛的 top-k 复制到随机其他岛。"""
        if num_islands <= 1:
            return
        for src in range(num_islands):
            members = self.island_members(src)
            if not members:
                continue
            members.sort(key=lambda x: x.score, reverse=True)
            n_migrate = max(1, int(len(members) * migration_rate))
            top = members[:n_migrate]
            dst = (src + np.random.randint(1, num_islands)) % num_islands
            for ind in top:
                clone = Individual(
                    id=self.make_id(),
                    code=ind.code,
                    score=ind.score,
                    generation=ind.generation,
                    parent_id=ind.id,
                    island_id=dst,
                )
                self.add(clone)

    def prune(self, max_size: int, elite_ratio: float, num_islands: int = 1):
        """Island-aware prune：每个岛独立保留配额，不互相挤占。"""
        if self.size() <= max_size:
            return

        if num_islands <= 1:
            self._prune_flat(max_size, elite_ratio)
            return

        # 每个岛的配额按成员数比例分配，但保底每岛至少 2 个
        islands = self.active_islands()
        island_counts = {isl: len(self.island_members(isl)) for isl in islands}
        total = sum(island_counts.values())
        min_per_island = 2

        # 先分配保底
        quotas: dict[int, int] = {}
        remaining_budget = max_size
        for isl in islands:
            quotas[isl] = min_per_island
            remaining_budget -= min_per_island

        # 剩余按比例分配
        remaining_budget = max(0, remaining_budget)
        if total > 0 and remaining_budget > 0:
            for isl in islands:
                extra = int(remaining_budget * island_counts[isl] / total)
                quotas[isl] += extra

        # 每个岛独立 prune
        keep = set()
        for isl in islands:
            members = sorted(
                self.island_members(isl), key=lambda x: x.score, reverse=True
            )
            quota = quotas.get(isl, min_per_island)
            n_elite = max(1, int(quota * elite_ratio))
            # 保留精英
            for ind in members[:n_elite]:
                keep.add(ind.id)
            # 剩余随机选
            rest = members[n_elite:]
            n_rest = quota - n_elite
            if rest and n_rest > 0:
                idxs = np.random.choice(
                    len(rest), size=min(n_rest, len(rest)), replace=False
                )
                for idx in idxs:
                    keep.add(rest[idx].id)

        self.individuals = {k: v for k, v in self.individuals.items() if k in keep}

    def _prune_flat(self, max_size: int, elite_ratio: float):
        """无 island 的全局 prune（fallback）。"""
        inds = self.all_sorted()
        n_elite = max(1, int(max_size * elite_ratio))
        keep = set(ind.id for ind in inds[:n_elite])
        remaining = [ind for ind in inds[n_elite:] if ind.id not in keep]
        n_rest = max_size - n_elite
        if remaining and n_rest > 0:
            idxs = np.random.choice(
                len(remaining), size=min(n_rest, len(remaining)), replace=False
            )
            for idx in idxs:
                keep.add(remaining[idx].id)
        self.individuals = {k: v for k, v in self.individuals.items() if k in keep}

    def top_k(self, k: int) -> list[Individual]:
        return self.all_sorted()[:k]

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:8]
