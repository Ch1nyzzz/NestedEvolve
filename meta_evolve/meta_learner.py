"""外层 meta-learning：Evolution Strategies 直接优化真实 validation score。

surrogate 保留用于：
  1. warmup 阶段收集 transitions 训练
  2. 预筛选 φ 扰动（可选，省 rollout 开销）
  3. 提供辅助信号（打印诊断）

φ 更新完全由 ES 梯度估计驱动，不依赖 surrogate 梯度。
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

import numpy as np

from .evolve_loop import run_inner_loop
from .llm_client import LLMClient
from .strategy import StrategyParams
from .surrogate import SurrogateModel
from .task_adapter import Task


@dataclass
class _PerturbationResult:
    """一个 φ 扰动的 rollout 结果。"""

    epsilon: np.ndarray  # 扰动向量
    params: StrategyParams  # 实际使用的参数
    scores: dict[str, float]  # {task_name: best_score}
    mean_score: float  # 跨任务平均 score
    transitions: list  # 给 surrogate 训练用


class MetaLearner:
    def __init__(
        self,
        surrogate: SurrogateModel,
        semantic_analyzer,
        train_tasks: list[Task],
        test_tasks: list[Task],
        llm: LLMClient,
        config: dict,
        n_meta_steps: int = 30,
        tasks_per_step: int = 4,
        adaptation_steps: int = 3,  # ES 不用，保留给 held-out adapt
        adaptation_lr: float = 0.01,  # ES 不用，保留给 held-out adapt
        meta_lr: float = 0.001,  # ES 学习率
        validation_steps: int = 5,  # ES 不用（initial = validation）
        warmup_steps: int = 5,
        gradient_clip: float = 1.0,
        max_concurrency: int = 50,
    ):
        self.surrogate = surrogate
        self.semantic_analyzer = semantic_analyzer
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.llm = llm
        self.config = config

        self.n_meta_steps = n_meta_steps
        self.tasks_per_step = tasks_per_step
        self.K = adaptation_steps
        self.alpha = adaptation_lr
        self.warmup_steps = warmup_steps

        # φ_init: 当前最优策略向量
        self.phi_init = StrategyParams().to_vector()

        # ES 超参
        self.es_lr = meta_lr
        self.es_sigma = 1.0  # 扰动标准差（足够大以产生明显不同的策略）
        self.es_n_candidates = 30  # 采样候选扰动数
        self.es_n_rollouts = 10  # 经 surrogate 筛选后实际跑的 rollout 数
        self.gradient_clip = gradient_clip

        self.inner_steps = config.get("inner_loop", {}).get("n_steps", 15)
        self.max_pop_size = config.get("inner_loop", {}).get("max_pop_size", 20)
        self.analyze_interval = config.get("inner_loop", {}).get("analyze_interval", 5)
        self.surrogate_train_epochs = config.get("surrogate", {}).get(
            "train_epochs", 10
        )

        self.global_llm_sem = asyncio.Semaphore(max_concurrency)

    async def _rollout(
        self,
        step: int,
        phi_vec: np.ndarray,
        tasks: list[Task],
    ) -> _PerturbationResult:
        """用给定 φ 在多个任务上跑 inner loop，返回平均 score。"""
        params = StrategyParams.from_vector(phi_vec)
        epsilon = phi_vec - self.phi_init  # 相对于 phi_init 的偏移

        results = await asyncio.gather(
            *[
                run_inner_loop(
                    task=task,
                    params=params,
                    n_steps=self.inner_steps,
                    llm=self.llm,
                    semantic_analyzer=self.semantic_analyzer,
                    max_pop_size=self.max_pop_size,
                    analyze_interval=self.analyze_interval,
                    global_sem=self.global_llm_sem,
                )
                for task in tasks
            ]
        )

        scores = {
            task.name: result.final_best_score for task, result in zip(tasks, results)
        }
        mean_score = float(np.mean(list(scores.values())))
        transitions = [t for r in results for t in r.transitions]

        return _PerturbationResult(
            epsilon=epsilon,
            params=params,
            scores=scores,
            mean_score=mean_score,
            transitions=transitions,
        )

    def _es_update(self, perturbation_results: list[_PerturbationResult]):
        """ES 梯度估计 + φ 更新。

        使用 antithetic sampling 的梯度估计:
        ∇_φ J ≈ (1/nσ) Σ_k score_k * ε_k

        score 做 rank-based fitness shaping 减少方差。
        """
        n = len(perturbation_results)
        if n == 0:
            return

        # rank-based fitness shaping (OpenAI ES 论文)
        scores = np.array([r.mean_score for r in perturbation_results])
        ranks = np.zeros(n)
        order = np.argsort(scores)
        for i, idx in enumerate(order):
            ranks[idx] = i
        # 归一化到 [-0.5, 0.5]
        shaped = (ranks / (n - 1) - 0.5) if n > 1 else np.zeros(n)

        # 梯度估计
        epsilons = np.array([r.epsilon for r in perturbation_results])
        grad = np.dot(shaped, epsilons) / (n * self.es_sigma)

        # clip
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.gradient_clip:
            grad = grad * (self.gradient_clip / grad_norm)

        # 更新
        self.phi_init = self.phi_init + self.es_lr * grad

    def _surrogate_prescreen(
        self, candidates: list[np.ndarray], task_ids: list[int], top_k: int
    ) -> list[np.ndarray]:
        """用 surrogate 对候选 φ 排序，返回预测最优的 top_k 个。

        warmup 阶段 surrogate 不准，直接随机选。
        """
        if self.surrogate._running_count < 50:
            # surrogate 数据不够，随机选
            idxs = np.random.choice(
                len(candidates), size=min(top_k, len(candidates)), replace=False
            )
            return [candidates[i] for i in idxs]

        # 用零状态做粗预测（状态未知时的先验）
        zero_state = np.zeros(self.surrogate.state_dim, dtype=np.float32)
        scores = []
        for phi in candidates:
            # 对所有 task_ids 取平均预测
            pred = np.mean(
                [self.surrogate.predict(phi, zero_state, tid) for tid in task_ids]
            )
            scores.append(pred)

        # 取 top_k
        top_idxs = np.argsort(scores)[-top_k:]
        return [candidates[i] for i in top_idxs]

    async def meta_train(self):
        """ES-driven meta-training 主循环。"""
        print(
            f"Meta-training (ES): {self.n_meta_steps} 步, "
            f"{len(self.train_tasks)} 训练任务, "
            f"{self.es_n_candidates} candidates → {self.es_n_rollouts} rollouts/step"
        )

        best_score = -float("inf")
        best_phi = self.phi_init.copy()

        for step in range(self.n_meta_steps):
            t0 = time.time()
            is_warmup = step < self.warmup_steps

            k = min(self.tasks_per_step, len(self.train_tasks))
            sampled_tasks = random.sample(self.train_tasks, k)
            task_ids = [t.task_id for t in sampled_tasks]

            # === 采样候选扰动（antithetic 成对） ===
            candidates = []
            for _ in range(self.es_n_candidates):
                eps = (
                    np.random.randn(StrategyParams.PHI_DIM).astype(np.float32)
                    * self.es_sigma
                )
                candidates.append(self.phi_init + eps)
                candidates.append(self.phi_init - eps)

            if is_warmup:
                candidates = [
                    np.random.randn(StrategyParams.PHI_DIM).astype(np.float32) * 1.0
                    for _ in range(self.es_n_candidates * 2)
                ]

            # === surrogate 预筛选 → 只跑 top-N ===
            perturbations = self._surrogate_prescreen(
                candidates, task_ids, self.es_n_rollouts
            )

            # === 并行 rollout ===
            rollout_results = await asyncio.gather(
                *[self._rollout(step, phi, sampled_tasks) for phi in perturbations]
            )

            # === 收集 transitions 训练 surrogate ===
            all_transitions = [t for r in rollout_results for t in r.transitions]
            surrogate_loss = self.surrogate.update(
                all_transitions,
                epochs=self.surrogate_train_epochs,
            )

            # === 跟踪 best ===
            for r in rollout_results:
                if r.mean_score > best_score:
                    best_score = r.mean_score
                    best_phi = (self.phi_init + r.epsilon).copy()

            scores = [r.mean_score for r in rollout_results]
            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))

            if is_warmup:
                elapsed = time.time() - t0
                print(
                    f"[Meta step {step}] warmup, "
                    f"scores=[{mean_score:.4f}±{np.std(scores):.4f}, max={max_score:.4f}], "
                    f"surrogate_loss={surrogate_loss:.6f}, {elapsed:.1f}s"
                )
                continue

            # === ES 更新 φ_init ===
            self._es_update(rollout_results)

            elapsed = time.time() - t0
            phi_now = StrategyParams.from_vector(self.phi_init)
            print(
                f"[Meta step {step}] "
                f"scores=[{mean_score:.4f}±{np.std(scores):.4f}, max={max_score:.4f}], "
                f"best_ever={best_score:.4f}, "
                f"surrogate_loss={surrogate_loss:.6f}, "
                f"φ=[expl={phi_now.exploration_rate:.3f}, "
                f"temp={phi_now.selection_temperature:.3f}, "
                f"mut={phi_now.mutation_strength:.3f}, "
                f"err_anal={phi_now.error_analysis:.3f}, "
                f"islands={phi_now.num_islands}], "
                f"{elapsed:.1f}s"
            )

        # 用历史最优 φ
        self.phi_init = best_phi
        final_params = StrategyParams.from_vector(self.phi_init)
        print(f"\nMeta-learned φ*: {final_params}")
        return final_params

    async def _eval_single_task(
        self, task: Task, n_eval_steps: int
    ) -> tuple[str, dict]:
        """单个 held-out 任务评估：meta-learned φ* vs default vs random。"""
        print(f"\n评估 held-out: {task.name}")

        adapted_params = StrategyParams.from_vector(self.phi_init)
        random_phi = np.random.randn(StrategyParams.PHI_DIM).astype(np.float32) * 0.5
        random_params = StrategyParams.from_vector(random_phi)

        common = dict(
            task=task,
            n_steps=n_eval_steps,
            llm=self.llm,
            semantic_analyzer=self.semantic_analyzer,
            max_pop_size=self.max_pop_size,
            analyze_interval=self.analyze_interval,
            global_sem=self.global_llm_sem,
        )

        r_adapted, r_default, r_random = await asyncio.gather(
            run_inner_loop(params=adapted_params, **common),
            run_inner_loop(params=StrategyParams(), **common),
            run_inner_loop(params=random_params, **common),
        )

        task_results = {
            "meta_adapted": r_adapted.final_best_score,
            "default": r_default.final_best_score,
            "random": r_random.final_best_score,
            "baseline": task.baseline_score,
        }
        print(
            f"  {task.name}: adapted={r_adapted.final_best_score:.6f} "
            f"default={r_default.final_best_score:.6f} "
            f"random={r_random.final_best_score:.6f}"
        )
        return task.name, task_results

    async def evaluate_held_out(self, n_eval_steps: int = 20) -> dict:
        """在 test tasks 上并行评估。"""
        pairs = await asyncio.gather(
            *[self._eval_single_task(task, n_eval_steps) for task in self.test_tasks]
        )
        return dict(pairs)
