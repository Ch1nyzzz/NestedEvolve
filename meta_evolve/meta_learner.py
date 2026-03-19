"""外层 meta-learning：ES φ优化 + Skill-based 路由学习。

包含两个 meta-learner：
1. MetaLearner — ES 优化 φ 向量（原有）
2. SkillMetaLearner — Skill Library 路由学习（新增）
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

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

        # φ: current 用于 ES 累积更新，best 只做 checkpoint
        self.phi_current = StrategyParams().to_vector()
        self.phi_best = self.phi_current.copy()
        self.best_score = -float("inf")

        # ES 超参
        self.es_lr = meta_lr
        self.es_sigma = 0.3  # logit 空间 ±0.3 → 参数空间合理扰动
        self.es_n_pairs = 10  # antithetic 对数（= 20 个 rollout）
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
        rng_seed: int | None = None,
    ) -> _PerturbationResult:
        """用给定 φ 在多个任务上跑 inner loop，返回平均 score。"""
        params = StrategyParams.from_vector(phi_vec)
        epsilon = phi_vec - self.phi_current

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
                    rng_seed=rng_seed + i * 1000 if rng_seed is not None else None,
                )
                for i, task in enumerate(tasks)
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

    def _es_update_paired(
        self,
        pair_results: list[tuple[_PerturbationResult, _PerturbationResult]],
        epsilons: list[np.ndarray],
    ):
        """成对差分 ES 梯度估计。

        每对 (R+, R-) 对应同一个 ε：
        grad ≈ (1/n) Σ_k (R+_k - R-_k) / (2σ) * ε_k

        不做 rank shaping——直接用分数差，保留尺度信息。
        """
        n = len(pair_results)
        if n == 0:
            return

        grad = np.zeros(StrategyParams.PHI_DIM, dtype=np.float32)
        for (r_plus, r_minus), eps in zip(pair_results, epsilons):
            diff = r_plus.mean_score - r_minus.mean_score
            grad += diff * eps / (2.0 * self.es_sigma)
        grad /= n

        # clip
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.gradient_clip:
            grad = grad * (self.gradient_clip / grad_norm)

        # 更新 current（不碰 best）
        self.phi_current = self.phi_current + self.es_lr * grad

    async def meta_train(self):
        """ES-driven meta-training：antithetic 成对差分，current/best 分离。"""
        print(
            f"Meta-training (ES): {self.n_meta_steps} 步, "
            f"{len(self.train_tasks)} 训练任务, "
            f"{self.es_n_pairs} pairs/step (={self.es_n_pairs * 2} rollouts)"
        )

        for step in range(self.n_meta_steps):
            t0 = time.time()
            is_warmup = step < self.warmup_steps

            k = min(self.tasks_per_step, len(self.train_tasks))
            sampled_tasks = random.sample(self.train_tasks, k)

            # === 采样 antithetic 对 ===
            epsilons = []
            phi_plus_list = []
            phi_minus_list = []

            center = (
                self.phi_current
                if not is_warmup
                else np.zeros(StrategyParams.PHI_DIM, dtype=np.float32)
            )
            sigma = self.es_sigma if not is_warmup else 1.0

            for _ in range(self.es_n_pairs):
                eps = np.random.randn(StrategyParams.PHI_DIM).astype(np.float32) * sigma
                epsilons.append(eps)
                phi_plus_list.append(center + eps)
                phi_minus_list.append(center - eps)

            # === 并行 rollout 所有 ±ε（成对共享 rng_seed）===
            rollout_coros = []
            for pair_idx, (phi_p, phi_m) in enumerate(
                zip(phi_plus_list, phi_minus_list)
            ):
                # +ε 和 -ε 用相同 seed → parent 选择一致 → 差分只反映 φ 差异
                pair_seed = step * 10000 + pair_idx * 100
                rollout_coros.append(
                    self._rollout(step, phi_p, sampled_tasks, rng_seed=pair_seed)
                )
                rollout_coros.append(
                    self._rollout(step, phi_m, sampled_tasks, rng_seed=pair_seed)
                )

            all_results = await asyncio.gather(*rollout_coros)

            # 重组成对
            pair_results = []
            for i in range(self.es_n_pairs):
                r_plus = all_results[2 * i]
                r_minus = all_results[2 * i + 1]
                pair_results.append((r_plus, r_minus))

            # === 收集 transitions 训练 surrogate ===
            all_transitions = [t for r in all_results for t in r.transitions]
            surrogate_loss = self.surrogate.update(
                all_transitions,
                epochs=self.surrogate_train_epochs,
            )

            # === 跟踪 best（独立于 current）===
            for r in all_results:
                if r.mean_score > self.best_score:
                    self.best_score = r.mean_score
                    self.phi_best = (center + r.epsilon).copy()

            scores = [r.mean_score for r in all_results]
            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            pair_diffs = [rp.mean_score - rm.mean_score for rp, rm in pair_results]
            mean_diff = float(np.mean(np.abs(pair_diffs)))

            if is_warmup:
                elapsed = time.time() - t0
                print(
                    f"[Meta step {step}] warmup, "
                    f"scores=[{mean_score:.4f}±{np.std(scores):.4f}, max={max_score:.4f}], "
                    f"|pair_diff|={mean_diff:.4f}, "
                    f"surrogate_loss={surrogate_loss:.6f}, {elapsed:.1f}s"
                )
                continue

            # === ES 成对差分更新 ===
            self._es_update_paired(pair_results, epsilons)

            elapsed = time.time() - t0
            phi_now = StrategyParams.from_vector(self.phi_current)
            print(
                f"[Meta step {step}] "
                f"scores=[{mean_score:.4f}±{np.std(scores):.4f}, max={max_score:.4f}], "
                f"|pair_diff|={mean_diff:.4f}, "
                f"best_ever={self.best_score:.4f}, "
                f"surrogate_loss={surrogate_loss:.6f}, "
                f"φ=[expl={phi_now.exploration_rate:.3f}, "
                f"temp={phi_now.selection_temperature:.3f}, "
                f"mut={phi_now.mutation_strength:.3f}, "
                f"err_anal={phi_now.error_analysis:.3f}, "
                f"diff_rw={phi_now.diff_vs_rewrite:.3f}, "
                f"islands={phi_now.num_islands}], "
                f"{elapsed:.1f}s"
            )

        # 最终导出用 best
        final_params = StrategyParams.from_vector(self.phi_best)
        print(f"\nMeta-learned φ* (best): {final_params}")
        print(f"Final φ (current): {StrategyParams.from_vector(self.phi_current)}")
        return final_params

    async def _adapt_phi_on_task(
        self, task: Task, support_steps: int = 5
    ) -> StrategyParams:
        """Support → Adapt：在新任务上跑短 support phase，用 surrogate 梯度适应 φ。"""
        phi = self.phi_current.copy()
        params = StrategyParams.from_vector(phi)

        # support phase: 短 rollout 收集 state
        support_result = await run_inner_loop(
            task=task,
            params=params,
            n_steps=support_steps,
            llm=self.llm,
            semantic_analyzer=self.semantic_analyzer,
            max_pop_size=self.max_pop_size,
            analyze_interval=self.analyze_interval,
            global_sem=self.global_llm_sem,
        )

        # 用 support 结果的 final_state 做 adaptation
        state = support_result.final_state
        for _ in range(self.K):
            grad = self.surrogate.compute_gradient(phi, state, task.task_id)
            phi = phi + self.alpha * grad

        # 也把 support transitions 喂给 surrogate
        self.surrogate.update(support_result.transitions, epochs=3)

        return StrategyParams.from_vector(phi)

    async def _eval_single_task(
        self, task: Task, n_eval_steps: int
    ) -> tuple[str, dict]:
        """单个 held-out 任务评估：support→adapt→query vs default vs random。"""
        print(f"\n评估 held-out: {task.name}")

        support_steps = min(5, n_eval_steps // 3)
        query_steps = n_eval_steps - support_steps

        # meta-adapted: support → adapt φ → query
        adapted_params = await self._adapt_phi_on_task(task, support_steps)
        print(
            f"  adapted φ: expl={adapted_params.exploration_rate:.3f}, "
            f"mut={adapted_params.mutation_strength:.3f}, "
            f"temp={adapted_params.llm_temperature:.3f}"
        )

        # unadapted: 直接用 φ_init（不做 support）
        unadapted_params = StrategyParams.from_vector(self.phi_current)

        random_phi = np.random.randn(StrategyParams.PHI_DIM).astype(np.float32) * 0.5
        random_params = StrategyParams.from_vector(random_phi)

        common = dict(
            task=task,
            n_steps=query_steps,
            llm=self.llm,
            semantic_analyzer=self.semantic_analyzer,
            max_pop_size=self.max_pop_size,
            analyze_interval=self.analyze_interval,
            global_sem=self.global_llm_sem,
        )

        r_adapted, r_unadapted, r_default, r_random = await asyncio.gather(
            run_inner_loop(params=adapted_params, **common),
            run_inner_loop(params=unadapted_params, **common),
            run_inner_loop(params=StrategyParams(), **common),
            run_inner_loop(params=random_params, **common),
        )

        task_results = {
            "meta_adapted": r_adapted.final_best_score,
            "meta_unadapted": r_unadapted.final_best_score,
            "default": r_default.final_best_score,
            "random": r_random.final_best_score,
            "baseline": task.baseline_score,
        }
        print(
            f"  {task.name}: adapted={r_adapted.final_best_score:.6f} "
            f"unadapted={r_unadapted.final_best_score:.6f} "
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


# ============================================================
# Skill-based Meta-Learner
# ============================================================


class SkillMetaLearner:
    """通过动态 Skill 生成 + 路由学习优化进化策略。

    Skill 不是预定义的，而是在运行过程中由 LLM 动态生成。
    Meta-learning 循环在多个任务上积累 skill library，学习：
    1. 哪些 axis 在什么任务类型上有效
    2. 哪些 skill 实例值得跨任务复用
    """

    def __init__(
        self,
        train_tasks: list[Task],
        test_tasks: list[Task],
        llm: LLMClient,
        config: dict,
        library_path: str | Path | None = None,
        n_meta_steps: int = 10,
        tasks_per_step: int = 3,
        n_segments: int = 3,
        steps_per_segment: int = 10,
    ):
        from .adapter import NativeAdapter
        from .skill_generator import SkillGenerator
        from .skill_library import SkillLibrary
        from .task_profile import get_task_profile

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.llm = llm
        self.config = config

        self.n_meta_steps = n_meta_steps
        self.tasks_per_step = tasks_per_step
        self.n_segments = n_segments
        self.steps_per_segment = steps_per_segment

        # Skill library（动态积累，可持久化）
        persist = Path(library_path) if library_path else None
        self.library = SkillLibrary(persist_path=persist)
        self.generator = SkillGenerator(llm)

        print(
            f"SkillMetaLearner initialized (library: {len(self.library.skills)} existing skills)"
        )

        # Task profiles
        self.task_profiles = {
            t.name: get_task_profile(t.name) for t in train_tasks + test_tasks
        }

        # Adapter
        params = StrategyParams.from_config(config)
        self.adapter = NativeAdapter(
            llm_model=config["llm"]["model"],
            params=params,
            llm=llm,
        )

    async def _orchestrated_run(self, task: Task):
        """用 SkillOrchestrator 跑单个任务。"""
        from .skill_orchestrator import SkillOrchestrator

        profile = self.task_profiles[task.name]
        orchestrator = SkillOrchestrator(
            library=self.library,
            generator=self.generator,
            task_profile=profile,
            adapter=self.adapter,
            config=self.config,
        )
        return await orchestrator.run(
            task=task,
            n_segments=self.n_segments,
            steps_per_segment=self.steps_per_segment,
        )

    async def meta_train(self):
        """Skill meta-learning 主循环。

        每步：采样任务 → orchestrated run（期间动态生成 skill）→ 蒸馏分析
        """
        from .skill_distiller import SkillDistiller

        distiller = SkillDistiller(self.llm)

        print(
            f"Skill Meta-Training: {self.n_meta_steps} 步, "
            f"{len(self.train_tasks)} 训练任务"
        )

        for step in range(self.n_meta_steps):
            t0 = time.time()

            # 1. 采样训练任务
            k = min(self.tasks_per_step, len(self.train_tasks))
            tasks = random.sample(self.train_tasks, k)
            print(f"\n[Skill Meta step {step}] tasks: {[t.name for t in tasks]}")

            # 2. 并行跑（每次 run 内部会动态生成 skill）
            results = await asyncio.gather(
                *[self._orchestrated_run(task) for task in tasks],
                return_exceptions=True,
            )

            # 收集轨迹
            trajectories = []
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"  [ERROR] {task.name}: {result}")
                    continue
                trajectories.append(result.trajectory)
                print(
                    f"  {task.name}: best={result.final_best_score:.6f} "
                    f"skills={result.trajectory.skills_used}"
                )

            if not trajectories:
                continue

            # 3. 跨任务蒸馏分析
            distill = await distiller.analyze(trajectories, self.library)
            print(f"  distill: {distill.observations[:120]}")
            if distill.recommendations:
                for rec in distill.recommendations[:3]:
                    print(f"    → {rec}")

            # 4. 保存 checkpoint
            self.library.prune()
            self.library.save()

            elapsed = time.time() - t0
            n_skills = len(self.library.skills)
            print(f"  [{elapsed:.1f}s] library: {n_skills} skills")

        print(f"\nSkill Meta-Training 完成, library: {len(self.library.skills)} skills")
        evidence = self.library.get_evidence_summary()
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
        return self.library

    async def evaluate_held_out(self) -> dict:
        """在 test tasks 上评估 skill-orchestrated 效果。"""
        results = {}
        for task in self.test_tasks:
            try:
                result = await self._orchestrated_run(task)
                results[task.name] = {
                    "skill_orchestrated": result.final_best_score,
                    "baseline": task.baseline_score,
                    "skills_used": result.trajectory.skills_used,
                    "improvement": result.trajectory.total_improvement,
                }
                print(
                    f"  {task.name}: score={result.final_best_score:.6f} "
                    f"Δ={result.trajectory.total_improvement:.6f}"
                )
            except Exception as e:
                results[task.name] = {"error": str(e)}
                print(f"  {task.name}: ERROR {e}")
        return results
