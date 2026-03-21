"""外层 meta-learning：基于 Skill Library 的跨任务路由学习。"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

from ..evolution.strategy import StrategyParams
from ..integrations.llm import LLMClient
from ..tasks.loader import Task


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
        task_artifact_path: str | Path | None = None,
        n_meta_steps: int = 10,
        tasks_per_step: int = 3,
        n_iterations: int = 30,
    ):
        from ..integrations.targets import NativeAdapter
        from ..tasks.profiles import get_task_profile
        from .generator import SkillGenerator
        from .library import SkillLibrary

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.llm = llm
        self.config = config

        self.n_meta_steps = n_meta_steps
        self.tasks_per_step = tasks_per_step
        self.n_iterations = n_iterations

        # Skill library（动态积累，可持久化）
        persist = Path(library_path) if library_path else None
        task_artifacts = Path(task_artifact_path) if task_artifact_path else None
        self.library = SkillLibrary(
            persist_path=persist,
            task_artifact_path=task_artifacts,
        )
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
        from .orchestrator import SkillOrchestrator

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
            n_iterations=self.n_iterations,
        )

    async def meta_train(self):
        """Skill meta-learning 主循环。

        每步：采样任务 → orchestrated run（期间动态生成 skill）→ 蒸馏分析
        """
        from .distiller import SkillDistiller

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
