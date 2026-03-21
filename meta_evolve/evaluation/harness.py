"""评估框架：对比默认 evolver 与 skill-orchestrated evolver。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..evolution.loop import run_inner_loop
from ..evolution.strategy import StrategyParams
from ..integrations.llm import LLMClient
from ..tasks.loader import Task


@dataclass
class EvalResult:
    task_name: str
    baseline_score: float
    results: dict[str, float] = field(default_factory=dict)
    trajectories: dict[str, list[float]] = field(default_factory=dict)


async def eval_default(
    task: Task, llm: LLMClient, n_iterations: int, batch_size: int = 8,
) -> tuple[float, list[float]]:
    """Baseline A: default evolver（无 skill）。"""
    result = await run_inner_loop(
        task=task,
        params=StrategyParams(),
        n_iterations=n_iterations,
        batch_size=batch_size,
        llm=llm,
    )
    return result.final_best_score, result.score_trajectory


async def eval_skill_orchestrated(
    task: Task,
    llm: LLMClient,
    config: dict,
    n_iterations: int,
) -> tuple[float, list[float]]:
    """Treatment: skill-orchestrated evolver（动态 skill 生成）。"""
    from ..integrations.targets import NativeAdapter
    from ..tasks.profiles import get_task_profile
    from ..skills.generator import SkillGenerator
    from ..skills.library import SkillLibrary
    from ..skills.orchestrator import SkillOrchestrator

    strategy_cfg = config.get("strategy", {})
    params = StrategyParams(**{k: v for k, v in strategy_cfg.items() if k != "PHI_DIM"})

    library = SkillLibrary()  # 每次 eval 从空 library 开始
    generator = SkillGenerator(llm)
    profile = get_task_profile(task.name)
    adapter = NativeAdapter(llm_model=config["llm"]["model"], params=params, llm=llm)

    orchestrator = SkillOrchestrator(
        library=library,
        generator=generator,
        task_profile=profile,
        adapter=adapter,
        config=config,
    )
    result = await orchestrator.run(
        task=task,
        n_iterations=n_iterations,
    )
    trajectory = [seg.best_score for seg in result.trajectory.segments]
    return result.final_best_score, trajectory


async def run_comparison(
    tasks: list[Task],
    llm: LLMClient,
    config: dict,
    n_iterations: int = 15,
    batch_size: int = 8,
    n_skill_iterations: int = 30,
    n_repeats: int = 3,
) -> list[EvalResult]:
    """对比实验：在给定任务上跑多种方法。"""
    results = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"评估: {task.name} (baseline={task.baseline_score:.6f})")
        print(f"{'='*60}")

        eval_result = EvalResult(
            task_name=task.name,
            baseline_score=task.baseline_score,
        )

        # 多次重复取平均
        default_scores = []
        skill_scores = []

        for rep in range(n_repeats):
            print(f"\n  --- Repeat {rep+1}/{n_repeats} ---")

            # Baseline A: default
            print("  Running default evolver...")
            t0 = time.time()
            score_a, traj_a = await eval_default(task, llm, n_iterations, batch_size)
            print(f"    default: {score_a:.6f} ({time.time()-t0:.1f}s)")
            default_scores.append(score_a)

            # Treatment: skill-orchestrated
            print("  Running skill-orchestrated evolver...")
            t0 = time.time()
            score_c, traj_c = await eval_skill_orchestrated(
                task, llm, config, n_skill_iterations,
            )
            print(f"    skill: {score_c:.6f} ({time.time()-t0:.1f}s)")
            skill_scores.append(score_c)

        eval_result.results = {
            "default_mean": float(np.mean(default_scores)),
            "default_std": float(np.std(default_scores)),
            "skill_mean": float(np.mean(skill_scores)),
            "skill_std": float(np.std(skill_scores)),
        }

        print(f"\n  {task.name} 汇总:")
        print(
            f"    default:  {eval_result.results['default_mean']:.6f} ± {eval_result.results['default_std']:.6f}"
        )
        print(
            f"    skill:    {eval_result.results['skill_mean']:.6f} ± {eval_result.results['skill_std']:.6f}"
        )
        results.append(eval_result)

    return results


def print_summary(results: list[EvalResult]):
    """打印对比汇总表。"""
    print(f"\n{'='*80}")
    print(
        f"{'Task':<25} {'Baseline':<10} {'Default':<15} {'Skill':<15} {'Δ(Skill-Default)':<15}"
    )
    print(f"{'='*80}")
    for r in results:
        default_mean = r.results.get("default_mean", 0)
        skill_mean = r.results.get("skill_mean", 0)
        delta = skill_mean - default_mean
        print(
            f"{r.task_name:<25} "
            f"{r.baseline_score:<10.4f} "
            f"{default_mean:<15.4f} "
            f"{skill_mean:<15.4f} "
            f"{delta:<+15.4f}"
        )
