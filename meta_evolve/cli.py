"""CLI 入口：运行内层进化循环或 skill-based meta-learning。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from .config import (
    artifact_path,
    load_config,
    shared_skill_artifact_path,
    shared_skill_library_path,
)


async def run_single_task(args, config):
    """单任务进化循环（Step 1 验证）。"""
    from .evolution.loop import run_inner_loop
    from .evolution.strategy import StrategyParams
    from .integrations.llm import LLMClient
    from .tasks.loader import load_task

    task_name = args.task
    bench_root = getattr(args, "benchmarks_root", None)

    print(f"加载任务: {task_name} ...")
    task = load_task(task_name, benchmarks_root=bench_root)
    print(f"  baseline_score = {task.baseline_score:.6f}")

    llm = LLMClient.from_config(config)

    if args.fixed_phi:
        params = StrategyParams()
    else:
        params = StrategyParams.from_config(config)

    print(f"开始进化 ({args.iterations} iterations × {args.batch_size} batch)，策略参数: {params}")
    t0 = time.time()

    result = await run_inner_loop(
        task=task,
        params=params,
        n_iterations=args.iterations,
        batch_size=args.batch_size,
        llm=llm,
        max_pop_size=config["inner_loop"]["max_pop_size"],
    )

    elapsed = time.time() - t0
    print(f"\n完成! 耗时 {elapsed:.1f}s")
    print(f"  初始 score: {task.baseline_score:.6f}")
    print(f"  最终 score: {result.final_best_score:.6f}")
    print(f"  提升: {result.final_best_score - task.baseline_score:.6f}")
    print(f"  种群: {result.final_population.size()} 个体")
    print(f"  轨迹: {result.score_trajectory}")

    out = {
        "task": task_name,
        "baseline": task.baseline_score,
        "final_best": result.final_best_score,
        "trajectory": result.score_trajectory,
        "elapsed_sec": elapsed,
    }
    out_path = artifact_path(f"results_metaevolve_{task_name.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存到 {out_path}")


async def run_skill_evolve(args, config):
    """用 SkillOrchestrator 跑单任务（Skill 动态生成 + 进化）。"""
    from .evolution.strategy import StrategyParams
    from .integrations.llm import LLMClient
    from .integrations.targets import NativeAdapter
    from .skills.generator import SkillGenerator
    from .skills.library import SkillLibrary
    from .skills.orchestrator import SkillOrchestrator
    from .tasks.loader import load_task
    from .tasks.profiles import get_task_profile

    task_name = args.task
    bench_root = getattr(args, "benchmarks_root", None)

    print(f"加载任务: {task_name} ...")
    task = load_task(task_name, benchmarks_root=bench_root)
    print(f"  baseline_score = {task.baseline_score:.6f}")

    llm = LLMClient.from_config(config)

    params = StrategyParams.from_config(config)

    library_path = shared_skill_library_path()
    task_artifact_path = shared_skill_artifact_path()
    library = SkillLibrary(
        persist_path=library_path,
        task_artifact_path=task_artifact_path,
    )
    generator = SkillGenerator(llm)
    print(f"  Skill library: {len(library.skills)} existing skills")
    print(f"  Task analysis artifacts: {len(library.task_artifacts)} tasks")

    profile = get_task_profile(task_name)
    adapter = NativeAdapter(llm_model=config["llm"]["model"], params=params, llm=llm)
    orchestrator = SkillOrchestrator(
        library=library,
        generator=generator,
        task_profile=profile,
        adapter=adapter,
        config=config,
    )

    t0 = time.time()
    result = await orchestrator.run(
        task=task,
        n_iterations=args.iterations,
    )
    elapsed = time.time() - t0

    print(f"\n完成! 耗时 {elapsed:.1f}s")
    print(f"  初始 score: {task.baseline_score:.6f}")
    print(f"  最终 score: {result.final_best_score:.6f}")
    print(f"  提升: {result.trajectory.total_improvement:.6f}")
    print(f"  生成 skills: {len(library.skills)}")
    print(f"  使用 skills: {result.trajectory.skills_used}")

    out = {
        "task": task_name,
        "baseline": task.baseline_score,
        "final_best": result.final_best_score,
        "improvement": result.trajectory.total_improvement,
        "skills_used": result.trajectory.skills_used,
        "n_skills_generated": len(library.skills),
        "elapsed_sec": elapsed,
        "trajectory": result.trajectory.to_summary(),
        "evidence": result.evidence_snapshot,
    }
    out_path = artifact_path(f"results_skill_evolve_{task_name.replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {out_path}")

    library.save()
    print(f"  Skill library 已保存到 {library_path}")
    print(f"  Task analysis artifacts 已保存到 {task_artifact_path}")


async def run_skill_meta(args, config):
    """Skill-based Meta-learning：跨任务动态 skill 积累。"""
    from .integrations.llm import LLMClient
    from .skills.meta import SkillMetaLearner
    from .tasks.loader import discover_tasks, load_task_parallel

    llm = LLMClient.from_config(config)
    bench_root = getattr(args, "benchmarks_root", None)

    # 支持 config 指定 train/test，也支持自动发现
    tasks_cfg = config.get("tasks", {})
    if tasks_cfg.get("train") and tasks_cfg.get("test"):
        all_names = tasks_cfg["train"] + tasks_cfg["test"]
        train_names = set(tasks_cfg["train"])
    else:
        all_names = discover_tasks(bench_root)
        n_train = max(1, int(len(all_names) * 0.7))
        train_names = set(all_names[:n_train])
        print(f"自动发现 {len(all_names)} 个任务，{len(train_names)} train / {len(all_names) - len(train_names)} test")

    print(f"并行加载 {len(all_names)} 个任务...")
    all_tasks = load_task_parallel(all_names, benchmarks_root=bench_root)
    train_tasks = [t for t in all_tasks if t.name in train_names]
    test_tasks = [t for t in all_tasks if t.name not in train_names]

    for t in all_tasks:
        tag = "train" if t.name in train_names else "test"
        print(f"  [{tag}] {t.name}: baseline={t.baseline_score:.6f}")

    skill_cfg = config.get("skill_meta", {})
    library_path = shared_skill_library_path()
    task_artifact_path = shared_skill_artifact_path()

    learner = SkillMetaLearner(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        llm=llm,
        config=config,
        library_path=library_path,
        task_artifact_path=task_artifact_path,
        n_meta_steps=skill_cfg.get("n_meta_steps", 10),
        tasks_per_step=skill_cfg.get("tasks_per_step", 3),
        n_iterations=skill_cfg.get("n_iterations", 30),
    )

    print("\n开始 Skill Meta-Learning（动态 skill 生成）...")
    library = await learner.meta_train()

    print("\n评估 held-out 任务...")
    results = await learner.evaluate_held_out()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    library.save()
    print(f"\nSkill library ({len(library.skills)} skills) 已保存到 {library_path}")
    print(f"Task analysis artifacts 已保存到 {task_artifact_path}")


def main():
    parser = argparse.ArgumentParser(description="MetaEvolve")
    sub = parser.add_subparsers(dest="command")

    # 公共参数
    def _add_common(p):
        p.add_argument("--config", default=None)
        p.add_argument("--benchmarks-root", default=None, help="benchmark 根目录（默认 meta_evolve/benchmarks）")

    # 列出可用任务
    p_list = sub.add_parser("list", help="列出所有可用 benchmark")
    _add_common(p_list)

    # 单任务进化
    p_single = sub.add_parser("evolve", help="单任务进化循环")
    p_single.add_argument("--task", required=True, help="benchmark 名称（如 math/circle_packing）")
    p_single.add_argument("--iterations", type=int, default=20, help="迭代数")
    p_single.add_argument("--batch-size", type=int, default=8, help="每迭代并行候选数")
    p_single.add_argument("--fixed-phi", action="store_true", help="使用默认策略参数")
    _add_common(p_single)

    # Skill-based 单任务进化
    p_skill = sub.add_parser("skill-evolve", help="Skill-orchestrated 单任务进化")
    p_skill.add_argument("--task", required=True, help="benchmark 名称（如 ADRS/cloudcast）")
    p_skill.add_argument("--iterations", type=int, default=50, help="迭代数")
    _add_common(p_skill)

    # Skill meta-learning
    p_skill_meta = sub.add_parser("skill-meta", help="Skill-based Meta-learning 训练")
    _add_common(p_skill_meta)

    # Refresh: 重置 skill library 到初始状态
    sub.add_parser("refresh", help="重置 skill library，只保留 starter skill")

    args = parser.parse_args()

    if args.command == "refresh":
        from .skills.filesystem import refresh_skills
        removed = refresh_skills()
        print(f"已清理 {removed} 项，skill library 已重置到初始状态")
        return

    if args.command == "list":
        from .tasks.loader import discover_tasks
        bench_root = getattr(args, "benchmarks_root", None)
        tasks = discover_tasks(bench_root)
        print(f"共 {len(tasks)} 个可用 benchmark:")
        for t in tasks:
            print(f"  {t}")
        return

    config = load_config(getattr(args, "config", None))

    if args.command == "evolve":
        asyncio.run(run_single_task(args, config))
    elif args.command == "skill-evolve":
        asyncio.run(run_skill_evolve(args, config))
    elif args.command == "skill-meta":
        asyncio.run(run_skill_meta(args, config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
