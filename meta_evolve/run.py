"""CLI 入口：运行内层进化循环或完整 meta-learning。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = str(Path(__file__).parent / "configs" / "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


async def run_single_task(args, config):
    """单任务进化循环（Step 1 验证）。"""
    from .evolve_loop import run_inner_loop
    from .llm_client import LLMClient
    from .strategy import StrategyParams
    from .task_adapter import PHASE1_TASKS, load_task

    task_name = args.task
    task_id = PHASE1_TASKS.index(task_name) if task_name in PHASE1_TASKS else 0

    print(f"加载任务: {task_name} ...")
    task = load_task(task_name, task_id)
    print(f"  baseline_score = {task.baseline_score:.6f}")

    llm = LLMClient(
        model=config["llm"]["model"],
        max_tokens=config["llm"]["max_tokens"],
    )

    if args.fixed_phi:
        params = StrategyParams()  # 默认值
    else:
        strategy_cfg = config.get("strategy", {})
        params = StrategyParams(
            **{k: v for k, v in strategy_cfg.items() if k != "PHI_DIM"}
        )

    # Step 2: 可选启用语义分析
    semantic_analyzer = None
    if args.semantic:
        from .semantic_analyzer import SemanticAnalyzer

        semantic_analyzer = SemanticAnalyzer(llm)

    print(f"开始进化 ({args.steps} 步)，策略参数: {params}")
    t0 = time.time()

    result = await run_inner_loop(
        task=task,
        params=params,
        n_steps=args.steps,
        llm=llm,
        semantic_analyzer=semantic_analyzer,
        max_pop_size=config["inner_loop"]["max_pop_size"],
        analyze_interval=config["inner_loop"]["analyze_interval"],
    )

    elapsed = time.time() - t0
    print(f"\n完成! 耗时 {elapsed:.1f}s")
    print(f"  初始 score: {task.baseline_score:.6f}")
    print(f"  最终 score: {result.final_best_score:.6f}")
    print(f"  提升: {result.final_best_score - task.baseline_score:.6f}")
    print(f"  种群: {result.final_population.size()} 个体")
    print(f"  轨迹: {result.score_trajectory}")

    # 保存结果
    out = {
        "task": task_name,
        "baseline": task.baseline_score,
        "final_best": result.final_best_score,
        "trajectory": result.score_trajectory,
        "elapsed_sec": elapsed,
        "n_transitions": len(result.transitions),
    }
    out_path = f"results_metaevolve_{task_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存到 {out_path}")


async def run_skill_evolve(args, config):
    """用 SkillOrchestrator 跑单任务（Skill 动态生成 + 进化）。"""
    from .adapter import NativeAdapter
    from .llm_client import LLMClient
    from .skill_generator import SkillGenerator
    from .skill_library import SkillLibrary
    from .skill_orchestrator import SkillOrchestrator
    from .strategy import StrategyParams
    from .task_adapter import PHASE1_TASKS, load_task
    from .task_profile import get_task_profile

    task_name = args.task
    task_id = PHASE1_TASKS.index(task_name) if task_name in PHASE1_TASKS else 0

    print(f"加载任务: {task_name} ...")
    task = load_task(task_name, task_id)
    print(f"  baseline_score = {task.baseline_score:.6f}")

    llm = LLMClient(
        model=config["llm"]["model"],
        max_tokens=config["llm"]["max_tokens"],
    )

    strategy_cfg = config.get("strategy", {})
    params = StrategyParams(**{k: v for k, v in strategy_cfg.items() if k != "PHI_DIM"})

    # 动态 skill library（从空开始，运行中生成）
    library_path = Path(__file__).parent / f"skill_library_{task_name}.json"
    library = SkillLibrary(persist_path=library_path)
    generator = SkillGenerator(llm)
    print(f"  Skill library: {len(library.skills)} existing skills")

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
        n_segments=args.segments,
        steps_per_segment=args.steps_per_seg,
    )
    elapsed = time.time() - t0

    print(f"\n完成! 耗时 {elapsed:.1f}s")
    print(f"  初始 score: {task.baseline_score:.6f}")
    print(f"  最终 score: {result.final_best_score:.6f}")
    print(f"  提升: {result.trajectory.total_improvement:.6f}")
    print(f"  生成 skills: {len(library.skills)}")
    print(f"  使用 skills: {result.trajectory.skills_used}")

    # 保存结果
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
    out_path = f"results_skill_evolve_{task_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到 {out_path}")

    # 持久化 skill library
    library.save()
    print(f"  Skill library 已保存到 {library_path}")


async def run_skill_meta(args, config):
    """Skill-based Meta-learning：跨任务动态 skill 积累。"""
    from .llm_client import LLMClient
    from .meta_learner import SkillMetaLearner
    from .task_adapter import load_task_parallel

    llm = LLMClient(
        model=config["llm"]["model"],
        max_tokens=config["llm"]["max_tokens"],
    )

    all_names = config["tasks"]["train"] + config["tasks"]["test"]
    train_names = set(config["tasks"]["train"])
    print(f"并行加载 {len(all_names)} 个任务...")
    all_tasks = load_task_parallel(all_names)
    train_tasks = [t for t in all_tasks if t.name in train_names]
    test_tasks = [t for t in all_tasks if t.name not in train_names]

    for t in all_tasks:
        tag = "train" if t.name in train_names else "test"
        print(f"  [{tag}] {t.name}: baseline={t.baseline_score:.6f}")

    skill_cfg = config.get("skill_meta", {})
    library_path = Path(__file__).parent / "skill_library_meta.json"

    learner = SkillMetaLearner(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        llm=llm,
        config=config,
        library_path=library_path,
        n_meta_steps=skill_cfg.get("n_meta_steps", 10),
        tasks_per_step=skill_cfg.get("tasks_per_step", 3),
        n_segments=skill_cfg.get("n_segments", 3),
        steps_per_segment=skill_cfg.get("steps_per_segment", 10),
    )

    print("\n开始 Skill Meta-Learning（动态 skill 生成）...")
    library = await learner.meta_train()

    print("\n评估 held-out 任务...")
    results = await learner.evaluate_held_out()
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 保存最终 library
    library.save()
    print(f"\nSkill library ({len(library.skills)} skills) 已保存到 {library_path}")


async def run_meta_learn(args, config):
    """Meta-learning 完整流程（Step 3）。"""
    from .llm_client import LLMClient
    from .meta_learner import MetaLearner
    from .semantic_analyzer import SemanticAnalyzer
    from .surrogate import SurrogateModel
    from .task_adapter import load_task_parallel

    llm = LLMClient(
        model=config["llm"]["model"],
        max_tokens=config["llm"]["max_tokens"],
    )

    all_names = config["tasks"]["train"] + config["tasks"]["test"]
    train_names = set(config["tasks"]["train"])
    print(f"并行加载 {len(all_names)} 个任务...")
    all_tasks = load_task_parallel(all_names)
    train_tasks = [t for t in all_tasks if t.name in train_names]
    test_tasks = [t for t in all_tasks if t.name not in train_names]
    for t in all_tasks:
        tag = "train" if t.name in train_names else "test"
        print(f"  [{tag}] {t.name}: baseline={t.baseline_score:.6f}")

    surrogate_cfg = config["surrogate"]
    surrogate = SurrogateModel(
        phi_dim=surrogate_cfg["phi_dim"],
        numeric_state_dim=surrogate_cfg["numeric_state_dim"],
        semantic_state_dim=surrogate_cfg["semantic_state_dim"],
        n_tasks=surrogate_cfg["n_tasks"],
        task_embed_dim=surrogate_cfg["task_embed_dim"],
        hidden=surrogate_cfg["hidden"],
        buffer_size=surrogate_cfg["buffer_size"],
    )

    semantic_analyzer = SemanticAnalyzer(llm) if args.semantic else None

    meta_cfg = config["meta"]
    learner = MetaLearner(
        surrogate=surrogate,
        semantic_analyzer=semantic_analyzer,
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        llm=llm,
        config=config,
        n_meta_steps=meta_cfg["n_meta_steps"],
        tasks_per_step=meta_cfg["tasks_per_step"],
        adaptation_steps=meta_cfg["adaptation_steps"],
        adaptation_lr=meta_cfg["adaptation_lr"],
        meta_lr=meta_cfg["meta_lr"],
        validation_steps=meta_cfg["validation_steps"],
        warmup_steps=meta_cfg["warmup_steps"],
        gradient_clip=meta_cfg["gradient_clip"],
        max_concurrency=meta_cfg.get("max_concurrency", 50),
    )

    print(f"\n开始 meta-learning ({meta_cfg['n_meta_steps']} 步)...")
    await learner.meta_train()

    print("\n评估 held-out 任务...")
    results = await learner.evaluate_held_out()
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="MetaEvolve")
    sub = parser.add_subparsers(dest="command")

    # 单任务进化
    p_single = sub.add_parser("evolve", help="单任务进化循环")
    p_single.add_argument("--task", required=True, help="benchmark 名称")
    p_single.add_argument("--steps", type=int, default=15)
    p_single.add_argument("--fixed-phi", action="store_true", help="使用默认策略参数")
    p_single.add_argument("--semantic", action="store_true", help="启用语义分析")
    p_single.add_argument("--config", default=None)

    # Meta-learning
    p_meta = sub.add_parser("meta", help="Meta-learning 训练 (ES φ优化)")
    p_meta.add_argument("--semantic", action="store_true", help="启用语义分析")
    p_meta.add_argument("--config", default=None)

    # Skill-based 单任务进化
    p_skill = sub.add_parser("skill-evolve", help="Skill-orchestrated 单任务进化")
    p_skill.add_argument("--task", required=True, help="benchmark 名称")
    p_skill.add_argument("--segments", type=int, default=3, help="segment 数")
    p_skill.add_argument(
        "--steps-per-seg", type=int, default=10, help="每 segment 步数"
    )
    p_skill.add_argument("--config", default=None)

    # Skill meta-learning
    p_skill_meta = sub.add_parser("skill-meta", help="Skill-based Meta-learning 训练")
    p_skill_meta.add_argument("--config", default=None)

    args = parser.parse_args()
    config = load_config(getattr(args, "config", None))

    if args.command == "evolve":
        asyncio.run(run_single_task(args, config))
    elif args.command == "meta":
        asyncio.run(run_meta_learn(args, config))
    elif args.command == "skill-evolve":
        asyncio.run(run_skill_evolve(args, config))
    elif args.command == "skill-meta":
        asyncio.run(run_skill_meta(args, config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
