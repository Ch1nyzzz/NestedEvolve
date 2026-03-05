"""NOA 统一入口 — 从 config JSON 读取全部参数，启动 Orchestrator。

L1-only: 设 nesting.max_spawn_calls = 0
嵌套优化: 设 nesting.max_spawn_calls >= 1
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from noa import Orchestrator
from noa.auto_adapter import auto_adapt
from scripts.archive_trajectories import archive_and_reset_trajectory_dir
from target_systems.hotpotqa_rag.evaluate import evaluate_batch, f1_score
from utils.data import load_hotpotqa
from utils.llm import resolve_model

load_dotenv()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/hotpotqa_nested.json"
    with open(config_path) as f:
        cfg = json.load(f)

    data = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    nest = cfg.get("nesting", {})
    spawn = cfg.get("spawn", {})
    history = cfg.get("history", {})
    output = cfg.get("output")

    model = resolve_model(opt.get("model", "gpt-4.1-mini"))
    project_root = Path(__file__).resolve().parent.parent
    source_dir = str(
        Path(__file__).resolve().parent.parent / "target_systems" / "hotpotqa_rag"
    )

    # 历史轨迹归档清理（默认执行一次）
    if history.get("archive_trajectories_on_start", True):
        cleanup_result = archive_and_reset_trajectory_dir(
            str(project_root),
            once=history.get("archive_once", True),
        )
        print(f"Trajectory cleanup: {cleanup_result}")

    # 数据
    n = data.get("n", 50)
    print(f"Loading HotpotQA data (n={n})...")
    dataset = load_hotpotqa(split=data.get("split", "validation"), n=n)

    # Adapter
    print("Setting up adapter...")
    adapter, target_factory = auto_adapt(source_dir=source_dir, model=model)

    # 启动 Orchestrator
    orch = Orchestrator(
        source_dir=source_dir,
        target_factory=target_factory,
        dataset=dataset,
        eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=4),
        score_fn=f1_score,
        l1_max_steps=opt.get("max_steps", 20),
        l1_n_samples=opt.get("n_samples", 50),
        l1_eval_n_samples=opt.get("eval_n_samples", 20),
        l1_model=model,
        l1_max_llm_calls=opt.get("max_llm_calls", 80),
        l1_max_evals=opt.get("max_evals", 12),
        l1_max_no_improve_steps=opt.get("max_no_improve_steps", 5),
        l1_max_tool_calls=opt.get("max_tool_calls", 10),
        l1_observer_tool_calls=opt.get("observer_tool_calls", 8),
        l1_optimizer_tool_calls=opt.get("optimizer_tool_calls", 5),
        max_depth=nest.get("max_depth", 3),
        max_spawn_calls=nest.get("max_spawn_calls", 2),
        spawn_config=spawn,
    )
    results = orch.run()

    # 输出
    print(f"\n{'='*60}")
    print(f"Baseline F1:   {results['baseline_score']:.2f}")
    print(f"Final F1:      {results['final_score']:.2f}")
    print(f"Improvement:   {results['final_score'] - results['baseline_score']:+.2f}")
    print(f"Total rounds:  {results['total_rounds']}")
    print(f"{'='*60}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
