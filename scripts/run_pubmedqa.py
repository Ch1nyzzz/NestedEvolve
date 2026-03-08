"""NOA 统一入口 — PubMedQA target system。

L1-only: 设 nesting.max_spawn_calls = 0
嵌套优化: 设 nesting.max_spawn_calls >= 1
"""

import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from noa import Orchestrator
from noa.auto_adapter import auto_adapt
from scripts.archive_trajectories import archive_and_reset_trajectory_dir
from target_systems.pubmedqa.evaluate import evaluate_batch, exact_match
from utils.data import load_pubmedqa


def main():
    load_dotenv()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pubmedqa_nested.json"
    with open(config_path) as f:
        cfg = json.load(f)

    data = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    nest = cfg.get("nesting", {})
    spawn = cfg.get("spawn", {})
    history = cfg.get("history", {})
    output = cfg.get("output")

    model = opt.get("model", "together_ai/moonshotai/Kimi-K2.5")
    project_root = Path(__file__).resolve().parent.parent
    source_dir = str(project_root / "target_systems" / "pubmedqa")

    # 历史轨迹归档清理
    if history.get("archive_trajectories_on_start", True):
        cleanup_result = archive_and_reset_trajectory_dir(
            str(project_root),
            once=history.get("archive_once", True),
        )
        print(f"Trajectory cleanup: {cleanup_result}")

    # 数据: train/test 完全隔离
    total_n = data.get("total_n", 500)
    test_n = data.get("test_n", 50)
    train_sample_size = data.get("train_sample_size", 25)
    print(f"Loading PubMedQA data (total={total_n})...")
    all_data = load_pubmedqa(split=data.get("split", "train"), n=total_n)

    rng = random.Random(42)
    test_set = rng.sample(all_data, min(test_n, len(all_data)))
    test_ids = {id(x) for x in test_set}
    train_pool = [x for x in all_data if id(x) not in test_ids]
    print(f"Test set: {len(test_set)} samples (fixed seed=42)")
    print(
        f"Train pool: {len(train_pool)} samples (sample {train_sample_size} per round)"
    )

    # Adapter
    print("Setting up adapter...")
    adapter, target_factory = auto_adapt(source_dir=source_dir, model=model)

    # 启动 Orchestrator
    orch = Orchestrator(
        source_dir=source_dir,
        target_factory=target_factory,
        dataset=train_pool,
        eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=15),
        score_fn=exact_match,
        l1_max_steps=opt.get("max_steps", 30),
        l1_n_samples=opt.get("n_samples", 30),
        l1_model=model,
        l1_max_llm_calls=opt.get("max_llm_calls", 150),
        l1_max_evals=opt.get("max_evals", 20),
        l1_max_no_improve_steps=opt.get("max_no_improve_steps", 8),
        max_depth=nest.get("max_depth", 2),
        max_spawn_calls=nest.get("max_spawn_calls", 1),
        spawn_config=spawn,
        train_pool=train_pool,
        test_set=test_set,
        train_sample_size=train_sample_size,
    )
    results = orch.run()

    # 输出
    print(f"\n{'='*60}")
    print(f"Baseline Acc:  {results['baseline_score']:.2f}")
    print(f"Final Acc:     {results['final_score']:.2f}")
    print(f"Improvement:   {results['final_score'] - results['baseline_score']:+.2f}")
    print(f"Total rounds:  {results['total_rounds']}")
    print(f"{'='*60}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
