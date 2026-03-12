"""手动测试 combined_framework_fixes 候选的 test_set eval。"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from utils.data import load_pubmedqa
from utils.llm import resolve_model


def main():
    load_dotenv()
    config_path = "configs/pubmedqa_nested.json"
    with open(config_path) as f:
        cfg = json.load(f)

    data = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    spawn = cfg.get("spawn", {})

    model = resolve_model(opt.get("model", "together_ai/moonshotai/Kimi-K2.5"))
    project_root = Path(__file__).resolve().parent.parent
    source_dir = str(project_root / "target_systems" / "pubmedqa")

    # 使用 combined_framework_fixes 候选的 noa 代码
    candidate_noa_dir = str(
        project_root
        / ".noa_runs"
        / "noa_unified_l2"
        / "_sandbox_candidates"
        / "combined_framework_fixes"
    )
    if not os.path.exists(candidate_noa_dir):
        print(f"ERROR: candidate dir not found: {candidate_noa_dir}")
        sys.exit(1)
    print(f"Using candidate noa dir: {candidate_noa_dir}")

    # 加载数据 - 使用相同的 train/test split
    total_n = data.get("total_n", 500)
    test_n = data.get("test_n", 50)
    train_sample_size = data.get("train_sample_size", 10)
    split = data.get("split", "train")

    all_data = load_pubmedqa(split=split, n=total_n, data_dir=source_dir)

    rng = random.Random(42)
    test_set = rng.sample(all_data, min(test_n, len(all_data)))
    test_ids = {id(x) for x in test_set}
    train_pool = [x for x in all_data if id(x) not in test_ids]
    dataset = train_pool

    cache_dir = os.path.join(str(project_root), ".noa_cache")
    dpp = serialize_dataset(dataset, cache_dir=cache_dir)
    train_pool_pp = serialize_dataset(train_pool, cache_dir=cache_dir)
    test_set_pp = serialize_dataset(test_set, cache_dir=cache_dir)

    ml1 = spawn.get("mini_l1", {})

    print("Running mini-L1 with combined_framework_fixes candidate...")
    print(f"  model: {model}")
    print(f"  max_steps: {ml1.get('max_steps', 10)}")
    print(f"  train_sample_size: {train_sample_size}")

    run_seed = 42
    result = run_layer_subprocess(
        noa_dir=candidate_noa_dir,
        project_root=str(project_root),
        target_source_dir=source_dir,
        dataset_pickle_path=dpp,
        layer_level=1,
        max_steps=ml1.get("max_steps", opt.get("max_steps", 10)),
        n_samples=ml1.get("n_samples", opt.get("n_samples", 10)),
        max_llm_calls=ml1.get("max_llm_calls", opt.get("max_llm_calls", 80)),
        max_evals=ml1.get("max_evals", opt.get("max_evals", 8)),
        max_no_improve_steps=ml1.get(
            "max_no_improve_steps", opt.get("max_no_improve_steps", 4)
        ),
        model=model,
        isolate_source=True,
        random_seed=run_seed,
        train_pool_pickle_path=train_pool_pp,
        test_set_pickle_path=test_set_pp,
        train_sample_size=train_sample_size,
        top_k=opt.get("top_k", 3),
        timeout=ml1.get("timeout", 14400),
    )

    final_score = result.get("final_score", 0)
    print(f"\n{'=' * 60}")
    print("combined_framework_fixes test eval result:")
    print(f"  final_score: {final_score}")
    print(f"  baseline:    {result.get('baseline_score', '?')}")
    print(f"  status:      {result.get('status', '?')}")
    print(f"  steps:       {result.get('iterations', result.get('steps', '?'))}")
    print(f"  accepted:    {result.get('accepted', '?')}")
    print(f"  duration:    {result.get('duration_sec', '?')}s")
    print(f"{'=' * 60}")

    output = "results_combined_candidate_test.json"
    with open(output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
