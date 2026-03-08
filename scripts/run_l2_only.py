"""直接 spawn L2 meta-optimizer，跳过 L1 优化。

基于上次 L1 运行的结果（results_nested.json），构造 L2 所需的上下文，
直接运行 L2 优化 noa/ 框架代码。
"""

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from noa.core.protocol import LayerContext
from noa.engine import NOptimizer
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from noa.workspace import WorkspaceManager
from utils.data import load_hotpotqa
from utils.llm import resolve_model


def main():
    load_dotenv()
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/hotpotqa_nested.json"
    with open(config_path) as f:
        cfg = json.load(f)

    # 读取上次 L1 结果
    results_path = sys.argv[2] if len(sys.argv) > 2 else "results_nested.json"
    with open(results_path) as f:
        prev_results = json.load(f)

    data = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    spawn = cfg.get("spawn", {})
    output = "results_l2_only.json"

    model = resolve_model(opt.get("model", "gpt-4.1-mini"))
    project_root = Path(__file__).resolve().parent.parent
    source_dir = str(project_root / "target_systems" / "hotpotqa_rag")

    # 使用 WorkspaceManager 隔离，避免直接修改原始 noa/ 代码
    ws = WorkspaceManager(str(project_root), source_dir, run_id="l2_only")
    ws_noa, ws_source = ws.setup()
    noa_dir = ws_noa
    source_dir = ws_source
    print(f"[L2-only] Workspace ready: {ws.run_dir}")

    # 加载数据: train/test 完全隔离
    total_n = data.get("total_n", data.get("n", 300))
    test_n = data.get("test_n", 50)
    train_sample_size = data.get("train_sample_size", 25)
    print(f"Loading HotpotQA data (total={total_n})...")
    all_data = load_hotpotqa(split=data.get("split", "validation"), n=total_n)

    rng = random.Random(42)
    test_set = rng.sample(all_data, min(test_n, len(all_data)))
    test_ids = {id(x) for x in test_set}
    train_pool = [x for x in all_data if id(x) not in test_ids]
    dataset = train_pool  # dataset 传给 subprocess 的是 train_pool
    print(f"Test set: {len(test_set)} samples (fixed seed=42)")
    print(
        f"Train pool: {len(train_pool)} samples (sample {train_sample_size} per round)"
    )

    # 序列化 dataset + train/test
    cache_dir = os.path.join(str(project_root), ".noa_cache")
    dpp = serialize_dataset(dataset, cache_dir=cache_dir)
    train_pool_pp = serialize_dataset(train_pool, cache_dir=cache_dir)
    test_set_pp = serialize_dataset(test_set, cache_dir=cache_dir)

    # 从上次结果构造 parent context
    l1_result = prev_results["rounds"][0]["l1_result"]
    baseline_score = prev_results["baseline_score"]
    final_score = prev_results["final_score"]
    l1_history = l1_result.get("history", [])

    parent_history = [
        {
            "iteration": i + 1,
            "accepted": True,
            "label": h.get("label", ""),
            "step": h.get("step", 0),
        }
        for i, h in enumerate(l1_history)
        if h.get("action") == "accept_candidate"
    ]
    parent_summary = (
        f"Initial: {baseline_score:.2f}, Current: {final_score:.2f}, "
        f"Delta: {final_score - baseline_score:+.2f}, "
        f"Accepted: {len(parent_history)}"
    )

    print(f"L1 context: {parent_summary}")
    print(f"L1 accepted patches: {[h['label'] for h in parent_history]}")

    # --- 构造 L2 的 target/eval/score ---

    ml1 = spawn.get("mini_l1", {})

    def child_target_factory(noa_source_dir):
        def target(question):
            run_seed = hash(question) & 0x7FFFFFFF
            result = run_layer_subprocess(
                noa_dir=noa_source_dir,
                project_root=str(project_root),
                target_source_dir=source_dir,
                dataset_pickle_path=dpp,
                layer_level=1,
                max_steps=ml1.get("max_steps", opt.get("max_steps", 50)),
                n_samples=ml1.get("n_samples", opt.get("n_samples", 50)),
                max_llm_calls=ml1.get("max_llm_calls", opt.get("max_llm_calls", 150)),
                max_evals=ml1.get("max_evals", opt.get("max_evals", 20)),
                max_no_improve_steps=ml1.get(
                    "max_no_improve_steps", opt.get("max_no_improve_steps", 8)
                ),
                model=model,
                isolate_source=True,
                random_seed=run_seed,
                train_pool_pickle_path=train_pool_pp,
                test_set_pickle_path=test_set_pp,
                train_sample_size=train_sample_size,
                top_k=opt.get("top_k", 3),
            )
            return SimpleNamespace(
                answer=str(result.get("final_score", 0)),
                intermediate=result,
            )

        return target

    def child_score_fn(prediction: str, ground_truth: str) -> float:
        try:
            return float(str(prediction).strip()) / 100.0
        except (TypeError, ValueError):
            return 0.0

    def child_eval_fn(target, dataset):
        scores, details, errors = [], [], []
        for ex in dataset:
            result = target(ex.question)
            raw = float(result.answer) if result.answer else 0.0
            s = raw / 100.0
            scores.append(s)
            detail = {"question": ex.question, "f1": round(s, 4), "raw_score": raw}
            inter = getattr(result, "intermediate", {}) or {}
            if inter.get("error"):
                detail["error"] = inter["error"][:500]
                detail["error_type"] = inter.get("error_type", "unknown")
                errors.append(inter["error"][:500])
            if inter.get("history"):
                detail["accepted"] = inter.get("accepted", 0)
                detail["steps"] = inter.get("steps", 0)
            details.append(detail)
        avg = sum(scores) / len(scores) if scores else 0.0
        out = {"score": avg, "details": details}
        if errors:
            out["subprocess_errors"] = errors
        return out

    # 每次 eval 跑 1 个完整 L1
    child_dataset = [
        SimpleNamespace(question="opt_run_1", answer="0"),
    ]

    # --- L2 LayerContext ---

    child_layer_context = LayerContext(
        layer_id="L2",
        level=2,
        writable_root=noa_dir,
        readable_roots=[noa_dir, source_dir],
        parent_history=parent_history,
        parent_summary=parent_summary,
        max_depth=2,
        max_spawn_calls=0,  # L2 不再 spawn
    )

    # --- 启动 L2 ---

    l2_cfg = spawn.get("l2", {})
    print(f"\n{'='*60}")
    print("Starting L2 meta-optimizer directly")
    print(f"  noa_dir: {noa_dir}")
    print(f"  model: {model}")
    print(f"  max_steps: {l2_cfg.get('max_steps', 12)}")
    print(f"  max_evals: {l2_cfg.get('max_evals', 8)}")
    print(f"  mini_l1 config: {ml1}")
    print(f"{'='*60}\n")

    l2 = NOptimizer(
        source_dir=noa_dir,
        target_factory=child_target_factory,
        dataset=child_dataset,
        eval_fn=child_eval_fn,
        max_steps=l2_cfg.get("max_steps", 12),
        n_samples=l2_cfg.get("n_samples", 2),
        model=model,
        score_fn=child_score_fn,
        max_llm_calls=l2_cfg.get("max_llm_calls", 80),
        max_evals=l2_cfg.get("max_evals", 10),
        max_no_improve_steps=l2_cfg.get("max_no_improve_steps", 5),
        layer_context=child_layer_context,
        observer_search_roots=[noa_dir],
        dataset_pickle_path=dpp,
        spawn_config=spawn,
    )

    result = l2.run()

    # 输出
    accepted = result.get("accepted", 0) or 0
    print(f"\n{'='*60}")
    print("L2 Result:")
    print(f"  Baseline score:  {result.get('baseline_score', 0):.2f}")
    print(f"  Final score:     {result.get('final_score', 0):.2f}")
    print(f"  Steps:           {result.get('steps', 0)}")
    print(f"  Accepted:        {accepted}")
    print(f"  NOA modified:    {accepted > 0}")
    print(f"{'='*60}")

    if output:
        out = {
            "l2_result": result,
            "l1_context": {
                "baseline_score": baseline_score,
                "final_score": final_score,
                "accepted_patches": parent_history,
                "parent_summary": parent_summary,
            },
        }
        with open(output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
