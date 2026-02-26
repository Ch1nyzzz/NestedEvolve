"""L1+L2 嵌套优化入口 — 用 Orchestrator 自动判断是否升级到 L2。"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import load_hotpotqa
from target_systems.hotpotqa_rag.evaluate import evaluate_batch, f1_score
from noa import Orchestrator
from noa.auto_adapter import auto_adapt


def main():
    parser = argparse.ArgumentParser(description="NOA L1+L2 Nested Optimizer")
    parser.add_argument("--n", type=int, default=50, help="Dataset size")
    parser.add_argument("--l1-iterations", type=int, default=1, help="L1 max iterations")
    parser.add_argument("--l1-samples", type=int, default=20, help="L1 samples per iteration")
    parser.add_argument("--l2-iterations", type=int, default=1, help="L2 max iterations")
    parser.add_argument("--l2-n-samples", type=int, default=None, help="L2 evaluate runs (default=1)")
    parser.add_argument("--l2-rounds", type=int, default=1, help="Max L2 rounds")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM model")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    args = parser.parse_args()

    # 目标系统源码目录
    source_dir = str(Path(__file__).resolve().parent.parent / "target_systems" / "hotpotqa_rag")

    # 加载数据
    print(f"Loading HotpotQA data (n={args.n})...")
    dataset = load_hotpotqa(split=args.split, n=args.n)

    # 生成 adapter + target_factory
    print("Setting up adapter...")
    adapter, target_factory = auto_adapt(source_dir=source_dir, model=args.model)

    # 运行嵌套优化
    orch = Orchestrator(
        source_dir=source_dir,
        target_factory=target_factory,
        dataset=dataset,
        eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=4),
        score_fn=f1_score,
        l1_max_iterations=args.l1_iterations,
        l1_n_samples=args.l1_samples,
        l1_model=args.model,
        l2_max_iterations=args.l2_iterations,
        l2_n_samples=args.l2_n_samples,
        l2_model=args.model,
        max_l2_rounds=args.l2_rounds,
    )
    results = orch.run()

    # 输出结果
    print(f"\n{'='*60}")
    print(f"Baseline F1:   {results['baseline_score']:.2f}")
    print(f"Final F1:      {results['final_score']:.2f}")
    print(f"Improvement:   {results['final_score'] - results['baseline_score']:+.2f}")
    print(f"Total rounds:  {results['total_rounds']}")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
