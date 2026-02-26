"""L1 优化入口 — 用 NOptimizer 优化 HotpotQA RAG Pipeline。"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import load_hotpotqa
from target_systems.hotpotqa_rag.evaluate import evaluate_batch, f1_score
from noa import NOptimizer
from noa.auto_adapter import auto_adapt


def main():
    parser = argparse.ArgumentParser(description="NOA L1 Optimizer for HotpotQA RAG")
    parser.add_argument("--n", type=int, default=50, help="Dataset size")
    parser.add_argument("--samples", type=int, default=20, help="Samples per iteration")
    parser.add_argument("--iterations", type=int, default=1, help="Max optimization iterations")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="LLM model")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--force-adapter", action="store_true", help="Force regenerate adapter")
    args = parser.parse_args()

    # 目标系统源码目录
    source_dir = str(Path(__file__).resolve().parent.parent / "target_systems" / "hotpotqa_rag")

    # 加载数据
    print(f"Loading HotpotQA data (n={args.n})...")
    dataset = load_hotpotqa(split=args.split, n=args.n)

    # 生成 adapter + target_factory
    print("Setting up adapter...")
    adapter, target_factory = auto_adapt(
        source_dir=source_dir,
        model=args.model,
        force=args.force_adapter,
    )

    # 运行 L1 优化
    optimizer = NOptimizer(
        source_dir=source_dir,
        target_factory=target_factory,
        dataset=dataset,
        eval_fn=lambda t, d: evaluate_batch(t, d, max_workers=4),
        max_iterations=args.iterations,
        n_samples=args.samples,
        model=args.model,
        score_fn=f1_score,
    )
    results = optimizer.run()

    # 输出结果
    print(f"\n{'='*50}")
    print(f"Baseline F1: {results['baseline_score']:.2f}")
    print(f"Final F1:    {results['final_score']:.2f}")
    print(f"Improvement: {results['final_score'] - results['baseline_score']:+.2f}")
    print(f"Iterations:  {results['iterations']}")
    print(f"Accepted:    {results['accepted']}/{results['iterations']}")
    print(f"{'='*50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
