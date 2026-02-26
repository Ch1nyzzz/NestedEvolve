"""基线评估入口 — 加载数据 → 构建 pipeline → 评估 → 输出 F1。"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 中的 API key
load_dotenv()

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data import load_hotpotqa, build_corpus
from target_systems.hotpotqa_rag.config import SystemConfig
from target_systems.hotpotqa_rag.pipeline import RAGPipeline
from target_systems.hotpotqa_rag.evaluate import evaluate_batch


def main():
    parser = argparse.ArgumentParser(description="HotpotQA RAG Baseline Evaluation")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--k", type=int, default=None, help="Override retriever k")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--workers", type=int, default=8, help="Max parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = SystemConfig.from_json(args.config)
    else:
        config = SystemConfig()

    if args.k is not None:
        config.retriever.k = args.k

    # 加载数据
    print(f"Loading HotpotQA data...")
    dataset = load_hotpotqa(split=args.split, n=args.n)

    # 构建语料库（本地检索需要）
    corpus = None
    if config.retriever.backend == "local":
        print("Building TF-IDF corpus...")
        corpus = build_corpus(split=args.split)

    print(f"Split: {args.split}, Examples: {len(dataset)}, Retriever k: {config.retriever.k}")

    # 构建 pipeline
    pipeline = RAGPipeline(config, corpus=corpus)

    # 评估
    results = evaluate_batch(pipeline, dataset, max_workers=args.workers)
    print(f"\n{'='*40}")
    print(f"Mean F1: {results['score']:.2f}")
    print(f"{'='*40}")

    # 保存结果
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
