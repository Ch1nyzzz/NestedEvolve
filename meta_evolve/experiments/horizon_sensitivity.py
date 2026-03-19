"""Horizon sensitivity 实验：测量不同 inner_steps 下 ES pair_diff 的增长。

用法:
    python -m meta_evolve.experiments.horizon_sensitivity [--pairs 5] [--tasks circle_packing,first_autocorr_ineq]

输出: 每个 horizon 的 median/mean |pair_diff|，判断 horizon 是否是 ES 信噪比瓶颈。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np

from meta_evolve.evolve_loop import run_inner_loop
from meta_evolve.llm_client import LLMClient
from meta_evolve.strategy import StrategyParams
from meta_evolve.task_adapter import load_task_parallel

HORIZONS = [10, 20, 40, 80]
ES_SIGMA = 0.3  # 与 MetaLearner 一致


async def run_horizon_experiment(
    tasks_names: list[str],
    n_pairs: int = 5,
    max_concurrency: int = 500,
    horizons: list[int] | None = None,
):
    horizons = horizons or HORIZONS
    llm = LLMClient()
    sem = asyncio.Semaphore(max_concurrency)

    # 加载任务
    tasks = load_task_parallel(tasks_names)
    print(f"任务: {[t.name for t in tasks]}")

    # 固定 φ center（默认策略）
    phi_center = StrategyParams().to_vector()

    # 预生成 epsilons（所有 horizon 共享同一组扰动）
    rng = np.random.RandomState(42)
    epsilons = [
        rng.randn(StrategyParams.PHI_DIM).astype(np.float32) * ES_SIGMA
        for _ in range(n_pairs)
    ]

    results = {}

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"Horizon = {horizon} steps, {n_pairs} pairs × {len(tasks)} tasks")
        print(f"{'='*60}")
        t0 = time.time()

        # 构建所有 rollout（+ε 和 -ε 成对）
        rollout_coros = []
        for pair_idx, eps in enumerate(epsilons):
            phi_plus = phi_center + eps
            phi_minus = phi_center - eps
            pair_seed = 99999 + pair_idx * 100  # 固定 seed

            for phi_vec in [phi_plus, phi_minus]:
                params = StrategyParams.from_vector(phi_vec)
                for i, task in enumerate(tasks):
                    rollout_coros.append(
                        run_inner_loop(
                            task=task,
                            params=params,
                            n_steps=horizon,
                            llm=llm,
                            max_pop_size=30,
                            analyze_interval=max(1, horizon // 4),
                            global_sem=sem,
                            rng_seed=pair_seed + i * 1000,
                        )
                    )

        # 并行执行所有 rollout
        all_results = await asyncio.gather(*rollout_coros)

        # 解析结果: 每 pair 有 2 * len(tasks) 个结果
        chunk = len(tasks)  # 每个 φ 对应 chunk 个任务
        pair_diffs = []
        for pair_idx in range(n_pairs):
            base = pair_idx * 2 * chunk
            # +ε 的各任务 scores
            plus_scores = [all_results[base + i].final_best_score for i in range(chunk)]
            # -ε 的各任务 scores
            minus_scores = [
                all_results[base + chunk + i].final_best_score for i in range(chunk)
            ]
            mean_plus = float(np.mean(plus_scores))
            mean_minus = float(np.mean(minus_scores))
            diff = mean_plus - mean_minus
            pair_diffs.append(diff)
            print(
                f"  pair {pair_idx}: +ε={mean_plus:.6f}, -ε={mean_minus:.6f}, diff={diff:+.6f}"
            )

        abs_diffs = [abs(d) for d in pair_diffs]
        median_diff = float(np.median(abs_diffs))
        mean_diff = float(np.mean(abs_diffs))
        elapsed = time.time() - t0

        results[horizon] = {
            "median_abs_diff": median_diff,
            "mean_abs_diff": mean_diff,
            "raw_diffs": pair_diffs,
            "elapsed_sec": elapsed,
        }
        print(f"\n  → median |pair_diff| = {median_diff:.6f}")
        print(f"  → mean   |pair_diff| = {mean_diff:.6f}")
        print(f"  → {elapsed:.1f}s")

    # 汇总
    print(f"\n{'='*60}")
    print("SUMMARY: horizon → median |pair_diff|")
    print(f"{'='*60}")
    for h in horizons:
        r = results[h]
        print(
            f"  {h:4d} steps: median={r['median_abs_diff']:.6f}, mean={r['mean_abs_diff']:.6f}"
        )

    # 保存结果
    out_path = os.path.join(
        os.path.dirname(__file__), "horizon_sensitivity_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Horizon sensitivity experiment")
    parser.add_argument("--pairs", type=int, default=5, help="antithetic 对数")
    parser.add_argument(
        "--tasks",
        type=str,
        default="circle_packing,first_autocorr_ineq",
        help="逗号分隔的任务名（少用几个省算力）",
    )
    parser.add_argument("--concurrency", type=int, default=500)
    parser.add_argument(
        "--horizons",
        type=str,
        default="10,20,40,80",
        help="逗号分隔的 horizon 值",
    )
    args = parser.parse_args()

    task_names = [t.strip() for t in args.tasks.split(",")]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    asyncio.run(
        run_horizon_experiment(
            tasks_names=task_names,
            n_pairs=args.pairs,
            max_concurrency=args.concurrency,
            horizons=horizons,
        )
    )


if __name__ == "__main__":
    main()
