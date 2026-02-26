"""Orchestrator — L1+L2 自主嵌套优化编排器。

运行 L1 后自动启动 L2（meta-optimizer），
L2 直接消费 L1 运行历史，仅在 Evaluate 阶段运行 mini L1 验证 patch。
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from noa.engine import _format_history, _sort_patterns_by_severity
from noa.core.protocol import SystemDescription, Diagnosis
from noa.stages.initiator import collect_sources
from noa.stages.analyzer import meta_analyze, _format_l1_history
from noa.stages.optimizer import optimize
from noa.stages.evaluator import evaluate
from noa.subprocess_runner import run_l1_subprocess, serialize_dataset
from noa.workspace import WorkspaceManager


class L1Target:
    """L2 的 "target" — 调用时在 subprocess 中运行 L1。"""

    def __init__(
        self,
        noa_dir: str,
        project_root: str,
        l0_source_dir: str,
        dataset_pickle_path: str,
        iterations: int,
        n_samples: int,
        model: str,
        eval_n_samples: int = 20,
    ):
        self.noa_dir = noa_dir
        self.project_root = project_root
        self.l0_source_dir = l0_source_dir
        self.dataset_pickle_path = dataset_pickle_path
        self.iterations = iterations
        self.n_samples = n_samples
        self.model = model
        self.eval_n_samples = eval_n_samples

    def __call__(self, question: str) -> SimpleNamespace:
        """question 被忽略，仅作触发。返回 L1 运行结果。"""
        result = run_l1_subprocess(
            noa_dir=self.noa_dir,
            project_root=self.project_root,
            l0_source_dir=self.l0_source_dir,
            dataset_pickle_path=self.dataset_pickle_path,
            iterations=self.iterations,
            n_samples=self.n_samples,
            eval_n_samples=self.eval_n_samples,
            model=self.model,
            isolate_source=True,  # L2 调用 L1 时必须隔离，避免 L1 patch 污染共享 target 代码
        )
        if result.get("error"):
            print(f"[L1Target] L1 subprocess error: {result['error'][:200]}")
        return SimpleNamespace(
            answer=str(result.get("final_score", 0)),
            intermediate=result,
        )


class Orchestrator:
    """薄层编排器 — 运行 L1 后判断是否升级到 L2。"""

    def __init__(
        self,
        source_dir: str,
        target_factory,
        dataset: list,
        eval_fn,
        score_fn,
        *,
        l1_max_iterations: int = 5,
        l1_n_samples: int = 20,
        l1_eval_n_samples: int = 20,
        l1_model: str = "gpt-4.1-mini",
        l2_max_iterations: int = 3,
        l2_n_samples: int | None = None,
        l2_model: str = "gpt-4.1-mini",
        max_l2_rounds: int = 2,
    ):
        self.source_dir = os.path.abspath(source_dir)
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.score_fn = score_fn

        self.l1_max_iterations = l1_max_iterations
        self.l1_n_samples = l1_n_samples
        self.l1_eval_n_samples = l1_eval_n_samples
        self.l1_model = l1_model

        self.l2_max_iterations = l2_max_iterations
        self.l2_n_samples = l2_n_samples if l2_n_samples is not None else 1
        self.l2_model = l2_model
        self.max_l2_rounds = max_l2_rounds

        # 推导项目根目录和 noa 目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.noa_dir = os.path.join(self.project_root, "noa")

        # 序列化 dataset（复用）
        cache_dir = os.path.join(self.project_root, ".noa_cache")
        self.dataset_pickle_path = serialize_dataset(dataset, cache_dir=cache_dir)

    def run(self) -> dict:
        """执行 L1+L2 嵌套优化，返回最终结果。

        原始代码不可变 — 所有修改在 workspace 副本上进行。
        """
        # 创建 workspace，复制原始代码
        ws = WorkspaceManager(self.project_root, self.source_dir)
        ws_noa, ws_source = ws.setup()
        self._ws_noa = ws_noa
        self._ws_source = ws_source

        print(f"[Orchestrator] Workspace 创建完成: {ws.run_dir}")

        all_rounds: list[dict] = []

        for round_idx in range(1 + self.max_l2_rounds):
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Round {round_idx} — 运行 L1")
            print(f"{'='*60}")

            l1_result = self._run_l1_subprocess()
            all_rounds.append({"round": round_idx, "l1_result": l1_result})
            ws.snapshot(f"after_l1_round_{round_idx}")

            print(f"[Orchestrator] L1 结果: baseline={l1_result.get('baseline_score', '?')}, "
                  f"final={l1_result.get('final_score', '?')}, "
                  f"accepted={l1_result.get('accepted', '?')}/{l1_result.get('iterations', '?')}")

            if l1_result.get("error"):
                print(f"[Orchestrator] L1 出错: {l1_result['error'][:200]}")

            # 最后一轮不再升级
            if round_idx >= self.max_l2_rounds:
                break

            print(f"\n{'='*60}")
            print(f"[Orchestrator] L1 已饱和 — 启动 L2 Meta-Optimizer")
            print(f"{'='*60}")

            l2_result = self._run_l2(l1_result)
            all_rounds[-1]["l2_result"] = l2_result
            ws.snapshot(f"after_l2_round_{round_idx}")

        final_result = all_rounds[-1]["l1_result"]
        print(f"\n[Orchestrator] 原始代码未修改")
        print(f"[Orchestrator] 快照目录: {ws.run_dir}")
        return {
            "final_score": final_result.get("final_score", 0),
            "baseline_score": all_rounds[0]["l1_result"].get("baseline_score", 0),
            "rounds": all_rounds,
            "total_rounds": len(all_rounds),
            "run_dir": ws.run_dir,
            "snapshots": ws.list_snapshots(),
        }

    def _run_l1_subprocess(self) -> dict:
        """在 subprocess 中运行 L1（使用 workspace 副本）。"""
        return run_l1_subprocess(
            noa_dir=self._ws_noa,
            project_root=self.project_root,
            l0_source_dir=self._ws_source,
            dataset_pickle_path=self.dataset_pickle_path,
            iterations=self.l1_max_iterations,
            n_samples=self.l1_n_samples,
            eval_n_samples=self.l1_eval_n_samples,
            model=self.l1_model,
        )

    def _run_l2(self, l1_result: dict) -> dict:
        """运行 L2 meta-optimizer：直接分析 L1 历史，跳过 Initiate/Observe。"""
        l1_history = l1_result.get("history", [])

        # 1. 收集 noa/ 源码
        source_files = collect_sources(self._ws_noa)

        # 2. 构建 SystemDescription（不调用 LLM，不运行 L1）
        sys_desc = SystemDescription(
            workflow_summary="NOA I-O-A-O-E optimizer loop: Initiate → Observe → Analyze → Optimize → Evaluate",
            component_names=["initiator", "observer", "analyzer", "optimizer", "evaluator"],
            source_files=source_files,
            source_dir=self._ws_noa,
            baseline_score=l1_result.get("final_score", 0),
        )

        # 3. L2 target_factory + eval_fn（仅 Evaluate 阶段用，跑完整 L1）
        project_root = self.project_root
        l0_source_dir = self._ws_source
        dataset_pickle = self.dataset_pickle_path
        l1_iters = self.l1_max_iterations
        l1_samples = self.l1_n_samples
        model = self.l2_model

        l1_eval_samples = self.l1_eval_n_samples

        def l2_target_factory(noa_dir: str) -> L1Target:
            return L1Target(
                noa_dir=noa_dir,
                project_root=project_root,
                l0_source_dir=l0_source_dir,
                dataset_pickle_path=dataset_pickle,
                iterations=l1_iters,
                n_samples=l1_samples,
                model=model,
                eval_n_samples=l1_eval_samples,
            )

        l2_n = self.l2_n_samples  # 默认 1
        l2_dataset = [
            SimpleNamespace(question="run_l1", answer="100", id=f"l1_eval_{i}")
            for i in range(l2_n)
        ]

        def l2_eval_fn(target, dataset_subset):
            scores = []
            for item in dataset_subset:
                result = target(item.question)
                scores.append(result.intermediate.get("final_score", 0))
            return {"score": sum(scores) / len(scores) if scores else 0}

        # 4. 循环：meta_analyze → 逐个 pattern optimize → evaluate
        print(f"[Orchestrator] L2 开始 meta-optimize noa/ 代码 (max_iterations={self.l2_max_iterations})")
        baseline = l1_result.get("final_score", 0)
        l2_history: list[dict] = []
        consecutive_no_accept_cycles = 0
        accepted_count = 0

        for i in range(self.l2_max_iterations):
            print(f"\n[L2] === Cycle {i+1}/{self.l2_max_iterations} ===")

            # 刷新源码
            sys_desc.source_files = collect_sources(self._ws_noa)

            # --- Meta-Analyze（0 次 L1 调用）---
            print(f"[L2] Meta-analyzing L1 history...")
            past = _format_history(l2_history)
            diagnosis = meta_analyze(sys_desc, l1_history, model=self.l2_model, past_attempts=past)
            print(f"[L2] Diagnosis: {diagnosis.summary}")

            if not diagnosis.failure_patterns:
                consecutive_no_accept_cycles += 1
                print(f"[L2] No patterns found ({consecutive_no_accept_cycles}/2).")
                if consecutive_no_accept_cycles >= 2:
                    print(f"[L2] Stopping (2 consecutive cycles with no progress).")
                    break
                continue

            # --- 逐个 pattern 优化 ---
            sorted_patterns = _sort_patterns_by_severity(diagnosis.failure_patterns)
            cycle_accepted = False

            for p_idx, pattern in enumerate(sorted_patterns):
                pat_name = pattern.get("pattern", "unknown")
                pat_severity = pattern.get("severity", "?")
                print(f"\n[L2]   --- Pattern {p_idx+1}/{len(sorted_patterns)}: [{pat_severity}] {pat_name} ---")

                single_diagnosis = Diagnosis(
                    failure_patterns=[pattern],
                    summary=pat_name,
                    raw_analysis=diagnosis.raw_analysis,
                )

                # --- Optimize（0 次 L1 调用）---
                past = _format_history(l2_history)
                print(f"[L2]   Generating patch for pattern: {pat_name}...")
                patch = optimize(sys_desc, single_diagnosis, model=self.l2_model, past_attempts=past)
                print(f"[L2]   Diffs: {len(patch.diffs)} blocks")
                print(f"[L2]   Rationale: {patch.rationale}")

                if not patch.diffs:
                    print(f"[L2]   Empty patch, skipping pattern.")
                    continue

                # --- Evaluate（运行完整 L1 验证）---
                print(f"[L2]   Evaluating patch (running full L1)...")
                result = evaluate(
                    source_files=sys_desc.source_files,
                    source_dir=self._ws_noa,
                    patch=patch,
                    dataset=l2_dataset,
                    eval_fn=l2_eval_fn,
                    target_factory=l2_target_factory,
                    baseline_score=baseline,
                    n_samples=l2_n,
                    seed=43,
                )

                status = "ACCEPTED" if result.accepted else "REJECTED"
                print(f"[L2]   {status}: {result.before_score:.2f} -> {result.after_score:.2f}")

                l2_history.append({
                    "iteration": i + 1,
                    "pattern": pat_name,
                    "severity": pat_severity,
                    "diagnosis": single_diagnosis.summary,
                    "diffs": patch.diffs,
                    "rationale": patch.rationale,
                    "before": result.before_score,
                    "after": result.after_score,
                    "accepted": result.accepted,
                })

                if result.accepted:
                    baseline = result.after_score
                    accepted_count += 1
                    cycle_accepted = True
                    # 刷新源码
                    sys_desc.source_files = collect_sources(self._ws_noa)

            if cycle_accepted:
                consecutive_no_accept_cycles = 0
            else:
                consecutive_no_accept_cycles += 1
                if consecutive_no_accept_cycles >= 2:
                    print(f"[L2] Stopping (2 consecutive cycles with no accepted patch).")
                    break

        print(f"\n[Orchestrator] L2 完成: final_score={baseline:.2f}, "
              f"accepted={accepted_count}/{len(l2_history)}")

        return {
            "final_score": baseline,
            "baseline_score": l1_result.get("final_score", 0),
            "iterations": len(l2_history),
            "accepted": accepted_count,
            "history": l2_history,
        }
