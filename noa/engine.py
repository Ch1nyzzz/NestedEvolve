"""NOptimizer — I-O-A-O-E 闭环编排器。"""

from __future__ import annotations

from typing import Callable

from noa.core.protocol import SystemDescription, Diagnosis, EvalResult
from noa.stages.initiator import initiate, collect_sources
from noa.stages.observer import observe
from noa.stages.analyzer import analyze
from noa.stages.optimizer import optimize
from noa.stages.evaluator import evaluate


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sort_patterns_by_severity(patterns: list[dict]) -> list[dict]:
    """按 severity 降序排列 failure patterns（high > medium > low）。"""
    return sorted(patterns, key=lambda p: _SEVERITY_ORDER.get(p.get("severity", "low"), 2))


class NOptimizer:
    """嵌套优化器，编排 Initiate -> Observe -> Analyze -> Optimize -> Evaluate 循环。

    自身也暴露 __call__ 接口，让 L2 可以以 noa/ 为 source_dir 优化 L1。
    """

    def __init__(
        self,
        source_dir: str,
        target_factory: Callable[[str], object],
        dataset: list,
        eval_fn,
        *,
        max_iterations: int = 5,
        n_samples: int = 20,
        model: str = "gpt-4.1-mini",
        system_description: str = "",
        score_fn,
        failure_threshold: float | None = None,
    ):
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.max_iterations = max_iterations
        self.n_samples = n_samples
        self.model = model
        self.system_description = system_description
        self.score_fn = score_fn
        self.failure_threshold = failure_threshold

        self.target = target_factory(source_dir)
        self.history: list[dict] = []
        self.sys_desc: SystemDescription | None = None

    def run(self) -> dict:
        """执行完整 I-O-A-O-E 循环，返回优化结果。"""
        # --- Initiate（LLM 内省，不跑 baseline）---
        print(f"\n[NOA] === Initiate ===")
        self.sys_desc = initiate(
            source_dir=self.source_dir,
            model=self.model,
            system_description=self.system_description,
        )
        print(f"[NOA] Workflow: {self.sys_desc.workflow_summary}")
        print(f"[NOA] Source files: {len(self.sys_desc.source_files)}")

        # --- 首次 Observe：既算 baseline，又作为第一轮 Analyze 的输入 ---
        print(f"[NOA] Initial observe ({self.n_samples} samples)...")
        init_trajectories = observe(
            self.target, self.dataset, self.n_samples,
            seed=43, score_fn=self.score_fn,  # seed=43 与 Evaluator 一致
        )
        baseline = sum(t.f1 for t in init_trajectories) / len(init_trajectories) * 100
        self.sys_desc.baseline_score = baseline
        print(f"[NOA] Baseline F1: {baseline:.2f}")

        consecutive_no_accept_cycles = 0
        cycle_idx = 0
        cached_trajectories = init_trajectories  # 第一轮复用

        while cycle_idx < self.max_iterations:
            cycle_idx += 1
            print(f"\n[NOA] === Cycle {cycle_idx}/{self.max_iterations} ===")

            # --- Observe（首轮复用 initiate 轨迹）---
            if cached_trajectories is not None:
                trajectories = cached_trajectories
                cached_trajectories = None
                print(f"[NOA] Reusing initial trajectories (skip observe)")
            else:
                print(f"[NOA] Observing ({self.n_samples} samples)...")
                obs_seed = 42 + cycle_idx
                trajectories = observe(self.target, self.dataset, self.n_samples, seed=obs_seed, score_fn=self.score_fn)
            self.sys_desc.source_files = collect_sources(self.source_dir)

            mean_f1 = sum(t.f1 for t in trajectories) / len(trajectories) * 100
            ft = self.failure_threshold
            if ft is None:
                scores = sorted(t.f1 for t in trajectories)
                ft = scores[len(scores) // 2] if scores else 0.5
            n_failures = sum(1 for t in trajectories if t.f1 < ft)
            print(f"[NOA] Observed F1: {mean_f1:.2f}, Failures: {n_failures}/{len(trajectories)}")

            # --- Analyze ---
            print(f"[NOA] Analyzing failures...")
            past = _format_history(self.history)
            diagnosis = analyze(self.sys_desc, trajectories, model=self.model, failure_threshold=self.failure_threshold, past_attempts=past)
            print(f"[NOA] Diagnosis: {diagnosis.summary}")

            if not diagnosis.failure_patterns:
                consecutive_no_accept_cycles += 1
                print(f"[NOA] No failure patterns found ({consecutive_no_accept_cycles}/2).")
                if consecutive_no_accept_cycles >= 2:
                    print(f"[NOA] Stopping (2 consecutive cycles with no progress).")
                    break
                continue
            # 不在这里重置 consecutive_no_accept_cycles，等内层循环判断是否有 accepted

            # --- 逐个 pattern 优化 ---
            sorted_patterns = _sort_patterns_by_severity(diagnosis.failure_patterns)
            cycle_accepted = False

            for p_idx, pattern in enumerate(sorted_patterns):
                pat_name = pattern.get("pattern", "unknown")
                pat_severity = pattern.get("severity", "?")
                print(f"\n[NOA]   --- Pattern {p_idx+1}/{len(sorted_patterns)}: [{pat_severity}] {pat_name} ---")

                # 构建单 pattern 的 Diagnosis
                single_diagnosis = Diagnosis(
                    failure_patterns=[pattern],
                    summary=pat_name,
                    raw_analysis=diagnosis.raw_analysis,
                )

                # --- Optimize ---
                past = _format_history(self.history)
                print(f"[NOA]   Generating patch for pattern: {pat_name}...")
                patch = optimize(self.sys_desc, single_diagnosis, model=self.model, past_attempts=past)
                print(f"[NOA]   Diffs: {len(patch.diffs)} blocks")
                print(f"[NOA]   Rationale: {patch.rationale}")

                if not patch.diffs:
                    print(f"[NOA]   Empty patch, skipping pattern.")
                    continue

                # --- Evaluate ---
                print(f"[NOA]   Evaluating patch...")
                result = evaluate(
                    source_files=self.sys_desc.source_files,
                    source_dir=self.source_dir,
                    patch=patch,
                    dataset=self.dataset,
                    eval_fn=self.eval_fn,
                    target_factory=self.target_factory,
                    baseline_score=baseline,
                    n_samples=self.n_samples,
                    seed=43,
                )

                status = "ACCEPTED" if result.accepted else "REJECTED"
                print(f"[NOA]   {status}: {result.before_score:.2f} -> {result.after_score:.2f}")

                self.history.append({
                    "iteration": cycle_idx,
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
                    cycle_accepted = True
                    # 刷新源码 + 重建 target + 更新 sys_desc
                    self.target = self.target_factory(self.source_dir)
                    self.sys_desc.source_files = collect_sources(self.source_dir)

            if cycle_accepted:
                consecutive_no_accept_cycles = 0
            else:
                consecutive_no_accept_cycles += 1
                if consecutive_no_accept_cycles >= 2:
                    print(f"[NOA] Stopping (2 consecutive cycles with no accepted patch).")
                    break

        print(f"\n[NOA] === Done ===")
        print(f"[NOA] Final F1: {baseline:.2f}")
        print(f"[NOA] Iterations: {len(self.history)}")
        accepted_count = sum(1 for h in self.history if h["accepted"])
        print(f"[NOA] Accepted patches: {accepted_count}/{len(self.history)}")

        return {
            "final_score": baseline,
            "baseline_score": self.sys_desc.baseline_score,
            "iterations": len(self.history),
            "accepted": accepted_count,
            "history": self.history,
        }

    def __call__(self, **kwargs) -> dict:
        """执行一轮完整优化并返回结果，供 L2 使用。"""
        return self.run()


def _truncate_line(line: str, max_chars: int = 100) -> str:
    return line if len(line) <= max_chars else line[:max_chars - 3] + "..."


def _format_diff_block(diff, max_lines: int = 30) -> str:
    """将单个 DiffBlock 格式化为紧凑摘要。"""
    search_lines = diff.search.splitlines()
    replace_lines = diff.replace.splitlines()
    # 截断到 max_lines
    if len(search_lines) > max_lines:
        search_lines = search_lines[:max_lines] + [f"... ({len(search_lines) - max_lines} more lines)"]
    if len(replace_lines) > max_lines:
        replace_lines = replace_lines[:max_lines] + [f"... ({len(replace_lines) - max_lines} more lines)"]
    search_text = "\n".join(_truncate_line(l) for l in search_lines)
    replace_text = "\n".join(_truncate_line(l) for l in replace_lines)
    return (
        f"## File: {diff.file_path}\n"
        f"<<<<<<< SEARCH\n{search_text}\n"
        f"=======\n{replace_text}\n"
        f">>>>>>> REPLACE"
    )


def _format_history(history: list[dict]) -> str:
    """将历史尝试格式化为 LLM 可读摘要。"""
    if not history:
        return ""
    parts = []
    for h in history:
        status = "ACCEPTED" if h["accepted"] else "REJECTED"
        header = f"### Attempt {h['iteration']} [{status}] {h['before']:.2f} → {h['after']:.2f}"
        diff_text = "\n".join(_format_diff_block(d) for d in h["diffs"])
        parts.append(f"{header}\nRationale: {h['rationale']}\n{diff_text}")
    return "\n\n".join(parts)
