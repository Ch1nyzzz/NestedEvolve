"""Analyzer — LLM 错误诊断（核心阶段，分配最多预算）。"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm import llm_call, DEFAULT_MODEL
from noa.core.protocol import SystemDescription, Trajectory, Diagnosis, FailurePool
from noa.core import prompts

log = logging.getLogger(__name__)
_MIN_ANALYZABLE_INTERMEDIATE_COVERAGE = 0.95


def _format_trajectory(t: Trajectory, label: str) -> str:
    """将单条轨迹格式化为 LLM 可读文本。"""
    parts = [
        f"[{label}] F1={t.f1:.2f}",
        f"Q: {t.question}",
        f"GT: {t.ground_truth}",
        f"Pred: {t.prediction}",
    ]
    if t.error:
        parts.append(f"ERROR: {t.error[:500]}")
    if not t.intermediate:
        parts.append("  intermediate: (missing)")
    elif _is_l1_result(t.intermediate):
        parts.append(_format_l1_intermediate(t.intermediate))
    else:
        for comp_name, output in t.intermediate.items():
            if isinstance(output, dict):
                for k, v in output.items():
                    val = str(v)
                    parts.append(f"  {comp_name}.{k}: {val}")
            else:
                parts.append(f"  {comp_name}: {str(output)[:500]}")
    return "\n".join(parts)


def _is_l1_result(intermediate: dict) -> bool:
    """检测 intermediate 是否为 L1 optimizer 运行结果。"""
    return "final_score" in intermediate and "history" in intermediate


def _format_l1_intermediate(result: dict) -> str:
    """将 L1 optimizer 运行结果格式化为 L2 analyzer 可读的结构化文本。"""
    final = result.get("final_score", 0)
    baseline = result.get("baseline_score", 0)
    accepted = result.get("accepted", 0)
    steps = result.get("steps", result.get("planner_steps", 0))
    budget = result.get("budget_usage", {})

    lines = [
        "  [META-NOTE: Below is a LOWER-LAYER optimizer run result. "
        "Diagnose the optimizer's STRATEGY flaws, not the target system's bugs.]",
        f"  [L1 Run Result] baseline={baseline:.2f} → final={final:.2f} (Δ={final - baseline:+.2f})",
        f"  accepted_patches={accepted}, steps={steps}",
    ]
    if budget:
        lines.append(f"  budget: {budget}")

    history = result.get("history", [])
    if not history:
        lines.append("  (no history)")
        return "\n".join(lines)

    # 提取 analyze/propose_patch/evaluate 步骤的关键信息
    iter_count = 0
    for h in history:
        action = h.get("action", "")
        payload = h.get("payload", {})

        if action == "analyze" and h.get("ok"):
            diag = payload.get("diagnosis_summary", "") or h.get("summary", "")
            n_patterns = payload.get("n_patterns", "?")
            lines.append(f"  [Analyze] {diag[:200]} (patterns={n_patterns})")

        elif action == "propose_patch" and h.get("ok"):
            n_diffs = payload.get("n_diffs", "?")
            rationale = payload.get("rationale", "")[:150]
            lines.append(f"  [Patch] diffs={n_diffs}, rationale={rationale}")
            for ds in payload.get("diff_summaries", [])[:3]:
                if isinstance(ds, dict):
                    fp = ds.get("file_path", "?")
                    search = ds.get("search", "")[:80]
                    replace = ds.get("replace", "")[:80]
                    lines.append(f"    {fp}: {search!r} → {replace!r}")

        elif action in ("evaluate_patch", "parallel_optimize"):
            iter_count += 1
            acc = payload.get("accepted", False)
            before = payload.get("before", 0)
            after = payload.get("after", 0)
            status = "ACCEPTED" if acc else "REJECTED"
            lines.append(f"  [Eval-{iter_count}] {status} {before:.2f}→{after:.2f}")
            fb = payload.get("eval_feedback", {})
            if fb.get("next_focus"):
                lines.append(f"    next_focus: {fb['next_focus'][:150]}")

    return "\n".join(lines)


def analyze_incremental(
    sys_desc: SystemDescription,
    trajectories: list[Trajectory],
    model: str = DEFAULT_MODEL,
    failure_threshold: float | None = None,
    past_attempts: str = "",
    pool: FailurePool | None = None,
    top_n: int = 10,
    max_concurrency: int = 8,
    probe=None,
    max_tool_calls: int = 10,
    layer_context: str = "",
    eval_feedback: str = "",
    stats: dict | None = None,
) -> tuple[Diagnosis, FailurePool]:
    """逐条分析失败轨迹，并行调用 LLM，累积到 FailurePool，返回 top-N pattern 的 Diagnosis。

    有 probe 时使用 agentic 模式（ReAct 循环 + 工具调用），无 probe 时 fallback 到单次 LLM 调用。
    """
    if pool is None:
        pool = FailurePool()

    if failure_threshold is None:
        scores = sorted(t.f1 for t in trajectories)
        failure_threshold = scores[len(scores) // 2] if scores else 0.5
    failures = [t for t in trajectories if t.f1 < failure_threshold]

    if not failures:
        return Diagnosis(
            failure_patterns=[], summary="No failures.", raw_analysis=""
        ), pool

    # 快照当前 pool 状态，所有并行调用共享同一份上下文
    pool_snapshot = pool.to_context_str()
    sys_context = sys_desc.to_context_str()
    source_code = sys_desc.get_source_context()
    trajectory_quality = _trajectory_quality_context(trajectories, failures)

    # 构建 eval feedback 上下文
    eval_feedback_section = ""
    if eval_feedback:
        eval_feedback_section = f"\n## Previous Evaluation Feedback\n{eval_feedback}\n"

    def _diagnose_one_simple(t: Trajectory) -> tuple[Trajectory, list[dict]]:
        """Fallback：单次 LLM 调用（无 probe）。"""
        traj_text = _format_trajectory(t, label="FAIL")
        prompt = prompts.SINGLE_ANALYZER_PROMPT.format(
            system_context=sys_context,
            source_code=source_code,
            trajectory=traj_text,
            trajectory_quality=trajectory_quality,
            pool_context=pool_snapshot,
            past_attempts=(past_attempts or "(none)") + eval_feedback_section,
            layer_context=layer_context,
        )
        resp = llm_call(
            prompt,
            model=model,
            max_tokens=4096,
            temperature=0,
            system=prompts.SINGLE_ANALYZER_SYSTEM,
        )
        if stats is not None:
            stats["llm_calls"] = stats.get("llm_calls", 0) + 1
        return t, _parse_patterns(resp.text)

    def _diagnose_one_agentic(t: Trajectory) -> tuple[Trajectory, list[dict]]:
        """Agentic 模式：ReAct 循环 + 工具调用。"""
        from noa.stages.agentic import agentic_loop

        traj_text = _format_trajectory(t, label="FAIL")
        user_content = prompts.AGENTIC_ANALYZER_PROMPT.format(
            system_context=sys_context,
            source_code=source_code,
            trajectory=traj_text,
            trajectory_quality=trajectory_quality,
            pool_context=pool_snapshot,
            past_attempts=(past_attempts or "(none)") + eval_feedback_section,
            layer_context=layer_context,
        )
        messages = [
            {"role": "system", "content": prompts.AGENTIC_ANALYZER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        tools = probe.get_tool_schemas()

        loop_stats: dict = {}
        final_text = agentic_loop(
            messages=messages,
            tools=tools,
            tool_executor=probe.execute_tool,
            model=model,
            max_tool_calls=max_tool_calls,
            parse_fn=lambda text: _parse_patterns(text) or None,
            json_retries=2,
            budget_exhausted_prompt="Tool call budget exhausted. Output your diagnosis now as JSON.",
            invalid_json_prompt="Your last response was not valid JSON list of patterns. Output ONLY valid JSON now.",
            stats=loop_stats,
        )
        if stats is not None:
            stats["llm_calls"] = stats.get("llm_calls", 0) + loop_stats.get(
                "llm_calls", 1
            )
        return t, _parse_patterns(final_text)

    diagnose_fn = _diagnose_one_agentic if probe is not None else _diagnose_one_simple
    mode_label = "agentic" if probe is not None else "simple"

    # 并行调用 LLM，受 max_concurrency 限制（避免 rate limit）
    workers = min(max_concurrency, len(failures))
    log.info(
        "Analyzing %d failures with %d parallel workers (%s mode)",
        len(failures),
        workers,
        mode_label,
    )

    results: list[tuple[Trajectory, list[dict]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(diagnose_fn, t): t for t in failures}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                t = futures[future]
                log.warning(
                    "Failed to diagnose trajectory: %s", t.question[:60], exc_info=True
                )

    # 按原始 failure 顺序合并到 pool（保持确定性）
    order = {id(t): i for i, t in enumerate(failures)}
    results.sort(key=lambda r: order.get(id(r[0]), 0))
    for t, patterns in results:
        pool.add(patterns, example_question=t.question)
    obs_gap = _observability_gap_pattern(trajectories, failures)
    if obs_gap is not None:
        pool.add([obs_gap], example_question="(observer data quality)")

    # 并行模式兜底：合并措辞不同但语义相同的 pattern
    n_merged = pool.consolidate()
    if n_merged:
        log.info("Consolidated %d duplicate patterns after parallel merge", n_merged)

    # 取 top-N 作为本轮诊断结果
    top_patterns = pool.top_n(top_n)

    # 校验 affected_file 是否在 source_files 范围内
    valid_files = {sf.path for sf in sys_desc.source_files}
    for p in top_patterns:
        af = p.get("affected_file", "")
        if af and af not in valid_files:
            log.warning("Pattern affected_file '%s' not in source_files, clearing", af)
            p["affected_file"] = ""
            p["root_cause"] = (
                p.get("root_cause", "")
                + f" [NOTE: Originally referenced '{af}' which is outside writable scope]"
            )

    summary = (
        "; ".join(f"{p['pattern']} (x{p['count']})" for p in top_patterns[:3])
        if top_patterns
        else "No patterns found."
    )

    log.info(
        "FailurePool: %d unique patterns from %d failures, top-%d selected",
        len(pool),
        len(failures),
        min(top_n, len(pool)),
    )

    return Diagnosis(
        failure_patterns=top_patterns,
        summary=summary,
        raw_analysis=f"Pool size: {len(pool)}, top {top_n} selected. Mode: {mode_label}.",
    ), pool


def _trajectory_quality_context(
    trajectories: list[Trajectory], failures: list[Trajectory]
) -> str:
    total = len(trajectories)
    fail_n = len(failures)
    total_with = sum(
        1
        for t in trajectories
        if isinstance(t.intermediate, dict) and bool(t.intermediate)
    )
    fail_with = sum(
        1 for t in failures if isinstance(t.intermediate, dict) and bool(t.intermediate)
    )
    total_cov = (total_with / total) if total else 0.0
    fail_cov = (fail_with / fail_n) if fail_n else 0.0
    return (
        f"- total trajectories: {total}\n"
        f"- trajectories with intermediate: {total_with} ({total_cov:.1%})\n"
        f"- failure trajectories: {fail_n}\n"
        f"- failures with intermediate: {fail_with} ({fail_cov:.1%})\n"
        "- note: low intermediate coverage means root-cause localization is under-observed."
    )


def _observability_gap_pattern(
    trajectories: list[Trajectory], failures: list[Trajectory]
) -> dict | None:
    if not trajectories:
        return None
    with_intermediate = sum(
        1
        for t in trajectories
        if isinstance(t.intermediate, dict) and bool(t.intermediate)
    )
    coverage = with_intermediate / len(trajectories)
    if coverage >= _MIN_ANALYZABLE_INTERMEDIATE_COVERAGE:
        return None
    missing_failures = sum(
        1
        for t in failures
        if not (isinstance(t.intermediate, dict) and bool(t.intermediate))
    )
    return {
        "pattern": "Trajectory reuse has incomplete component evidence (missing intermediate)",
        "root_cause": (
            "A substantial portion of trajectories lacks component-level intermediate outputs. "
            "Without boundary evidence, diagnosis degenerates into output-text guesses."
        ),
        "affected_component": "Observer data reuse / analyzer evidence pipeline",
        "severity": "high",
        "affected_file": "noa/stages/observer.py",
        "suggested_fix": (
            "Replay or top-up reused trajectories until intermediate coverage is high before proposing patches."
        ),
        "count": max(1, missing_failures),
        "evidence": f"intermediate coverage={coverage:.1%}, missing_failure_samples={missing_failures}",
        "confidence": 0.95,
    }


def format_parent_history(l1_history: list[dict]) -> str:
    """将 L1 history 列表格式化为 LLM 可读文本。"""
    if not l1_history:
        return "(no L1 history)"
    parts = []
    for h in l1_history:
        status = "ACCEPTED" if h.get("accepted") else "REJECTED"
        before = h.get("before", 0)
        after = h.get("after", 0)
        header = f"### Iteration {h.get('iteration', '?')} [{status}] {before:.2f} → {after:.2f}"
        parts.append(header)
        parts.append(f"Diagnosis: {h.get('diagnosis', 'N/A')}")
        parts.append(f"Rationale: {h.get('rationale', 'N/A')}")

        eval_error = h.get("eval_error")
        if eval_error:
            parts.append(f"Eval Error: {eval_error[:300]}")
        eval_details = h.get("eval_details", [])
        low_score = [d for d in eval_details if d.get("f1", 1) < 0.5][:3]
        if low_score:
            parts.append("Low-score samples:")
            for s in low_score:
                parts.append(
                    f"  Q: {s.get('question', '?')[:80]} | F1={s.get('f1', '?')}"
                )

        diffs = h.get("diffs", [])
        for d in diffs:
            if isinstance(d, dict):
                fp = d.get("file_path", "?")
                search = d.get("search", "")[:200]
                replace = d.get("replace", "")[:200]
            else:
                fp = getattr(d, "file_path", "?")
                search = getattr(d, "search", "")[:200]
                replace = getattr(d, "replace", "")[:200]
            parts.append(f"  File: {fp}")
            parts.append(f"  SEARCH: {search}")
            parts.append(f"  REPLACE: {replace}")
        parts.append("")
    return "\n".join(parts)


def _parse_patterns(text: str) -> list[dict]:
    """从 LLM 输出中提取 failure patterns JSON 列表。"""
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return _normalize_patterns(result)
    except json.JSONDecodeError:
        pass
    left = text.find("[")
    right = text.rfind("]")
    if left != -1 and right != -1:
        try:
            result = json.loads(text[left : right + 1])
            if isinstance(result, list):
                return _normalize_patterns(result)
        except json.JSONDecodeError:
            pass
    return []


def _normalize_patterns(patterns: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        item = dict(p)
        if "confidence" not in item:
            item["confidence"] = 0.5
        else:
            try:
                item["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
            except Exception:
                item["confidence"] = 0.5
        if "evidence" not in item:
            evidence = item.get("root_cause") or item.get("suggested_fix") or ""
            item["evidence"] = str(evidence)[:300]
        normalized.append(item)
    return normalized
