"""Analyzer — LLM 错误诊断（核心阶段，分配最多预算）。"""

from __future__ import annotations

import json
import logging

from utils.llm import llm_call
from noa.core.protocol import SystemDescription, Trajectory, Diagnosis, FailurePool
from noa.core import prompts

log = logging.getLogger(__name__)


def analyze(
    sys_desc: SystemDescription,
    trajectories: list[Trajectory],
    model: str = "gpt-4.1-mini",
    failure_threshold: float | None = None,
    past_attempts: str = "",
) -> Diagnosis:
    """分析失败轨迹 + 源码，输出结构化诊断。"""
    if failure_threshold is None:
        scores = sorted(t.f1 for t in trajectories)
        failure_threshold = scores[len(scores) // 2] if scores else 0.5
    failures = [t for t in trajectories if t.f1 < failure_threshold]
    successes = [t for t in trajectories if t.f1 >= failure_threshold]

    # 构造轨迹文本：所有失败 + 少量成功做对比
    traj_lines = []
    for t in failures:
        traj_lines.append(_format_trajectory(t, label="FAIL"))
    for t in successes[:3]:
        traj_lines.append(_format_trajectory(t, label="OK"))

    prompt = prompts.ANALYZER_PROMPT.format(
        system_context=sys_desc.to_context_str(),
        source_code=sys_desc.get_source_context(),
        n_failures=len(failures),
        n_total=len(trajectories),
        trajectories="\n---\n".join(traj_lines) if traj_lines else "(no trajectories)",
        past_attempts=past_attempts or "(none)",
    )

    resp = llm_call(
        prompt, model=model, max_tokens=16384,
        temperature=0, system=prompts.ANALYZER_SYSTEM,
    )

    patterns = _parse_patterns(resp.text)
    summary = "; ".join(p.get("pattern", "") for p in patterns[:3]) if patterns else "No patterns found."

    return Diagnosis(
        failure_patterns=patterns,
        summary=summary,
        raw_analysis=resp.text,
    )


def _format_trajectory(t: Trajectory, label: str) -> str:
    """将单条轨迹格式化为 LLM 可读文本。"""
    parts = [
        f"[{label}] F1={t.f1:.2f}",
        f"Q: {t.question}",
        f"GT: {t.ground_truth}",
        f"Pred: {t.prediction}",
    ]
    for comp_name, output in t.intermediate.items():
        if isinstance(output, dict):
            for k, v in output.items():
                val = str(v)
                if len(val) > 200:
                    val = val[:200] + "..."
                parts.append(f"  {comp_name}.{k}: {val}")
    return "\n".join(parts)


def analyze_incremental(
    sys_desc: SystemDescription,
    trajectories: list[Trajectory],
    model: str = "gpt-4.1-mini",
    failure_threshold: float | None = None,
    past_attempts: str = "",
    pool: FailurePool | None = None,
    top_n: int = 5,
) -> tuple[Diagnosis, FailurePool]:
    """逐条分析失败轨迹，累积到 FailurePool，返回 top-N pattern 的 Diagnosis。"""
    if pool is None:
        pool = FailurePool()

    if failure_threshold is None:
        scores = sorted(t.f1 for t in trajectories)
        failure_threshold = scores[len(scores) // 2] if scores else 0.5
    failures = [t for t in trajectories if t.f1 < failure_threshold]

    if not failures:
        return Diagnosis(failure_patterns=[], summary="No failures.", raw_analysis=""), pool

    for t in failures:
        traj_text = _format_trajectory(t, label="FAIL")

        prompt = prompts.SINGLE_ANALYZER_PROMPT.format(
            system_context=sys_desc.to_context_str(),
            source_code=sys_desc.get_source_context(),
            trajectory=traj_text,
            pool_context=pool.to_context_str(),
            past_attempts=past_attempts or "(none)",
        )

        resp = llm_call(
            prompt, model=model, max_tokens=4096,
            temperature=0, system=prompts.SINGLE_ANALYZER_SYSTEM,
        )

        new_patterns = _parse_patterns(resp.text)
        pool.add(new_patterns, example_question=t.question)

    # 取 top-N 作为本轮诊断结果
    top_patterns = pool.top_n(top_n)
    summary = "; ".join(
        f"{p['pattern']} (x{p['count']})" for p in top_patterns[:3]
    ) if top_patterns else "No patterns found."

    log.info("FailurePool: %d unique patterns from %d failures, top-%d selected",
             len(pool), len(failures), min(top_n, len(pool)))

    return Diagnosis(
        failure_patterns=top_patterns,
        summary=summary,
        raw_analysis=f"Pool size: {len(pool)}, top {top_n} selected.",
    ), pool


def meta_analyze(
    sys_desc: SystemDescription,
    l1_history: list[dict],
    model: str = "gpt-4.1-mini",
    past_attempts: str = "",
) -> Diagnosis:
    """L2 专用：直接分析 L1 运行历史，不需要重新 Observe。"""
    prompt = prompts.L2_META_ANALYZER_PROMPT.format(
        source_code=sys_desc.get_source_context(),
        l1_history=_format_l1_history(l1_history),
        past_attempts=past_attempts or "(none)",
    )

    resp = llm_call(
        prompt, model=model, max_tokens=16384,
        temperature=0, system=prompts.L2_META_ANALYZER_SYSTEM,
    )

    patterns = _parse_patterns(resp.text)
    summary = "; ".join(p.get("pattern", "") for p in patterns[:3]) if patterns else "No patterns found."

    return Diagnosis(
        failure_patterns=patterns,
        summary=summary,
        raw_analysis=resp.text,
    )


def _format_l1_history(l1_history: list[dict]) -> str:
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
            return result
    except json.JSONDecodeError:
        pass
    left = text.find("[")
    right = text.rfind("]")
    if left != -1 and right != -1:
        try:
            result = json.loads(text[left:right + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []
