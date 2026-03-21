"""Managed hook for the global target-task starter skill (diagnosis)."""

from __future__ import annotations

import json
import re
from typing import Any


_SYSTEM_PROMPT = """You are the startup analysis skill of MetaEvolve.
Your job is NOT to write code directly, but to analyze the target task, evaluator, and mutable code region before evolution begins,
so as to produce high-value early search guidance for subsequent evolvers.
Like the startup diagnostics of EvoX/AdaEvolve: first understand the task structure, then enumerate candidate method families and scoring priorities.
Output must be JSON."""


async def run(context: dict[str, Any]) -> dict[str, Any]:
    task = context["task"]
    obs = context["obs"]
    sys_desc = context["sys_desc"]
    llm = context["llm"]

    print(f"    [skill-hook] analyze target task: {task.name}")
    system_msg, user_msg = _build_messages(sys_desc, obs)
    raw_response = ""
    parsed = None
    source = "fallback"
    try:
        raw_response = await llm.generate(
            system_msg,
            user_msg,
            temperature=0.2,
            max_tokens=4096,
        )
        parsed = _parse_json(raw_response)
        if parsed:
            source = "llm"
    except Exception as e:
        raw_response = f"[hook-error] {e}"

    guidance = _normalize_guidance(parsed, task_name=task.name, sys_desc=sys_desc)
    return {
        "task_name": task.name,
        "skill_id": "system_starter_target_analysis",
        "source": source,
        "guidance": guidance,
        "raw_response": raw_response[:4000],
    }


def _build_messages(sys_desc, obs: dict[str, Any]) -> tuple[str, str]:
    diagnostics = obs.get("diagnostic_packet", {})
    user_msg = f"""{sys_desc.format_for_prompt()}

## Cold-Start Context
- Current task: {obs.get("task_name", "unknown")}
- Current best score: {obs.get("best_score", 0):.6f}
- Baseline score: {obs.get("baseline_score", 0):.6f}
- Current phase: {diagnostics.get("phase", "early")}
- Diagnostic tags: {diagnostics.get("diagnostic_tags", [])}

## Analysis Required
1. Explain the task objective, input/output format, and what the evaluator actually rewards
2. Identify hard constraints and failure modes that must be satisfied first
3. Enumerate 3-5 candidate method families worth trying early — do not give code details yet
4. Identify the most likely bottleneck at cold-start and where the evolver's first changes should focus
5. The output will be injected directly into subsequent evolve prompts — write it concretely and action-oriented

Output JSON:
{{
  "description": "One-sentence summary of the startup analysis for this task",
  "principle": "Overall cold-start principle",
  "heuristics": ["heuristic1", "heuristic2", "heuristic3"],
  "transfer_rule": "When to reuse this analysis",
  "candidate_methods": ["method family 1: why worth trying first", "method family 2: why worth trying first"],
  "evaluator_focus": ["scoring priority 1", "scoring priority 2"],
  "constraints": ["hard constraint 1", "hard constraint 2"],
  "root_cause": "Main limitation of the current baseline/initial program",
  "bottleneck": "Primary bottleneck at cold-start phase",
  "fix_direction": "Direction for the evolver's first batch of attempts",
  "intermediate_insight": "Key insight about the task structure or evaluator"
}}"""
    return _SYSTEM_PROMPT, user_msg


def _parse_json(response: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_guidance(
    raw: dict[str, Any] | None,
    *,
    task_name: str,
    sys_desc,
) -> dict[str, Any]:
    fallback = _fallback_guidance(task_name=task_name, sys_desc=sys_desc)
    if not raw:
        return fallback

    data = dict(raw)
    for key in ("heuristics", "candidate_methods", "evaluator_focus", "constraints"):
        value = data.get(key, [])
        if isinstance(value, str):
            data[key] = [value]
        elif isinstance(value, list):
            data[key] = [str(item) for item in value if str(item).strip()]
        else:
            data[key] = []

    for key in (
        "description",
        "principle",
        "transfer_rule",
        "root_cause",
        "bottleneck",
        "fix_direction",
        "intermediate_insight",
    ):
        if key in data and data[key] is not None:
            data[key] = str(data[key])

    guidance = dict(fallback)
    guidance.update({k: v for k, v in data.items() if v})
    return guidance


def _fallback_guidance(*, task_name: str, sys_desc) -> dict[str, Any]:
    contract = getattr(sys_desc, "evaluator_contract", {}) or {}
    tags = getattr(sys_desc.profile, "tags", lambda: [])()
    candidate_methods: list[str] = []
    if "geometry" in tags:
        candidate_methods.append("Geometric/constructive methods: directly exploit the spatial structure of the problem to construct high-quality solutions rather than only applying local perturbations.")
    if "continuous" in tags or "optimization" in tags:
        candidate_methods.append("Continuous parameterization + numerical optimization: explicitly parameterize key degrees of freedom, then perform systematic search or tuning.")
    if "combinatorial" in tags or "discrete" in tags:
        candidate_methods.append("Combinatorial search / greedy / local improvement: leverage discrete structure for verifiable construction and repair.")
    if "signal_processing" in tags or "algebra" in tags:
        candidate_methods.append("Analytical derivation + structured approximation: first capture the mathematical structure of the target quantity, then design approximations or constructions.")
    if not candidate_methods:
        candidate_methods = [
            "Structured construction: propose a class of interpretable candidate solution structures, then search over their parameters.",
            "Multi-paradigm early exploration: use a small number of highly diverse candidate method families to quickly determine evaluator preferences."
        ]

    reward_signals = contract.get("reward_signals", [])[:4]
    constraints = contract.get("hard_constraints", [])[:4]
    penalties = contract.get("penalties", [])[:3]
    focus = reward_signals or ["Identify the evaluator's primary scoring signal first"]
    if penalties:
        focus = focus + [f"Avoid failures/penalties: {', '.join(penalties)}"]

    return {
        "description": f"Analyze the objective, scoring mechanism, and candidate method space of {task_name} before starting evolution",
        "principle": "In the cold-start phase, prioritize building a task model: clarify what the evaluator rewards, what causes failures, and which method families are most promising — then apply directed mutation.",
        "heuristics": [
            "Decompose task objective, constraints, and mutable code region separately",
            "Ensure the first batch of candidates covers diverse method families rather than only tweaking parameters",
            "Once an effective method family is found, narrow down to local fine-tuning"
        ],
        "transfer_rule": "Applicable to the first analysis of a new task, until stronger task-specific skills have been accumulated for that task.",
        "candidate_methods": candidate_methods[:5],
        "evaluator_focus": focus,
        "constraints": constraints,
        "root_cause": "Insufficient structural understanding of the target task and evaluator, which leads the evolver to make low-value or ineffective changes.",
        "bottleneck": "Lack of structured judgment about candidate method families and scoring priorities at cold-start.",
        "fix_direction": "First perform structured exploration across candidate method families, then refine implementation around the most promising direction.",
        "intermediate_insight": f"Task profile tags: {', '.join(tags[:6])}" if tags else "The task should first extract structured optimization objectives from the system prompt and evaluator."
    }
