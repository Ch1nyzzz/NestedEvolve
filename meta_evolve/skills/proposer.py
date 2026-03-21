"""SkillProposer：分析 WHY + 发散 brainstorm，为 generator 提供结构化提案。"""

from __future__ import annotations

import json
from typing import Any

from ..integrations.llm import LLMClient
from .generator import SkillGenerator
from .models import SkillAxis, SkillLevel


# ============================================================
# System prompts
# ============================================================

_SYSTEM_PROPOSE = (
    "You are an expert diagnostic analyst for evolutionary optimization. "
    "Your job is to deeply analyze WHY the current optimization is struggling, "
    "what previous skills missed, and propose a clear direction for improvement. "
    "Be specific and evidence-based — reference concrete scores, error patterns, and failed attempts."
)

_SYSTEM_BRAINSTORM = (
    "You are a creative algorithm researcher. "
    "Given a diagnostic analysis of the current problem, brainstorm 2-3 fundamentally different approaches. "
    "Each approach must be concrete enough to implement, with honest pros and cons. "
    "Then select the single best approach and explain why."
)

# ============================================================
# Axis descriptions for proposer context
# ============================================================

_AXIS_DESC: dict[SkillAxis, str] = {
    SkillAxis.REFLECTION: (
        "reflection: algorithmic breakthrough, paradigm shift. "
        "Suitable when the current approach has hit a ceiling and needs a fundamentally different method."
    ),
    SkillAxis.DIAGNOSIS: (
        "diagnosis: error localization, bottleneck analysis, root cause identification. "
        "Suitable when errors are frequent, many zero scores, or failures follow a pattern."
    ),
    SkillAxis.STRATEGY: (
        "strategy: search policy adjustment, exploration/exploitation balance. "
        "Suitable when direction is correct but improvement is slowing or search is inefficient."
    ),
}

_LEVEL_DESC: dict[SkillLevel, str] = {
    SkillLevel.GENERAL: "general: cross-task transferable principle (abstract, reusable across 10+ tasks)",
    SkillLevel.TEMPLATE: "template: task-type specific method (concrete approach for this class of problems)",
    SkillLevel.EPHEMERAL: "ephemeral: immediate action for current state (executable, one-time use)",
}


# ============================================================
# SkillProposer
# ============================================================


class SkillProposer:
    """轻量提案器：分析 WHY → brainstorm 方案 → 输出结构化提案供 generator 实现。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ----------------------------------------------------------
    # propose: 分析当前问题 + 历史失败
    # ----------------------------------------------------------

    async def propose(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        observation: dict[str, Any],
        max_retries: int = 3,
    ) -> dict | None:
        """分析当前演化状态，输出结构化提案。

        Returns dict with keys:
            problem_analysis, why_previous_failed, proposed_approach, key_constraints
        Or None after all retries exhausted.
        """
        prompt = self._build_propose_prompt(axis, level, observation)
        for attempt in range(max_retries):
            try:
                response = await self.llm.generate(
                    _SYSTEM_PROPOSE, prompt, temperature=0.3, max_tokens=8192,
                )
                result = SkillGenerator._parse_json(response)
                if result and "problem_analysis" in result:
                    return result
                print(f"    [skill-proposer] propose attempt {attempt+1}/{max_retries}: missing required fields, retrying...")
            except Exception as e:
                print(f"    [skill-proposer] propose attempt {attempt+1}/{max_retries} error: {e}")
        print(f"    [skill-proposer] propose: all {max_retries} attempts failed")
        return None

    # ----------------------------------------------------------
    # brainstorm: 发散 2-3 方案并选择
    # ----------------------------------------------------------

    async def brainstorm(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        observation: dict[str, Any],
        proposal: dict,
        max_retries: int = 3,
    ) -> dict | None:
        """基于 proposal 发散 2-3 种方案，选出最优。

        Returns dict with keys:
            approaches (list), selected (str), selection_reason (str)
        Or None after all retries exhausted.
        """
        prompt = self._build_brainstorm_prompt(axis, level, observation, proposal)
        for attempt in range(max_retries):
            try:
                response = await self.llm.generate(
                    _SYSTEM_BRAINSTORM, prompt, temperature=0.6, max_tokens=8192,
                )
                result = SkillGenerator._parse_json(response)
                if result and "approaches" in result and "selected" in result:
                    return result
                print(f"    [skill-proposer] brainstorm attempt {attempt+1}/{max_retries}: missing required fields, retrying...")
            except Exception as e:
                print(f"    [skill-proposer] brainstorm attempt {attempt+1}/{max_retries} error: {e}")
        print(f"    [skill-proposer] brainstorm: all {max_retries} attempts failed")
        return None

    # ----------------------------------------------------------
    # Prompt builders
    # ----------------------------------------------------------

    def _build_propose_prompt(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        obs: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        diag = obs.get("diagnostic_packet", {})
        dyn = obs.get("step_dynamics", {})

        # 1. Evolution state
        parts.append(f"""## Current Evolution State
- Task: {obs.get('task_name', '?')}
- Best score: {obs.get('best_score', 0):.6f} (baseline: {obs.get('baseline_score', 0):.6f})
- Phase: {diag.get('phase', 'early')}
- Total iterations: {dyn.get('total_iters', 0)}, Total evaluations: {dyn.get('total_evals', 0)}
- Iterations since last improvement: {dyn.get('iters_since_last_improvement', 0)}
- Recent stagnation ratio: {dyn.get('stagnant_ratio', 0):.0%}
- Error rate: {obs.get('error_rate', 0):.0%}""")

        # 2. Failure evidence
        failure_breakdown = diag.get("failure_breakdown", {})
        if failure_breakdown:
            parts.append(f"\n**Failure distribution**: {dict(failure_breakdown)}")

        regressive = diag.get("regressive_edits", [])
        if regressive:
            parts.append("\n**Recent regressive edits (worsened score):**")
            for edit in regressive[-3:]:
                parts.append(
                    f"- {edit.get('parent_score', 0):.4f} → {edit.get('score', 0):.4f} "
                    f"({edit.get('outcome', '?')})"
                )
                if edit.get("diff_summary"):
                    parts.append(f"  diff: {edit['diff_summary'][:200]}")

        effective = diag.get("effective_edits", [])
        if effective:
            parts.append("\n**Recent effective edits (improved score):**")
            for edit in effective[-3:]:
                parts.append(
                    f"- {edit.get('parent_score', 0):.4f} → {edit.get('score', 0):.4f} "
                    f"({edit.get('outcome', '?')})"
                )
                if edit.get("diff_summary"):
                    parts.append(f"  diff: {edit['diff_summary'][:200]}")

        # 3. Skill history with failure analysis
        history = obs.get("skill_history", [])
        if history:
            parts.append("\n## Previously Tried Skills — analyze WHY each succeeded or failed")
            for h in history[-10:]:
                eff = "✓ effective" if h.get("improvement", 0) > 0 else "✗ ineffective"
                parts.append(f"- [{eff}] {h.get('summary', '')[:200]}")
                if h.get("failure_reason"):
                    parts.append(f"  failure reason: {h['failure_reason'][:150]}")

        # 4. Axis and level context
        parts.append(f"""
## Target Skill Specification
- Axis: {_AXIS_DESC[axis]}
- Level: {_LEVEL_DESC[level]}

Analyze the evidence above and determine:
1. What is the CORE problem preventing further improvement?
2. WHY did previous skills fail to solve it?
3. What approach would address the root cause?""")

        # 5. Output schema
        parts.append("""
## Output (JSON)
```json
{
  "problem_analysis": "What is the core problem? Be specific — reference scores, error types, patterns",
  "why_previous_failed": "Why did previous skills not solve this? Reference specific skill attempts",
  "proposed_approach": "What direction should the new skill take? Be concrete about the method/algorithm",
  "key_constraints": ["constraint 1", "constraint 2"]
}
```""")

        return "\n".join(parts)

    def _build_brainstorm_prompt(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        obs: dict[str, Any],
        proposal: dict,
    ) -> str:
        parts: list[str] = []

        # 1. Proposer analysis
        parts.append(f"""## Diagnostic Analysis (from proposer)
- **Problem**: {proposal.get('problem_analysis', 'N/A')}
- **Why previous failed**: {proposal.get('why_previous_failed', 'N/A')}
- **Proposed direction**: {proposal.get('proposed_approach', 'N/A')}
- **Constraints**: {json.dumps(proposal.get('key_constraints', []), ensure_ascii=False)}""")

        # 2. Evaluator code (abbreviated)
        sys_desc = obs.get("_sys_desc")
        if sys_desc and sys_desc.evaluator_source:
            eval_src = sys_desc.evaluator_source[:5000]
            parts.append(f"\n## Evaluator Code (scoring mechanism)\n```python\n{eval_src}\n```")

        # 3. Current best program
        best_code = obs.get("best_code_snippet", "")
        if best_code:
            parts.append(f"\n## Current Best Program (score: {obs.get('best_score', 0):.6f})\n```python\n{best_code[:3000]}\n```")

        # 4. Axis + level
        parts.append(f"""
## Target
- Axis: {axis.value} — {_AXIS_DESC[axis]}
- Level: {level.value} — {_LEVEL_DESC[level]}""")

        # 5. Instructions
        parts.append("""
## Instructions

Brainstorm 2-3 **fundamentally different** approaches to address the diagnosed problem.
Each approach must be a distinct algorithm/method, not a variation of the same idea.

For each approach:
- **name**: short identifier
- **idea**: core concept (2-3 sentences)
- **pros**: advantages
- **cons**: disadvantages or risks

Then select the BEST approach and explain why in 1-2 sentences.

## Output (JSON)
```json
{
  "approaches": [
    {"name": "approach_A", "idea": "...", "pros": "...", "cons": "..."},
    {"name": "approach_B", "idea": "...", "pros": "...", "cons": "..."}
  ],
  "selected": "approach_A",
  "selection_reason": "..."
}
```""")

        return "\n".join(parts)
