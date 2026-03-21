"""SkillGenerator：参考 AdaEvolve paradigm breakthrough 模式的分层 Skill 生成器。"""

from __future__ import annotations

import json
import re
from typing import Any

from ..integrations.llm import LLMClient
from .models import GeneratedSkill, SkillAxis, SkillLevel, SkillPolarity


# ============================================================
# Output schema per (axis, level)
# ============================================================

_POSITIVE_SCHEMAS: dict[tuple[SkillAxis, SkillLevel], str] = {
    (SkillAxis.REFLECTION, SkillLevel.GENERAL): """{
  "description": "one-line description",
  "idea": "core breakthrough idea (specific, actionable)",
  "actions": ["specific action 1", "specific action 2"],
  "cautions": "implementation cautions",
  "transfer_rule": "which task types can reuse this",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_reflection",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    # analyze evolution state and return dynamic guidance\\n    return {\\n        'guidance': {'principle': '...', 'idea': '...', 'actions': ['...']}\\n    }"
  }
}""",
    (SkillAxis.REFLECTION, SkillLevel.TEMPLATE): """{
  "description": "one-line description",
  "idea": "breakthrough idea for this type of problem",
  "actions": ["specific action 1", "specific action 2", "specific action 3"],
  "cautions": "implementation cautions",
  "approach_type": "algorithm/method category",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_reflection",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {'idea': '...', 'actions': ['...']}\\n    }"
  }
}""",
    (SkillAxis.REFLECTION, SkillLevel.EPHEMERAL): """{
  "description": "one-line description",
  "idea": "breakthrough for current window",
  "actions": ["execute immediately action 1", "execute immediately action 2"],
  "avoid": ["things to absolutely avoid"],
  "cautions": "cautions",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_reflection",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {'idea': '...', 'actions': ['...'], 'avoid': ['...']}\\n    }"
  }
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.GENERAL): """{
  "description": "one-line description",
  "idea": "core diagnostic insight",
  "root_cause": "root cause analysis",
  "actions": ["diagnostic action 1", "diagnostic action 2"],
  "transfer_rule": "when to reuse",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_diagnosis",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    diag = obs.get('diagnostic_packet', {})\\n    # analyze errors, population state, etc. and return dynamic diagnostic guidance\\n    return {\\n        'guidance': {'root_cause': '...', 'bottleneck': '...', 'fix_direction': '...', 'actions': ['...']}\\n    }"
  }
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.TEMPLATE): """{
  "description": "one-line description",
  "idea": "root cause of this type of stagnation/failure",
  "root_cause": "template-level root cause",
  "bottleneck": "where is the bottleneck",
  "fix_direction": "fix direction (specific to algorithm/code level)",
  "actions": ["specific diagnostic action 1", "specific diagnostic action 2"],
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_diagnosis",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    diag = obs.get('diagnostic_packet', {})\\n    return {\\n        'guidance': {'root_cause': '...', 'bottleneck': '...', 'fix_direction': '...', 'actions': ['...']}\\n    }"
  }
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.EPHEMERAL): """{
  "description": "one-line description",
  "idea": "most likely root cause and fix direction",
  "root_cause": "current root cause",
  "bottleneck": "current bottleneck",
  "fix_direction": "next step to fix (specific to code)",
  "actions": ["execute now action 1", "execute now action 2"],
  "avoid": ["things to stop doing"],
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_diagnosis",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {'root_cause': '...', 'fix_direction': '...', 'actions': ['...'], 'avoid': ['...']}\\n    }"
  }
}""",
    (SkillAxis.STRATEGY, SkillLevel.GENERAL): """{
  "description": "one-line description",
  "idea": "core search strategy idea",
  "actions": ["strategy action 1", "strategy action 2"],
  "mode": "exploration / exploitation / balanced",
  "transfer_rule": "when to reuse",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_strategy",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {'principle': '...', 'actions': ['...']},\\n        'search_policy': {\\n            'name': 'my_strategy',\\n            'parent_selection': 'softmax|ucb|elite|tournament|random',\\n            'context_mode': 'default|diverse|top_only|recent_effective',\\n            'temperature': 0.7,\\n            'mutation_strength': 0.5,\\n            'prompt_supplement': 'additional guidance for evolver'\\n        }\\n    }"
  }
}""",
    (SkillAxis.STRATEGY, SkillLevel.TEMPLATE): """{
  "description": "one-line description",
  "idea": "search strategy breakthrough for this type of stagnation",
  "actions": ["specific strategy action 1", "specific strategy action 2"],
  "approach_type": "method category",
  "cautions": "cautions",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_strategy",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {...},\\n        'search_policy': {\\n            'name': '...',\\n            'parent_selection': '...',\\n            'context_mode': '...',\\n            'temperature': ...,\\n            'mutation_strength': ...,\\n            'prompt_supplement': '...'\\n        }\\n    }"
  }
}""",
    (SkillAxis.STRATEGY, SkillLevel.EPHEMERAL): """{
  "description": "one-line description",
  "idea": "search breakthrough for current window",
  "actions": ["execute now action 1", "execute now action 2"],
  "avoid": ["approaches to avoid"],
  "emphasis": "specific guidance for code generation LLM",
  "delivery": {
    "mode": "hook",
    "hook_name": "generated_strategy",
    "hook_mode": "always",
    "hook_entrypoint": "run",
    "hook_code": "async def run(context):\\n    obs = context['obs']\\n    return {\\n        'guidance': {...},\\n        'search_policy': {\\n            'name': '...',\\n            'parent_selection': '...',\\n            'context_mode': '...',\\n            'temperature': ...,\\n            'mutation_strength': ...,\\n            'prompt_supplement': '...'\\n        }\\n    }"
  }
}""",
}

_NEGATIVE_SCHEMA = """{
  "description": "one-line description of this anti-skill",
  "failure_pattern": "recurring failure pattern (specific to code change type)",
  "forbidden_actions": ["thing to stop doing 1 (specific)", "thing to stop doing 2 (specific)"],
  "replacement_actions": ["replacement strategy 1 (specific)", "replacement strategy 2 (specific)"],
  "rationale": "why these actions should be forbidden (evidence-based)"
}"""


# ============================================================
# System prompts
# ============================================================

# ============================================================
# Abstraction ladder per skill level
# ============================================================

_ABSTRACTION_LADDER: dict[SkillLevel, tuple[str, str]] = {
    SkillLevel.GENERAL: (
        "principle-level",
        "Cross-task transferable thinking principles. "
        "Example: 'Choose statistical methods appropriate for your sample type and inference goals'. "
        "Do NOT mention specific libraries, functions, or parameters. "
        "The skill must apply to 10+ different tasks.",
    ),
    SkillLevel.TEMPLATE: (
        "method-level",
        "Specific approaches for this type of problem. "
        "Example: 'Use sample standard deviation (n-1) for inferential statistics'. "
        "Name concrete methods/algorithms but NOT exact code or library calls.",
    ),
    SkillLevel.EPHEMERAL: (
        "action-level",
        "Concrete executable instructions for the current state. "
        "Example: 'Use np.std(data, ddof=1) with the current dataset'. "
        "Include exact function calls, parameters, and code-level details.",
    ),
}


_SYSTEM_POSITIVE = (
    "You are an expert algorithm researcher and optimization strategist. "
    "Think carefully and deeply. "
    "Analyze the problem thoroughly, understand the evaluation metric "
    "by reading the evaluator code, and generate a breakthrough skill "
    "that is correct, actionable, and will actually help improve the solution. "
    "Focus on ideas that are fundamentally different from what has been tried."
)

_SYSTEM_NEGATIVE = (
    "You are an expert at diagnosing evolutionary optimization failures. "
    "Based on concrete evidence (failed edits, score regressions, error traces), "
    "identify specific code change patterns that must be forbidden, "
    "and propose concrete replacement strategies."
)


# ============================================================
# SkillGenerator
# ============================================================


class SkillGenerator:
    """LLM 驱动的 Skill 动态生成器。参考 AdaEvolve paradigm breakthrough 模式。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._id_counter = 0

    async def generate(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        level: SkillLevel | None = None,
        source_skill_ids: list[str] | None = None,
        max_retries: int = 3,
        proposal: dict | None = None,
        brainstorm: dict | None = None,
    ) -> GeneratedSkill | None:
        """生成正向 skill。失败重试，不 fallback。"""
        skill_level = level or self._infer_level(observation)
        guidance = None
        for attempt in range(max_retries):
            guidance = await self._generate_guidance(
                axis=axis,
                level=skill_level,
                observation=observation,
                proposal=proposal,
                brainstorm=brainstorm,
            )
            if guidance is not None:
                break
            print(f"    [skill-gen] attempt {attempt+1}/{max_retries} failed, retrying...")
        if guidance is None:
            print(f"    [skill-gen] all {max_retries} attempts failed, skipping")
            return None
        return self._build_skill(
            axis=axis,
            level=skill_level,
            polarity=SkillPolarity.POSITIVE,
            guidance=guidance,
            observation=observation,
            task_name=task_name,
            generation=generation,
            source_skill_ids=source_skill_ids or [],
        )

    async def generate_anti_skill(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str] | None = None,
        max_retries: int = 3,
    ) -> GeneratedSkill | None:
        """生成 anti-skill。失败重试，不 fallback。"""
        guidance = None
        for attempt in range(max_retries):
            guidance = await self._generate_guidance(
                axis=axis,
                level=SkillLevel.EPHEMERAL,
                observation=observation,
                polarity=SkillPolarity.NEGATIVE,
            )
            if guidance is not None:
                break
            print(f"    [skill-gen] anti-skill attempt {attempt+1}/{max_retries} failed, retrying...")
        if guidance is None:
            print(f"    [skill-gen] anti-skill all {max_retries} attempts failed, skipping")
            return None
        return self._build_skill(
            axis=axis,
            level=SkillLevel.EPHEMERAL,
            polarity=SkillPolarity.NEGATIVE,
            guidance=guidance,
            observation=observation,
            task_name=task_name,
            generation=generation,
            source_skill_ids=source_skill_ids or [],
        )

    async def edit(
        self,
        target_skill: GeneratedSkill,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        max_retries: int = 3,
        proposal: dict | None = None,
    ) -> GeneratedSkill | None:
        """编辑已有 skill，生成改进版本。"""
        observation = dict(observation)
        observation["_edit_target"] = target_skill
        guidance = None
        for attempt in range(max_retries):
            guidance = await self._generate_guidance(
                axis=target_skill.axis,
                level=target_skill.level,
                observation=observation,
                proposal=proposal,
            )
            if guidance is not None:
                break
            print(f"    [skill-edit] attempt {attempt+1}/{max_retries} failed, retrying...")
        if guidance is None:
            print(f"    [skill-edit] all {max_retries} attempts failed, skipping")
            return None
        edited = self._build_skill(
            axis=target_skill.axis,
            level=target_skill.level,
            polarity=SkillPolarity.POSITIVE,
            guidance=guidance,
            observation=observation,
            task_name=task_name,
            generation=generation,
            source_skill_ids=[target_skill.skill_id],
        )
        edited.parent_skill_id = target_skill.skill_id
        edited.edit_generation = target_skill.edit_generation + 1
        return edited

    async def distill_general(self, axis, obs, task_name, generation,
                              proposal=None, brainstorm=None):
        return await self.generate(axis, obs, task_name, generation,
                                   level=SkillLevel.GENERAL, proposal=proposal, brainstorm=brainstorm)

    async def compile_template(
        self, axis, obs, task_name, generation,
        source_skill_ids=None, source_skills=None,
        proposal=None, brainstorm=None,
    ):
        if source_skills:
            obs = dict(obs)
            obs["_parent_general_skills"] = [
                {"skill_id": s.skill_id, "description": s.description, "guidance": s.guidance}
                for s in source_skills
            ]
        return await self.generate(
            axis, obs, task_name, generation,
            level=SkillLevel.TEMPLATE, source_skill_ids=source_skill_ids,
            proposal=proposal, brainstorm=brainstorm,
        )

    async def instantiate_ephemeral(self, axis, obs, task_name, generation,
                                    source_skill_ids=None, proposal=None, brainstorm=None):
        return await self.generate(axis, obs, task_name, generation,
                                   level=SkillLevel.EPHEMERAL, source_skill_ids=source_skill_ids,
                                   proposal=proposal, brainstorm=brainstorm)

    # ----------------------------------------------------------
    # Core generation
    # ----------------------------------------------------------

    async def _generate_guidance(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        observation: dict[str, Any],
        polarity: SkillPolarity = SkillPolarity.POSITIVE,
        proposal: dict | None = None,
        brainstorm: dict | None = None,
    ) -> dict | None:
        system_msg = _SYSTEM_NEGATIVE if polarity == SkillPolarity.NEGATIVE else _SYSTEM_POSITIVE
        user_msg = self._build_prompt(axis, level, observation, polarity,
                                      proposal=proposal, brainstorm=brainstorm)
        try:
            response = await self.llm.generate(
                system_msg, user_msg, temperature=0.4, max_tokens=8192,
            )
            return self._parse_json(response)
        except Exception as e:
            print(f"    [skill-gen] LLM error: {e}")
            return None

    def _build_prompt(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        obs: dict[str, Any],
        polarity: SkillPolarity,
        proposal: dict | None = None,
        brainstorm: dict | None = None,
    ) -> str:
        """参考 AdaEvolve 6 步分析框架构建 prompt。"""
        parts: list[str] = []
        sys_desc = obs.get("_sys_desc")

        # === 1. Evaluator code (help LLM understand "what is good") ===
        if sys_desc and sys_desc.evaluator_source:
            eval_src = sys_desc.evaluator_source if len(sys_desc.evaluator_source) < 15000 else sys_desc.evaluator_source[:15000]
            parts.append("## Step 1: Evaluator Code (understand scoring mechanism)")
            parts.append(f"```python\n{eval_src}\n```")

        # === 2. 当前最优程序 ===
        best_code = obs.get("best_code_snippet", "")
        best_score = obs.get("best_score", 0)
        baseline = obs.get("baseline_score", 0)
        if best_code:
            parts.append(f"\n## Step 2: Current Best Program (score: {best_score:.6f}, baseline: {baseline:.6f})")
            parts.append(f"```python\n{best_code}\n```")

        # === 3. Analysis (proposer or self-analysis) ===
        if proposal:
            parts.append(f"""
## Step 3: Proposer Analysis (pre-analyzed — use this as your foundation)

**Core Problem**: {proposal.get('problem_analysis', 'N/A')}
**Why Previous Skills Failed**: {proposal.get('why_previous_failed', 'N/A')}
**Proposed Direction**: {proposal.get('proposed_approach', 'N/A')}
**Constraints**: {json.dumps(proposal.get('key_constraints', []), ensure_ascii=False)}

Build your skill based on this analysis. Do NOT re-analyze from scratch.""")
        else:
            parts.append("""
## Step 3: Analysis Framework (must complete before generating skill)

**A. Analyze the Evaluator**
- What does the scoring function reward? What does it penalize?
- What causes zero or low scores?
- What is the approximate theoretical optimal score?

**B. Analyze the Current Program**
- What algorithm/method is currently used?
- What are its strengths? Where is the bottleneck?
- How far is it from the theoretical optimum, and in what ways?

**C. Identify Breakthrough Directions**
- What fundamentally different approaches might work?
- Where is the ceiling of the current approach?
- What structural changes are needed to break through?""")

        # === 4. Evolution state (refined key metrics) ===
        dyn = obs.get("step_dynamics", {})
        diag = obs.get("diagnostic_packet", {})
        parts.append(f"""
## Step 4: Evolution State
- Task: {obs.get('task_name', '?')}
- Current best score: {best_score:.6f} (baseline: {baseline:.6f})
- Total iterations: {dyn.get('total_iters', 0)}, Total evaluations: {dyn.get('total_evals', 0)}
- Iterations since last improvement: {dyn.get('iters_since_last_improvement', 0)} iterations
- Recent stagnation ratio: {dyn.get('stagnant_ratio', 0):.0%}
- Phase: {diag.get('phase', 'early')}""")

        # === 5. Failure evidence (specific, not statistical) ===
        failure_breakdown = diag.get("failure_breakdown", {})
        if failure_breakdown:
            parts.append(f"\nFailure distribution: {dict(failure_breakdown)}")

        # Recent regressive edits (how things were broken)
        regressive = diag.get("regressive_edits", [])
        if regressive:
            parts.append("\n**Recent regressive edits (these changes worsened the score):**")
            for edit in regressive[-3:]:
                parts.append(
                    f"- score {edit.get('parent_score', 0):.4f} → {edit.get('score', 0):.4f} "
                    f"({edit.get('outcome', '?')})"
                )
                if edit.get("diff_summary"):
                    parts.append(f"  diff: {edit['diff_summary'][:200]}")

        # Recent effective edits (what changes worked)
        effective = diag.get("effective_edits", [])
        if effective:
            parts.append("\n**Recent effective edits (these changes improved the score):**")
            for edit in effective[-3:]:
                parts.append(
                    f"- score {edit.get('parent_score', 0):.4f} → {edit.get('score', 0):.4f} "
                    f"({edit.get('outcome', '?')})"
                )
                if edit.get("diff_summary"):
                    parts.append(f"  diff: {edit['diff_summary'][:200]}")

        # === Parent general skills (for template derivation) ===
        parent_generals = obs.get("_parent_general_skills", [])
        if parent_generals:
            parts.append("\n## Parent General Skills — derive your template from these")
            parts.append("Your template skill must be a task-specific concretization of one of these general principles:")
            for pg in parent_generals:
                parts.append(f"- **{pg['skill_id']}**: {pg['description']}")
                parts.append(f"  guidance: {json.dumps(pg['guidance'], ensure_ascii=False)[:500]}")

        # === 6. Previously tried skills (avoid repetition) ===
        history = obs.get("skill_history", [])
        if history:
            parts.append("\n## Step 5: Previously Tried Skills - do not repeat")
            for h in history[-8:]:
                eff = "✓" if h.get("improvement", 0) > 0 else "✗"
                parts.append(
                    f"  [{eff}] {h.get('summary', '')[:150]}"
                )

        # === Edit target (如果是编辑模式) ===
        edit_target = obs.get("_edit_target")
        if edit_target is not None:
            ev = edit_target.evidence
            idea = edit_target.guidance.get("idea") or edit_target.guidance.get("principle") or ""
            parts.append(f"""
## Edit Target Skill (MODIFY, do not create from scratch)

### What this skill does
- ID: {edit_target.skill_id}
- Axis: {edit_target.axis.value}, Level: {edit_target.level.value}
- Original idea: {idea}
- Current guidance: {json.dumps(edit_target.guidance, indent=2, ensure_ascii=False)}

### Performance analysis
- Activations: {ev.activations}, Avg improvement: {ev.avg_improvement:.6f}
- Success rate: {ev.success_rate:.0%}
- Outcome history: {ev.outcome_types[-8:]}
- Matched phases: {ev.matched_phases[-5:]}

### Edit instructions
1. Analyze WHY this skill failed in recent activations — is the idea wrong, or is the guidance incomplete?
2. Identify what to PRESERVE (parts that contributed to effective outcomes)
3. Identify what to FIX or EXTEND (gaps that caused neutral/regressive outcomes)
4. Output the COMPLETE updated guidance JSON (same schema as the original)""")

        # === Brainstorm (pre-analyzed or inline) ===
        if brainstorm and brainstorm.get("approaches"):
            approaches_text = []
            for i, a in enumerate(brainstorm["approaches"]):
                label = chr(65 + i)  # A, B, C
                approaches_text.append(
                    f"**Approach {label} ({a.get('name', '?')})**: {a.get('idea', '')}\n"
                    f"  Pros: {a.get('pros', '')} | Cons: {a.get('cons', '')}"
                )
            parts.append(f"""
## Brainstorm Result (pre-analyzed — implement the SELECTED approach)

{chr(10).join(approaches_text)}

**Selected**: {brainstorm.get('selected', '?')} — {brainstorm.get('selection_reason', '')}

Generate the skill for the SELECTED approach only.""")
        else:
            parts.append("""
## Brainstorm (MANDATORY before generating)

Before committing to a single skill, brainstorm 2-3 alternative approaches:

**Approach A**: [core idea] — Pros: ... Cons: ...
**Approach B**: [core idea] — Pros: ... Cons: ...
**Approach C** (optional): [core idea] — Pros: ...

Then select the BEST approach and explain why in 1 sentence.
Only generate the skill for the selected approach.""")

        # === Abstraction level constraint ===
        if polarity != SkillPolarity.NEGATIVE:
            abs_name, abs_desc = _ABSTRACTION_LADDER[level]
            parts.append(f"""
## Abstraction Level Constraint: {abs_name}
Your skill MUST target the **{abs_name}** abstraction:
{abs_desc}
Skills that violate this abstraction level will be rejected.""")

        # === 7. 生成指令 + output schema ===
        schema = _NEGATIVE_SCHEMA if polarity == SkillPolarity.NEGATIVE else _POSITIVE_SCHEMAS.get(
            (axis, level), _POSITIVE_SCHEMAS[(SkillAxis.STRATEGY, SkillLevel.TEMPLATE)]
        )
        mode = "anti-skill" if polarity == SkillPolarity.NEGATIVE else f"{level.value} skill"

        hook_guide = ""
        if polarity != SkillPolarity.NEGATIVE:
            hook_guide = """
**Important: Every skill MUST include the delivery.hook_code field with executable Python code.**
hook_code is an `async def run(context)` function. The context dict contains:
- obs: full observation state (diagnostic_packet, step_dynamics, best_score, error_rate, stagnation, population_summary, best_code_snippet, etc.)
- task: the current task object
- llm: LLM client (can call `await llm.generate(system, user, temperature=..., max_tokens=...)`)
- library: skill library
- population: current population (if available)

The function must return a dict with at least:
- guidance: dict of dynamic guidance to inject into the evolver prompt

Write real, dynamic Python logic that inspects obs to compute guidance — do NOT just return static strings.
Read the obs fields to adapt behavior to the current evolution state (stagnation level, error rate, phase, etc.).

**Rich context access**: The hook runs as real Python with full access to task sources:
- `context['task'].evaluator_source` — full evaluator source code (understand scoring logic deeply)
- `context['task'].full_initial_code` — complete initial program (understand starting point)
- `context['task'].initial_code` — mutable EVOLVE-BLOCK code
- `context['task'].system_prompt` — task system prompt
- `context['sys_desc'].evaluator_contract` — extracted reward/penalty/constraint keywords
- `context['sys_desc'].profile` — 8-dimension task profile (eval_cost, signal_density, search_space, etc.)
- Standard Python imports: pathlib, json, re, math, numpy, collections, etc.
- `context['llm']` — can call `await llm.generate(system, user, temperature=..., max_tokens=...)` for deep analysis

Write hooks that READ these sources dynamically for deeper analysis rather than relying solely on obs dict summaries."""
            if axis == SkillAxis.STRATEGY:
                hook_guide += """

For strategy skills, also return search_policy in the result dict:
- search_policy: dict controlling search behavior:
  - name: strategy name
  - parent_selection: "softmax" | "ucb" | "elite" | "tournament" | "random"
  - context_mode: "default" | "diverse" | "top_only" | "recent_effective"
  - temperature: float (LLM temperature)
  - mutation_strength: float (mutation strength)
  - crossover_rate: float (crossover rate)
  - prompt_supplement: str (extra text to inject into evolver prompt)"""

        parts.append(f"""
## Step 6: Generate {axis.value} {mode}

Requirements:
1. Must be grounded in the evaluator code and current program analysis above — no vague generalities
2. idea must be specific and actionable (precise to the algorithm/library/code change level)
3. Must be different from previously tried skills
4. actions must be instructions that a code generation LLM can directly execute
{hook_guide}
Output JSON (strictly follow the format):
{schema}""")

        return "\n".join(parts)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _infer_level(observation: dict[str, Any]) -> SkillLevel:
        diag = observation.get("diagnostic_packet", {})
        total_iters = observation.get("total_iters_completed", 0)
        if diag.get("phase") in ("late", "stalled") or total_iters >= 12:
            return SkillLevel.EPHEMERAL
        if total_iters >= 4 or diag.get("population", {}).get("convergence"):
            return SkillLevel.TEMPLATE
        return SkillLevel.GENERAL

    def _build_skill(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        polarity: SkillPolarity,
        guidance: dict[str, Any],
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str],
    ) -> GeneratedSkill:
        # 提取 delivery 信息（如果有）
        delivery = guidance.pop("delivery", {})
        hook_name = None
        hook_mode = None
        hook_entrypoint = None
        hook_source = None
        if isinstance(delivery, dict) and delivery.get("mode") == "hook":
            hook_source = delivery.get("hook_code")
            if isinstance(hook_source, str) and hook_source.strip():
                try:
                    compile(hook_source, "<generated-skill-hook>", "exec")
                    hook_name = delivery.get("hook_name")
                    hook_mode = delivery.get("hook_mode", "always")
                    hook_entrypoint = delivery.get("hook_entrypoint", "run")
                except SyntaxError:
                    hook_source = None

        description = guidance.pop(
            "description",
            f"[{polarity.value}:{level.value}:{axis.value}] auto-generated skill",
        )
        diag = observation.get("diagnostic_packet", {})
        self._id_counter += 1
        skill_id = (
            f"{polarity.value}_{level.value}_{axis.value}_{task_name}_{generation}_{self._id_counter}"
        )
        return GeneratedSkill(
            skill_id=skill_id,
            axis=axis,
            level=level,
            polarity=polarity,
            description=description,
            guidance=guidance,
            source_observation=self._summarize_observation(observation),
            task_origin=task_name,
            generation=generation,
            task_tags=list(diag.get("task_tags", [])),
            applicable_stages=[diag.get("phase", "early")],
            trigger_diagnostics=list(diag.get("diagnostic_tags", [])),
            source_skill_ids=source_skill_ids,
            hook_name=hook_name or (skill_id if hook_source else None),
            hook_mode=hook_mode,
            hook_entrypoint=hook_entrypoint,
            hook_source=hook_source,
        )

    @staticmethod
    def _parse_json(response: str) -> dict | None:
        m = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{[\s\S]*\}", response)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _summarize_observation(observation: dict[str, Any]) -> str:
        diag = observation.get("diagnostic_packet", {})
        return json.dumps(
            {
                "task": observation.get("task_name"),
                "best_score": observation.get("best_score"),
                "phase": diag.get("phase"),
                "diagnostic_tags": diag.get("diagnostic_tags", []),
                "failure_breakdown": diag.get("failure_breakdown", {}),
            },
            ensure_ascii=False,
        )
