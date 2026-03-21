"""SystemDescription: tells the skill generator what system is being optimized."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import Task
from .profiles import TaskProfile


@dataclass
class SystemDescription:
    task_name: str
    domain_description: str  # Task.system_prompt
    profile: TaskProfile  # 8-dimension task profile
    initial_code_preview: str  # first 800 chars of the EVOLVE-BLOCK
    mutable_scope: str  # description of the mutable scope
    evaluator_source: str = ""  # evaluator source code
    evaluator_contract: dict = field(default_factory=dict)

    def format_for_prompt(self) -> str:
        profile = self.profile
        tags = ", ".join(profile.tags())
        lines = [
            "## Target System",
            f"**Task**: {self.task_name}",
            f"**Profile**: {tags}",
            f"**Mutable Scope**: {self.mutable_scope}",
            "",
            "**Domain Description**:",
            self.domain_description[:600],
            "",
            "**Initial Code Preview**:",
            f"```python\n{self.initial_code_preview}\n```",
        ]
        contract = self.evaluator_contract
        if contract:
            lines.append("")
            lines.append("**Evaluator Contract**:")
            if contract.get("reward_signals"):
                lines.append(
                    "- Reward signals: " + ", ".join(contract["reward_signals"][:6])
                )
            if contract.get("penalties"):
                lines.append("- Penalties: " + ", ".join(contract["penalties"][:6]))
            if contract.get("hard_constraints"):
                lines.append(
                    "- Hard constraints: " + ", ".join(contract["hard_constraints"][:6])
                )
        if self.evaluator_source:
            lines.append("")
            lines.append("**Scoring Function Preview**:")
            src = self.evaluator_source if len(self.evaluator_source) < 15000 else self.evaluator_source[:15000]
            lines.append(f"```python\n{src}\n```")
        return "\n".join(lines)


def _extract_evaluator_contract(evaluator_source: str) -> dict:
    """Extract simple reward/penalty/constraint hints from the evaluator source code."""
    if not evaluator_source:
        return {}

    reward_signals: list[str] = []
    penalties: list[str] = []
    constraints: list[str] = []

    keywords = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]{2,}",
        evaluator_source,
    )
    for token in keywords:
        low = token.lower()
        if any(k in low for k in ("score", "reward", "fitness", "accuracy", "combined")):
            reward_signals.append(token)
        if any(k in low for k in ("penalty", "error", "cost", "timeout", "fail")):
            penalties.append(token)
        if any(k in low for k in ("constraint", "valid", "limit", "bound", "must")):
            constraints.append(token)

    def _uniq(items: list[str]) -> list[str]:
        out: list[str] = []
        for item in items:
            if item not in out:
                out.append(item)
        return out

    return {
        "reward_signals": _uniq(reward_signals)[:12],
        "penalties": _uniq(penalties)[:12],
        "hard_constraints": _uniq(constraints)[:12],
    }


def build_system_description(task: Task, profile: TaskProfile) -> SystemDescription:
    return SystemDescription(
        task_name=task.name,
        domain_description=task.system_prompt,
        profile=profile,
        initial_code_preview=task.initial_code[:800],
        mutable_scope="Python functions within the EVOLVE-BLOCK",
        evaluator_source=task.evaluator_source,
        evaluator_contract=_extract_evaluator_contract(task.evaluator_source),
    )
