"""SystemDescription：让 skill 生成器知道在优化什么系统。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .task_adapter import Task
from .task_profile import TaskProfile


@dataclass
class SystemDescription:
    task_name: str
    domain_description: str  # Task.system_prompt
    profile: TaskProfile  # 8 维任务画像
    initial_code_preview: str  # EVOLVE-BLOCK 前 800 字符
    mutable_scope: str  # 可变范围描述
    evaluator_source: str = ""  # 评估器源码
    evaluator_contract: dict = field(default_factory=dict)

    def format_for_prompt(self) -> str:
        profile = self.profile
        tags = ", ".join(profile.tags())
        lines = [
            "## 目标系统",
            f"**任务**: {self.task_name}",
            f"**画像**: {tags}",
            f"**可变范围**: {self.mutable_scope}",
            "",
            "**领域描述**:",
            self.domain_description[:600],
            "",
            "**初始代码预览**:",
            f"```python\n{self.initial_code_preview}\n```",
        ]
        contract = self.evaluator_contract
        if contract:
            lines.append("")
            lines.append("**评估器契约**:")
            if contract.get("reward_signals"):
                lines.append(
                    "- 奖励信号: " + ", ".join(contract["reward_signals"][:6])
                )
            if contract.get("penalties"):
                lines.append("- 惩罚项: " + ", ".join(contract["penalties"][:6]))
            if contract.get("hard_constraints"):
                lines.append(
                    "- 硬约束: " + ", ".join(contract["hard_constraints"][:6])
                )
        if self.evaluator_source:
            lines.append("")
            lines.append("**评分函数预览**:")
            lines.append(f"```python\n{self.evaluator_source[:2000]}\n```")
        return "\n".join(lines)


def _extract_evaluator_contract(evaluator_source: str) -> dict:
    """从评估器源码中提炼简易 reward/penalty/constraint 线索。"""
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
        mutable_scope="EVOLVE-BLOCK 内的 Python 函数",
        evaluator_source=task.evaluator_source,
        evaluator_contract=_extract_evaluator_contract(task.evaluator_source),
    )
