"""Skill 系统：Axis 是生成方向，Skill 是运行时 LLM 动态生成的策略。

激活机制：不用硬编码 trigger，而是把所有 skill 的 one-liner description
列给 orchestrator LLM，由它根据当前状态选择激活哪些。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillAxis(str, Enum):
    """Skill 的 3 个生成方向。仅作分类。"""

    REFLECTION = "reflection"
    """反思 + 范式突破。"""

    DIAGNOSIS = "diagnosis"
    """广义诊断：失败分析、中间输出分析、瓶颈识别、改进方向。"""

    STRATEGY = "strategy"
    """进化策略自适应：搜索模式、变异控制、种群参数。"""


@dataclass
class GeneratedSkill:
    """一个由 LLM 在运行时生成的具体策略。"""

    skill_id: str
    axis: SkillAxis
    description: str  # 一句话描述，用于 orchestrator 选择
    guidance: dict[str, Any]  # 完整指导内容（被选中时注入 prompt）
    source_observation: str
    task_origin: str
    generation: int
    evidence: SkillEvidence = field(default_factory=lambda: SkillEvidence())

    def format_for_prompt(self) -> str:
        """被选中后，格式化为注入 mutation prompt 的文本。"""
        parts = []
        g = self.guidance

        if self.axis == SkillAxis.REFLECTION:
            parts.append("### 突破思路")
            if g.get("idea"):
                parts.append(f"**核心思路**: {g['idea']}")
            if g.get("description"):
                parts.append(g["description"])
            if g.get("cautions"):
                parts.append(f"**注意**: {g['cautions']}")

        elif self.axis == SkillAxis.DIAGNOSIS:
            parts.append("### 诊断分析")
            for key, label in [
                ("failure_pattern", "问题模式"),
                ("root_cause", "根因"),
                ("bottleneck", "当前瓶颈"),
                ("fix_direction", "修复方向"),
                ("intermediate_insight", "中间结果洞察"),
            ]:
                if g.get(key):
                    parts.append(f"**{label}**: {g[key]}")
            if g.get("avoid"):
                parts.append(f"**避免**: {'; '.join(g['avoid'])}")

        elif self.axis == SkillAxis.STRATEGY:
            mode = g.get("mode", "balanced")
            parts.append(f"### 进化策略: {mode.upper()}")
            if g.get("emphasis"):
                parts.append(g["emphasis"])
            if g.get("adjustments"):
                for adj in g["adjustments"]:
                    parts.append(f"- {adj}")

        return "\n".join(parts)


@dataclass
class SkillEvidence:
    """Skill 的效果追踪。"""

    activations: int = 0
    improvements: list[float] = field(default_factory=list)

    @property
    def avg_improvement(self) -> float:
        return (
            sum(self.improvements) / len(self.improvements)
            if self.improvements
            else 0.0
        )

    @property
    def success_rate(self) -> float:
        if not self.improvements:
            return 0.0
        return sum(1 for x in self.improvements if x > 1e-6) / len(self.improvements)

    @property
    def confidence(self) -> float:
        return min(1.0, self.activations / 10.0)

    def record(self, improvement: float):
        self.activations += 1
        self.improvements.append(improvement)
        if len(self.improvements) > 20:
            self.improvements = self.improvements[-20:]
