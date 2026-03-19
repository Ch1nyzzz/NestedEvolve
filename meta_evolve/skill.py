"""Skill 系统：支持分层 skill、anti-skill 与运行时编译。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillAxis(str, Enum):
    """Skill 的内容方向。"""

    REFLECTION = "reflection"
    DIAGNOSIS = "diagnosis"
    STRATEGY = "strategy"


class SkillLevel(str, Enum):
    """Skill 的抽象层级。"""

    GENERAL = "general"
    TEMPLATE = "template"
    EPHEMERAL = "ephemeral"


class SkillPolarity(str, Enum):
    """Skill 的极性：正向 skill 或 anti-skill。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class SkillEvidence:
    """Skill 的效果追踪。"""

    activations: int = 0
    improvements: list[float] = field(default_factory=list)
    matched_tasks: list[str] = field(default_factory=list)
    matched_phases: list[str] = field(default_factory=list)
    stagnation_levels: list[int] = field(default_factory=list)
    error_rates: list[float] = field(default_factory=list)
    co_active_counts: list[int] = field(default_factory=list)

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

    def record(
        self,
        improvement: float,
        task_name: str | None = None,
        phase: str | None = None,
        stagnation: int = 0,
        error_rate: float = 0.0,
        co_active_count: int = 1,
    ):
        self.activations += 1
        self.improvements.append(improvement)
        if task_name:
            self.matched_tasks.append(task_name)
            if len(self.matched_tasks) > 30:
                self.matched_tasks = self.matched_tasks[-30:]
        if phase:
            self.matched_phases.append(phase)
            if len(self.matched_phases) > 30:
                self.matched_phases = self.matched_phases[-30:]
        self.stagnation_levels.append(stagnation)
        self.error_rates.append(error_rate)
        self.co_active_counts.append(co_active_count)
        # 保持历史窗口一致
        cap = 20
        if len(self.improvements) > cap:
            self.improvements = self.improvements[-cap:]
        if len(self.stagnation_levels) > cap:
            self.stagnation_levels = self.stagnation_levels[-cap:]
        if len(self.error_rates) > cap:
            self.error_rates = self.error_rates[-cap:]
        if len(self.co_active_counts) > cap:
            self.co_active_counts = self.co_active_counts[-cap:]


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
    level: SkillLevel = SkillLevel.EPHEMERAL
    polarity: SkillPolarity = SkillPolarity.POSITIVE
    task_tags: list[str] = field(default_factory=list)
    applicable_stages: list[str] = field(default_factory=list)
    trigger_diagnostics: list[str] = field(default_factory=list)
    source_skill_ids: list[str] = field(default_factory=list)
    evidence: SkillEvidence = field(default_factory=SkillEvidence)

    @property
    def is_negative(self) -> bool:
        return self.polarity == SkillPolarity.NEGATIVE

    def matches_context(
        self,
        task_tags: list[str] | None = None,
        phase: str | None = None,
        diagnostics: list[str] | None = None,
    ) -> float:
        """估计当前上下文和该 skill 的匹配度。"""
        score = 0.0
        if task_tags and self.task_tags:
            overlap = len(set(task_tags) & set(self.task_tags))
            score += overlap / max(len(set(self.task_tags)), 1)
        if phase and self.applicable_stages:
            if phase in self.applicable_stages:
                score += 0.75
        if diagnostics and self.trigger_diagnostics:
            overlap = len(set(diagnostics) & set(self.trigger_diagnostics))
            score += 0.5 * overlap / max(len(set(self.trigger_diagnostics)), 1)
        return score

    def format_for_prompt(
        self,
        current_phase: str | None = None,
        diagnostics: list[str] | None = None,
    ) -> str:
        """被选中后，格式化为注入 mutation prompt 的文本。"""
        prefix = (
            "### Anti-Skill"
            if self.polarity == SkillPolarity.NEGATIVE
            else f"### {self.level.value.capitalize()} Skill"
        )
        parts = [f"{prefix} · {self.axis.value}"]
        if self.description:
            parts.append(f"**摘要**: {self.description}")
        if current_phase:
            parts.append(f"**当前阶段**: {current_phase}")
        if diagnostics:
            parts.append(f"**触发诊断**: {', '.join(diagnostics[:4])}")

        g = self.guidance
        if self.polarity == SkillPolarity.NEGATIVE:
            if g.get("failure_pattern"):
                parts.append(f"**禁止模式**: {g['failure_pattern']}")
            if g.get("forbidden_actions"):
                for item in g["forbidden_actions"]:
                    parts.append(f"- 不要: {item}")
            if g.get("replacement_actions"):
                for item in g["replacement_actions"]:
                    parts.append(f"- 替代: {item}")
            if g.get("rationale"):
                parts.append(f"**原因**: {g['rationale']}")
            return "\n".join(parts)

        if self.level == SkillLevel.GENERAL:
            if g.get("principle"):
                parts.append(f"**原则**: {g['principle']}")
            for item in g.get("heuristics", []):
                parts.append(f"- 启发: {item}")
            if g.get("transfer_rule"):
                parts.append(f"**迁移条件**: {g['transfer_rule']}")
        elif self.level == SkillLevel.TEMPLATE:
            if g.get("pattern"):
                parts.append(f"**适用模式**: {g['pattern']}")
            for item in g.get("actions", []):
                parts.append(f"- 模板动作: {item}")
            if g.get("stage_hint"):
                parts.append(f"**阶段提示**: {g['stage_hint']}")
        else:
            if g.get("objective"):
                parts.append(f"**当前目标**: {g['objective']}")
            for item in g.get("actions", []):
                parts.append(f"- 立即执行: {item}")
            if g.get("avoid"):
                parts.append(f"**避免**: {'; '.join(g['avoid'])}")

        if self.axis == SkillAxis.REFLECTION and g.get("idea"):
            parts.append(f"**核心思路**: {g['idea']}")
        if self.axis == SkillAxis.DIAGNOSIS:
            for key, label in [
                ("root_cause", "根因"),
                ("bottleneck", "当前瓶颈"),
                ("fix_direction", "修复方向"),
                ("intermediate_insight", "中间洞察"),
            ]:
                if g.get(key):
                    parts.append(f"**{label}**: {g[key]}")
        if self.axis == SkillAxis.STRATEGY and g.get("mode"):
            parts.append(f"**搜索模式**: {g['mode']}")
            for adj in g.get("adjustments", []):
                parts.append(f"- 策略调整: {adj}")
        return "\n".join(parts)
