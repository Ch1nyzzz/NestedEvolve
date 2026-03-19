"""轨迹数据结构：记录进化过程中每步的状态。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .task_profile import TaskProfile


@dataclass
class StepRecord:
    """单步进化记录。"""

    step: int
    score: float
    delta_score: float
    activated_skills: list[str] = field(default_factory=list)
    active_anti_skills: list[str] = field(default_factory=list)
    error_summary: str | None = None
    code_snippet: str | None = None  # 代码前 200 字符摘要
    parent_id: str | None = None
    parent_score: float | None = None
    parent_best: float | None = None
    outcome_type: str | None = None
    admitted_to_population: bool = False
    diff_summary: str | None = None
    diagnostic_tags: list[str] = field(default_factory=list)


@dataclass
class SelectionEvent:
    """一个 skill 选择窗口的记录。"""

    start_step: int
    n_steps: int
    activated_skills: list[str] = field(default_factory=list)
    active_anti_skills: list[str] = field(default_factory=list)
    generated_skills: list[str] = field(default_factory=list)
    generated_anti_skills: list[str] = field(default_factory=list)


@dataclass
class SegmentResult:
    """一个 segment（多步进化）的结果。"""

    best_score: float
    initial_score: float = 0.0
    trajectory: list[StepRecord] = field(default_factory=list)
    skill_context_used: bool = False
    activated_skills: list[str] = field(default_factory=list)
    active_anti_skills: list[str] = field(default_factory=list)
    selection_events: list[SelectionEvent] = field(default_factory=list)
    # 丰富数据（供 skill 生成器使用）
    population_snapshot: list[dict] | None = (
        None  # [{code, score, island_id, error}, ...]
    )
    error_details: list[dict] | None = None  # [{step, error_full, code_snippet}, ...]
    island_stats: dict | None = None  # {island_id: {size, best, mean}}
    diagnostic_packet: dict | None = None

    @property
    def improvement(self) -> float:
        return self.best_score - self.initial_score


@dataclass
class RunTrajectory:
    """完整运行轨迹（跨多个 segment）。"""

    task_name: str
    task_profile: TaskProfile
    segments: list[SegmentResult] = field(default_factory=list)
    total_improvement: float = 0.0
    skills_used: list[str] = field(default_factory=list)
    anti_skills_used: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        """压缩为 LLM 可读摘要。"""
        lines = [f"Task: {self.task_name}"]
        lines.append(f"Total improvement: {self.total_improvement:.6f}")
        lines.append(f"Skills used: {', '.join(self.skills_used) or 'none'}")
        lines.append(
            f"Anti-skills used: {', '.join(self.anti_skills_used) or 'none'}"
        )
        for i, seg in enumerate(self.segments):
            skills_str = ", ".join(seg.activated_skills) or "none"
            anti_str = ", ".join(seg.active_anti_skills) or "none"
            lines.append(
                f"  Segment {i}: {seg.initial_score:.4f} → {seg.best_score:.4f} "
                f"(Δ={seg.improvement:.4f}, skills=[{skills_str}], anti=[{anti_str}])"
            )
            if seg.selection_events:
                events_str = "; ".join(
                    f"step {event.start_step}+{event.n_steps-1}: "
                    f"skills={','.join(event.activated_skills) or 'none'} "
                    f"anti={','.join(event.active_anti_skills) or 'none'}"
                    for event in seg.selection_events[-3:]
                )
                lines.append(f"    selections: {events_str}")
            for rec in seg.trajectory[-3:]:  # 只展示最后 3 步
                lines.append(
                    f"    step {rec.step}: score={rec.score:.4f} Δ={rec.delta_score:.4f}"
                    + (
                        f" skills=[{', '.join(rec.activated_skills)}]"
                        if rec.activated_skills
                        else " skills=[none]"
                    )
                    + (
                        f" anti=[{', '.join(rec.active_anti_skills)}]"
                        if rec.active_anti_skills
                        else " anti=[none]"
                    )
                    + (f" outcome={rec.outcome_type}" if rec.outcome_type else "")
                    + (f" err={rec.error_summary}" if rec.error_summary else "")
                )
        return "\n".join(lines)


@dataclass
class OrchestratedResult:
    """SkillOrchestrator 的完整结果。"""

    trajectory: RunTrajectory
    evidence_snapshot: dict[str, dict] = field(default_factory=dict)
    final_best_score: float = 0.0
