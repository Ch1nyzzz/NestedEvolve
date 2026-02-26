"""统一层间协议 — 所有数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceFile:
    path: str       # 相对路径，如 "components.py"
    content: str    # 完整内容


@dataclass
class SystemDescription:
    workflow_summary: str
    component_names: list[str]
    source_files: list[SourceFile]
    source_dir: str
    baseline_score: float

    def get_source_context(self) -> str:
        """将源码文件序列化为 LLM 可读上下文。"""
        parts = []
        for sf in self.source_files:
            parts.append(f"## File: {sf.path}\n```python\n{sf.content}\n```")
        return "\n\n".join(parts)

    def to_context_str(self) -> str:
        """序列化为 LLM 可读系统摘要（不含完整源码）。"""
        file_list = ", ".join(sf.path for sf in self.source_files)
        return (
            f"Workflow: {self.workflow_summary}\n"
            f"Components: {' → '.join(self.component_names)}\n"
            f"Baseline F1: {self.baseline_score:.2f}\n"
            f"Source files: {file_list}"
        )


@dataclass
class Trajectory:
    question: str
    ground_truth: str
    prediction: str
    f1: float
    intermediate: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    failure_patterns: list[dict]    # [{pattern, root_cause, affected_component, severity, affected_file, suggested_fix}]
    summary: str
    raw_analysis: str = ""


@dataclass
class DiffBlock:
    file_path: str  # 相对路径
    search: str     # 要匹配的原文
    replace: str    # 替换内容


@dataclass
class DeltaPatch:
    diffs: list[DiffBlock]
    rationale: str
    target_patterns: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    before_score: float
    after_score: float
    accepted: bool
    patch: DeltaPatch | None = None
    details: list[dict] = field(default_factory=list)
