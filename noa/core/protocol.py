"""统一层间协议 — 所有数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher


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


# --------------- Failure Pool ---------------

@dataclass
class PooledPattern:
    """聚合后的 failure pattern，带出现次数。"""
    pattern: str
    root_cause: str
    affected_component: str
    severity: str
    affected_file: str
    suggested_fix: str
    count: int = 1
    examples: list[str] = field(default_factory=list)  # 触发该 pattern 的 question 摘要


class FailurePool:
    """累积 failure patterns 的池子，支持相似 pattern 合并和按频率排序。"""

    def __init__(self, similarity_threshold: float = 0.6):
        self.patterns: list[PooledPattern] = []
        self.similarity_threshold = similarity_threshold

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _find_similar(self, pattern_desc: str) -> int | None:
        """返回池中最相似 pattern 的 index，不够相似则返回 None。"""
        best_idx, best_score = None, 0.0
        for i, p in enumerate(self.patterns):
            score = self._similarity(pattern_desc, p.pattern)
            if score > best_score:
                best_idx, best_score = i, score
        if best_score >= self.similarity_threshold:
            return best_idx
        return None

    def add(self, raw_patterns: list[dict], example_question: str = "") -> None:
        """将 LLM 诊断出的 patterns 合并入池。

        如果 LLM 返回了 ``merge_to`` 字段（池中已有 pattern 的 index），
        优先按该字段合并；否则回退到字符串相似度匹配。
        """
        q_short = example_question[:80] if example_question else ""
        for raw in raw_patterns:
            desc = raw.get("pattern", "")
            merge_idx = raw.get("merge_to")  # LLM 可指定合并目标

            # 优先用 LLM 给的 merge_to
            if merge_idx is not None and 0 <= merge_idx < len(self.patterns):
                existing = self.patterns[merge_idx]
                existing.count += 1
                if q_short and q_short not in existing.examples:
                    existing.examples.append(q_short)
                # 如果新诊断 severity 更高，升级
                if _severity_rank(raw.get("severity", "low")) < _severity_rank(existing.severity):
                    existing.severity = raw["severity"]
                continue

            # 回退到字符串相似度
            sim_idx = self._find_similar(desc)
            if sim_idx is not None:
                existing = self.patterns[sim_idx]
                existing.count += 1
                if q_short and q_short not in existing.examples:
                    existing.examples.append(q_short)
                if _severity_rank(raw.get("severity", "low")) < _severity_rank(existing.severity):
                    existing.severity = raw["severity"]
            else:
                self.patterns.append(PooledPattern(
                    pattern=desc,
                    root_cause=raw.get("root_cause", ""),
                    affected_component=raw.get("affected_component", ""),
                    severity=raw.get("severity", "medium"),
                    affected_file=raw.get("affected_file", ""),
                    suggested_fix=raw.get("suggested_fix", ""),
                    count=1,
                    examples=[q_short] if q_short else [],
                ))

    def top_n(self, n: int) -> list[dict]:
        """返回出现次数最多的前 n 个 pattern（dict 格式，兼容下游）。"""
        ranked = sorted(self.patterns, key=lambda p: (-p.count, _severity_rank(p.severity)))
        results = []
        for p in ranked[:n]:
            results.append({
                "pattern": p.pattern,
                "root_cause": p.root_cause,
                "affected_component": p.affected_component,
                "severity": p.severity,
                "affected_file": p.affected_file,
                "suggested_fix": p.suggested_fix,
                "count": p.count,
                "examples": p.examples,
            })
        return results

    def to_context_str(self) -> str:
        """序列化为 LLM 可读的池摘要，供单条诊断时参考。"""
        if not self.patterns:
            return "(empty pool)"
        lines = []
        for i, p in enumerate(self.patterns):
            lines.append(
                f"[{i}] (count={p.count}) {p.pattern} | "
                f"severity={p.severity} | component={p.affected_component}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.patterns)


def _severity_rank(s: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(s, 2)
