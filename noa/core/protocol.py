"""统一层间协议 — 所有数据结构定义。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class SourceFile:
    path: str  # 相对路径，如 "components.py"
    content: str  # 完整内容


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
            lang = "json" if sf.path.endswith(".json") else "python"
            parts.append(f"## File: {sf.path}\n```{lang}\n{sf.content}\n```")
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
    error: str | None = None
    trace_source: str = "fresh"  # fresh | reused | replayed
    intermediate_complete: bool = False


@dataclass
class Diagnosis:
    failure_patterns: list[
        dict
    ]  # [{pattern, root_cause, affected_component, severity, affected_file, suggested_fix}]
    summary: str
    raw_analysis: str = ""


@dataclass
class DiffBlock:
    file_path: str  # 相对路径
    search: str  # 要匹配的原文
    replace: str  # 替换内容


@dataclass
class DeltaPatch:
    diffs: list[DiffBlock]
    rationale: str
    target_patterns: list[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class EvalResult:
    before_score: float
    after_score: float
    accepted: bool
    patch: DeltaPatch | None = None
    details: list[dict] = field(default_factory=list)
    error: str | None = None
    artifacts: dict = field(default_factory=dict)
    delta: float = 0.0
    failure_reason: str | None = None
    feedback: dict = field(default_factory=dict)  # LLM结构化反馈


# --------------- Layer Context ---------------


@dataclass
class LayerContext:
    """层级上下文 — 控制递归深度、spawn 预算、读写权限。"""

    layer_id: str  # "L1", "L2", ... 唯一标识
    level: int  # 1=L1, 2=L2, ...
    writable_root: str  # 当前层唯一可写的根目录（绝对路径）
    readable_roots: list[str]  # 当前层可读的所有目录
    parent_history: list[dict]  # 下层运行历史
    parent_summary: str = ""
    max_depth: int = 3  # 递归深度上限
    max_spawn_calls: int = 2  # 每层最大 spawn 次数
    spawn_calls_used: int = 0  # 已 spawn 次数

    def can_spawn_sublayer(self) -> bool:
        return (
            self.level < self.max_depth and self.spawn_calls_used < self.max_spawn_calls
        )

    def check_write_permission(self, abs_path: str) -> bool:
        """检查路径是否在 writable_root 内。规范化后校验，防 ../ 穿越。"""
        normalized = os.path.realpath(abs_path)
        writable = os.path.realpath(self.writable_root)
        return normalized == writable or normalized.startswith(writable + os.sep)

    def check_read_permission(self, abs_path: str) -> bool:
        """检查路径是否在任一 readable_root 内。"""
        normalized = os.path.realpath(abs_path)
        for root in self.readable_roots:
            r = os.path.realpath(root)
            if normalized == r or normalized.startswith(r + os.sep):
                return True
        return False

    def to_prompt_context(self) -> str:
        """L1 返回 ""，L2+ 返回含历史和只读源码的文本。"""
        if self.level <= 1:
            return ""
        parts = [f"## Layer Context ({self.layer_id}, level={self.level})"]

        # 角色约束：明确告诉 L2+ 它的优化目标是框架代码，而非目标系统
        writable_name = os.path.basename(self.writable_root.rstrip("/"))
        parts.append(
            f"### Role Constraints\n"
            f"You are a META-OPTIMIZER at {self.layer_id}. "
            f"Your optimization target is the optimizer framework code in `{writable_name}/`, NOT the end-user target system.\n"
            f"- You can ONLY modify files under `{writable_name}/`\n"
            f"- The trajectories below show how the lower-layer optimizer performed. "
            f"Diagnose WHY the optimizer's strategy failed, not what the target system did wrong.\n"
            f"- If you see component names like 'QuestionRewriter', 'Retriever' etc. in trajectory data, "
            f"those are the TARGET SYSTEM's components that the lower-layer was trying to optimize. "
            f"Do NOT try to modify those files — they are outside your scope.\n"
            f"- Focus on: analyzer prompts, optimizer prompts, evaluation logic, planner guardrails, "
            f"observation strategies, and other optimizer framework code."
        )

        if self.parent_summary:
            parts.append(f"### Parent Summary\n{self.parent_summary}")
        if self.parent_history:
            parts.append("### Parent Run History")
            for h in self.parent_history[-20:]:
                status = "ACCEPTED" if h.get("accepted") else "REJECTED"
                before = h.get("before", 0)
                after = h.get("after", 0)
                header = f"Iteration {h.get('iteration', '?')} [{status}] {before:.2f} → {after:.2f}"
                parts.append(header)
                parts.append(f"  Diagnosis: {h.get('diagnosis', 'N/A')}")
                parts.append(f"  Rationale: {h.get('rationale', 'N/A')}")
                diffs = h.get("diffs", [])
                for d in diffs:
                    if isinstance(d, dict):
                        fp = d.get("file_path", "?")
                        search = d.get("search", "")[:200]
                        replace = d.get("replace", "")[:200]
                    else:
                        fp = getattr(d, "file_path", "?")
                        search = getattr(d, "search", "")[:200]
                        replace = getattr(d, "replace", "")[:200]
                    parts.append(f"  File: {fp} SEARCH: {search} REPLACE: {replace}")
        parts.append(
            f"Spawn budget: {self.spawn_calls_used}/{self.max_spawn_calls}, max_depth={self.max_depth}"
        )
        return "\n".join(parts)


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
                if _severity_rank(raw.get("severity", "low")) < _severity_rank(
                    existing.severity
                ):
                    existing.severity = raw["severity"]
                continue

            # 回退到字符串相似度
            sim_idx = self._find_similar(desc)
            if sim_idx is not None:
                existing = self.patterns[sim_idx]
                existing.count += 1
                if q_short and q_short not in existing.examples:
                    existing.examples.append(q_short)
                if _severity_rank(raw.get("severity", "low")) < _severity_rank(
                    existing.severity
                ):
                    existing.severity = raw["severity"]
            else:
                self.patterns.append(
                    PooledPattern(
                        pattern=desc,
                        root_cause=raw.get("root_cause", ""),
                        affected_component=raw.get("affected_component", ""),
                        severity=raw.get("severity", "medium"),
                        affected_file=raw.get("affected_file", ""),
                        suggested_fix=raw.get("suggested_fix", ""),
                        count=1,
                        examples=[q_short] if q_short else [],
                    )
                )

    def consolidate(self) -> int:
        """合并池中相似度 >= threshold 的 pattern 对，返回合并次数。

        用于并行诊断后兜底去重：并行调用无法使用 merge_to，
        可能导致同一 pattern 以不同措辞被分别添加。
        此方法做一轮贪心合并，将相似条目合并到 count 更高的一方。
        """
        merged_count = 0
        i = 0
        while i < len(self.patterns):
            j = i + 1
            while j < len(self.patterns):
                if (
                    self._similarity(self.patterns[i].pattern, self.patterns[j].pattern)
                    >= self.similarity_threshold
                ):
                    # 把 j 合并到 i
                    self.patterns[i].count += self.patterns[j].count
                    for ex in self.patterns[j].examples:
                        if ex not in self.patterns[i].examples:
                            self.patterns[i].examples.append(ex)
                    if _severity_rank(self.patterns[j].severity) < _severity_rank(
                        self.patterns[i].severity
                    ):
                        self.patterns[i].severity = self.patterns[j].severity
                    # 保留更长/更详细的 root_cause 和 suggested_fix
                    if len(self.patterns[j].root_cause) > len(
                        self.patterns[i].root_cause
                    ):
                        self.patterns[i].root_cause = self.patterns[j].root_cause
                    if len(self.patterns[j].suggested_fix) > len(
                        self.patterns[i].suggested_fix
                    ):
                        self.patterns[i].suggested_fix = self.patterns[j].suggested_fix
                    self.patterns.pop(j)
                    merged_count += 1
                else:
                    j += 1
            i += 1
        return merged_count

    def top_n(self, n: int) -> list[dict]:
        """返回出现次数最多的前 n 个 pattern（dict 格式，兼容下游）。"""
        ranked = sorted(
            self.patterns, key=lambda p: (-p.count, _severity_rank(p.severity))
        )
        results = []
        for p in ranked[:n]:
            results.append(
                {
                    "pattern": p.pattern,
                    "root_cause": p.root_cause,
                    "affected_component": p.affected_component,
                    "severity": p.severity,
                    "affected_file": p.affected_file,
                    "suggested_fix": p.suggested_fix,
                    "count": p.count,
                    "examples": p.examples,
                }
            )
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
