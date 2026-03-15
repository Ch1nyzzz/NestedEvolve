"""统一层间协议 — 所有数据结构定义。"""

from __future__ import annotations

import json
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

    def get_source_index(self) -> str:
        """只返回文件索引（路径+行数+首行摘要），不含完整源码。供 L2 按需读取。"""
        lines = []
        for sf in self.source_files:
            n_lines = sf.content.count("\n") + 1
            # 提取文件开头的 docstring 或首个 class/def 作为摘要
            first_lines = sf.content.strip().split("\n")[:3]
            summary = " | ".join(ln.strip() for ln in first_lines if ln.strip())[:120]
            lines.append(f"- {sf.path} ({n_lines} lines): {summary}")
        return "\n".join(lines)

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
class PatchOp:
    """单个结构化 patch 操作。"""

    op: str  # "update", "create", "delete", "insert_after", "insert_before", "overwrite"
    file_path: str
    # update: search + replace
    search: str = ""
    replace: str = ""
    occurrence: int = 1
    must_be_unique: bool = True
    context_before: str = ""
    context_after: str = ""
    # create/delete: 整文件操作
    content: str = ""
    # insert_after/insert_before: 锚点插入
    anchor: str = ""
    anchor_occurrence: int = 1
    anchor_must_be_unique: bool = True
    new_lines: str = ""


@dataclass
class PatchValidationError:
    """结构化的 patch 验证错误。"""

    op_index: int
    code: str  # file_not_found, search_not_found, search_not_unique, etc.
    file_path: str
    message: str


@dataclass
class StructuredPatch:
    """一组 PatchOp + 元数据。事务型接口。"""

    ops: list[PatchOp]
    rationale: str
    target_patterns: list[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class EvalResult:
    before_score: float
    after_score: float
    accepted: bool
    patch: DeltaPatch | StructuredPatch | None = None
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
    parent_context_files: list[str] = field(default_factory=list)
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
            f"- You can modify ANY code in `{writable_name}/` — agent logic, candidate selection, "
            f"evaluation orchestration, prompts, config, utilities."
        )

        if self.parent_summary:
            parts.append(f"### Parent Summary\n{self.parent_summary}")
        if self.parent_context_files:
            parts.append("### Full Parent Context Files")
            parts.extend(
                f"- Read `{path}` for the complete untruncated parent context."
                for path in self.parent_context_files
            )
        if self.parent_history:
            parts.append("### Parent (L1) Recent Run Snapshot")
            for h in self.parent_history[-30:]:
                action = h.get("action", h.get("iteration", "?"))
                step = h.get("step", "?")
                if action == "observe":
                    parts.append(
                        f"  [Step {step}] OBSERVE: {h.get('n_trajectories', 0)} trajectories, "
                        f"mean_f1={h.get('mean_f1', 0)}"
                    )
                elif action == "analyze":
                    patterns = h.get("patterns", [])
                    parts.append(f"  [Step {step}] ANALYZE: {len(patterns)} patterns")
                    for p in patterns[:3]:
                        if isinstance(p, dict):
                            parts.append(
                                f"    - [{p.get('severity', '?')}] {p.get('pattern', '')[:100]}"
                                f" (file: {p.get('affected_file', 'N/A')})"
                            )
                        else:
                            parts.append(f"    - {str(p)[:120]}")
                elif action == "checkpoint_candidate":
                    ops = h.get("ops", [])
                    n_ops = len(ops) if isinstance(ops, list) else 0
                    files = (
                        list(
                            {
                                op.get("file_path", "?")
                                for op in ops[:5]
                                if isinstance(op, dict)
                            }
                        )
                        if isinstance(ops, list)
                        else []
                    )
                    parts.append(
                        f"  [Step {step}] CHECKPOINT: {h.get('label', '?')} "
                        f"ok={h.get('ok', '?')} ops={n_ops} files={files} "
                        f"rationale={h.get('rationale', '')[:100]}"
                    )
                elif action == "eval_candidate":
                    status = "ACCEPTED" if h.get("accepted") else "REJECTED"
                    parts.append(
                        f"  [Step {step}] EVAL: {h.get('label', '?')} [{status}] "
                        f"{h.get('before_score', 0):.2f} → {h.get('after_score', 0):.2f}"
                    )
                    if h.get("error"):
                        parts.append(f"    error: {h['error'][:200]}")
                elif action == "accept_candidate":
                    parts.append(
                        f"  [Step {step}] ACCEPT: {h.get('label', '?')} score={h.get('score', 0):.2f}"
                    )
                elif action == "reject_candidate":
                    parts.append(
                        f"  [Step {step}] REJECT: {h.get('label', '?')} "
                        f"reason={h.get('reason', '')[:150]}"
                    )
                elif action == "validation_eval":
                    val_status = "COMMITTED" if h.get("accepted") else "REJECTED"
                    parts.append(
                        f"  [Step {step}] VAL_EVAL: {h.get('label', '?')} [{val_status}] "
                        f"val_score={h.get('val_score', 0):.2f} "
                        f"(baseline={h.get('baseline_score', 0):.2f}, train={h.get('train_score', 0):.2f})"
                    )
                elif action in ("final_eval_commit", "final_eval_no_commit"):
                    parts.append(
                        f"  [Step {step}] FINAL_EVAL: {h.get('label', 'none')} "
                        f"test_score={h.get('test_score', 0):.2f} "
                        f"baseline={h.get('baseline_score', 0):.2f}"
                    )
                elif action == "test_eval_candidate":
                    parts.append(
                        f"  [Step {step}] TEST_EVAL: {h.get('label', '?')} "
                        f"test_score={h.get('test_score', 0):.2f} "
                        f"train_score={h.get('train_score', 0):.2f}"
                    )
                else:
                    # fallback: 旧格式或其他 action
                    parts.append(f"  [{action}] {json.dumps(h, default=str)[:200]}")

            # --- Decision quality summary ---
            all_evals = [
                h for h in self.parent_history if h.get("action") == "eval_candidate"
            ]
            val_evals = [
                h for h in self.parent_history if h.get("action") == "validation_eval"
            ]
            final_evals = [
                h
                for h in self.parent_history
                if h.get("action") in ("final_eval_commit", "test_eval_candidate")
            ]
            if all_evals:
                n_zero = sum(1 for e in all_evals if e.get("after_score", 0) == 0)
                best_train = max(
                    (e.get("after_score", 0) for e in all_evals), default=0
                )
                best_val = (
                    max((e.get("val_score", 0) for e in val_evals), default=0)
                    if val_evals
                    else None
                )
                best_test = (
                    max((e.get("test_score", 0) for e in final_evals), default=0)
                    if final_evals
                    else None
                )
                parts.append("\n### L1 Decision Quality Summary")
                parts.append(
                    f"  Total evals: {len(all_evals)}, zero-score: {n_zero} ({100*n_zero//max(len(all_evals),1)}%)"
                )
                parts.append(f"  Best train score: {best_train:.2f}")
                if best_val is not None:
                    parts.append(f"  Best val score: {best_val:.2f}")
                if best_test is not None:
                    parts.append(f"  Best test score: {best_test:.2f}")
                if best_val and best_test and best_val > best_test + 5:
                    parts.append(
                        f"  ⚠ Val→Test gap: {best_val:.2f} → {best_test:.2f} "
                        f"(delta={best_val - best_test:+.2f}). "
                        f"Investigate: are good candidates being lost in the selection pipeline?"
                    )
                if n_zero > len(all_evals) * 0.4:
                    parts.append(
                        f"  ⚠ High zero-score rate ({n_zero}/{len(all_evals)}). "
                        f"Investigate: are patches frequently breaking the target system?"
                    )
                # Detect repeated errors
                from collections import Counter

                zero_errors = [
                    e.get("error", "")
                    for e in all_evals
                    if e.get("after_score", 0) == 0 and e.get("error")
                ]
                if zero_errors:
                    error_counts = Counter(zero_errors).most_common(3)
                    repeated = [(err, cnt) for err, cnt in error_counts if cnt >= 2]
                    if repeated:
                        parts.append("  ⚠ Repeated errors across rounds:")
                        for err, cnt in repeated:
                            parts.append(f"    - {cnt}x: {err[:120]}")

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


# --------------- Optimization Budget ---------------


@dataclass
class OptimizationBudget:
    """优化预算 — 控制步数、LLM 调用、评估次数。"""

    max_steps: int = 20
    max_llm_calls: int = 999999
    max_no_improve_steps: int = 5
    target_delta: float = float("inf")
    max_spawn_calls: int = 2
    step_count: int = 0
    llm_calls_used: int = 0
    evals_used: int = 0
    spawn_calls_used: int = 0
    no_improve_count: int = 0

    def reached_limit(self) -> bool:
        return self.step_count >= self.max_steps

    def stagnation_detected(self) -> bool:
        return self.no_improve_count >= self.max_no_improve_steps

    def to_summary(self) -> dict:
        return {
            "steps": f"{self.step_count}/{self.max_steps}",
            "llm_calls": f"{self.llm_calls_used}/{self.max_llm_calls}",
            "evals": self.evals_used,
            "spawn_calls": f"{self.spawn_calls_used}/{self.max_spawn_calls}",
            "max_no_improve_steps": self.max_no_improve_steps,
            "no_improve_count": self.no_improve_count,
            "target_delta": self.target_delta,
        }
