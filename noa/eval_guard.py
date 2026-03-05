"""Eval Guard — 评测前置过滤器，减少不必要的 patch 评测。

三层过滤:
1. Quality Gate: quality_score 低于阈值的 patch 直接跳过
2. Rejected Patch Dedup: 与历史 rejected patch diff 相似度过高则跳过
3. Progressive Sampling: Stage 3 分批采样，提前决策
"""

from __future__ import annotations

from difflib import SequenceMatcher

from noa.core.protocol import DeltaPatch, DiffBlock, EvalResult


# ── Quality Gate ──────────────────────────────────────────────


_DEFAULT_QUALITY_THRESHOLD = 0.4


def quality_gate(
    patch: DeltaPatch,
    baseline_score: float,
    threshold: float = _DEFAULT_QUALITY_THRESHOLD,
) -> EvalResult | None:
    """如果 patch quality_score 低于阈值，直接返回 REJECTED EvalResult。

    Returns None 表示通过 gate，需要继续评测。
    """
    if patch.quality_score >= threshold:
        return None

    return EvalResult(
        before_score=baseline_score,
        after_score=baseline_score,
        accepted=False,
        patch=patch,
        artifacts={
            "stage": 0,
            "reason": f"Quality gate: {patch.quality_score:.2f} < {threshold:.2f}",
            "quality_score": patch.quality_score,
        },
        delta=0.0,
        failure_reason="quality_gate",
    )


# ── Rejected Patch Dedup ─────────────────────────────────────


def _diff_signature(diffs: list[DiffBlock]) -> str:
    """将 patch 的 diffs 压缩为可比较的签名字符串。"""
    parts = []
    for d in sorted(diffs, key=lambda x: x.file_path):
        parts.append(f"{d.file_path}::{d.search}::{d.replace}")
    return "\n".join(parts)


class RejectedPatchTracker:
    """跟踪已 rejected 的 patch，检测新 patch 是否与之过于相似。"""

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._rejected_signatures: list[tuple[str, str]] = []  # (signature, reason)

    def record_rejection(self, patch: DeltaPatch, reason: str = "") -> None:
        sig = _diff_signature(patch.diffs)
        if sig:
            self._rejected_signatures.append((sig, reason))

    def is_duplicate(self, patch: DeltaPatch) -> tuple[bool, str]:
        """检查新 patch 是否与历史 rejected patch 相似。

        Returns (is_dup, reason_str).
        """
        if not self._rejected_signatures:
            return False, ""

        sig = _diff_signature(patch.diffs)
        if not sig:
            return False, ""

        for prev_sig, prev_reason in self._rejected_signatures:
            similarity = SequenceMatcher(None, sig, prev_sig).ratio()
            if similarity >= self.similarity_threshold:
                return True, (
                    f"Similar to previously rejected patch "
                    f"(similarity={similarity:.2f}, prev_reason={prev_reason})"
                )

        return False, ""

    def __len__(self) -> int:
        return len(self._rejected_signatures)


def dedup_check(
    patch: DeltaPatch,
    tracker: RejectedPatchTracker,
    baseline_score: float,
) -> EvalResult | None:
    """与历史 rejected patch 比较，相似则直接返回 REJECTED。

    Returns None 表示通过检查。
    """
    is_dup, reason = tracker.is_duplicate(patch)
    if not is_dup:
        return None

    return EvalResult(
        before_score=baseline_score,
        after_score=baseline_score,
        accepted=False,
        patch=patch,
        artifacts={
            "stage": 0,
            "reason": f"Dedup gate: {reason}",
        },
        delta=0.0,
        failure_reason="dedup_rejected",
    )


# ── Progressive Sampling ─────────────────────────────────────


_PROGRESSIVE_FIRST_BATCH = 6
_PROGRESSIVE_CLEAR_WIN = 0.03   # after_score - baseline > 阈值 → 直接接受
_PROGRESSIVE_CLEAR_LOSS = -0.02  # after_score - baseline < 阈值 → 直接拒绝


def should_stop_early(
    batch_score: float,
    baseline_score: float,
    batch_size: int,
    total_samples: int,
) -> str | None:
    """判断是否可以根据部分采样结果提前决策。

    Returns:
        "accept" — 明确优于 baseline，无需继续
        "reject" — 明确劣于 baseline，无需继续
        None — 结果不确定，需继续采样
    """
    delta = batch_score - baseline_score

    # 采样比例越高，置信度越高，阈值可以更宽松
    ratio = batch_size / max(total_samples, 1)

    if ratio >= 0.8:
        # 已经采了足够多，任何差异都可信
        return "accept" if delta > 0 else "reject"

    if delta > _PROGRESSIVE_CLEAR_WIN:
        return "accept"

    if delta < _PROGRESSIVE_CLEAR_LOSS:
        return "reject"

    return None
