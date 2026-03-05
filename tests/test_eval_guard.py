"""Tests for noa.eval_guard — quality gate and dedup."""

import pytest

from noa.core.protocol import DeltaPatch, DiffBlock
from noa.eval_guard import (
    RejectedPatchTracker,
    dedup_check,
    quality_gate,
)


# ── helpers ──────────────────────────────────────────────────


def _make_patch(quality: float = 0.8, file_path: str = "comp.py",
                search: str = "old", replace: str = "new") -> DeltaPatch:
    return DeltaPatch(
        diffs=[DiffBlock(file_path=file_path, search=search, replace=replace)],
        rationale="test",
        quality_score=quality,
    )


# ── Quality Gate ─────────────────────────────────────────────


class TestQualityGate:
    def test_passes_high_quality(self):
        patch = _make_patch(quality=0.7)
        result = quality_gate(patch, baseline_score=50.0)
        assert result is None  # passes through

    def test_rejects_low_quality(self):
        patch = _make_patch(quality=0.2)
        result = quality_gate(patch, baseline_score=50.0)
        assert result is not None
        assert result.accepted is False
        assert result.failure_reason == "quality_gate"

    def test_boundary_at_threshold(self):
        patch = _make_patch(quality=0.4)
        result = quality_gate(patch, baseline_score=50.0, threshold=0.4)
        assert result is None  # exactly at threshold → passes

    def test_just_below_threshold(self):
        patch = _make_patch(quality=0.39)
        result = quality_gate(patch, baseline_score=50.0, threshold=0.4)
        assert result is not None


# ── Rejected Patch Dedup ─────────────────────────────────────


class TestRejectedPatchTracker:
    def test_empty_tracker_no_dup(self):
        tracker = RejectedPatchTracker()
        patch = _make_patch()
        is_dup, _ = tracker.is_duplicate(patch)
        assert is_dup is False

    def test_exact_duplicate_detected(self):
        tracker = RejectedPatchTracker()
        patch1 = _make_patch(search="old code", replace="new code")
        tracker.record_rejection(patch1, "no_improvement")

        patch2 = _make_patch(search="old code", replace="new code")
        is_dup, reason = tracker.is_duplicate(patch2)
        assert is_dup is True
        assert "similarity" in reason

    def test_slightly_similar_patch_passes(self):
        """similarity_threshold 默认 0.9，略有不同的 patch 不应被拦截。"""
        tracker = RejectedPatchTracker()  # default threshold=0.9
        patch1 = _make_patch(search="old code here", replace="new code here")
        tracker.record_rejection(patch1, "no_improvement")

        # different enough to be below 0.9
        patch2 = _make_patch(search="old code here", replace="completely rewritten code block v2")
        is_dup, _ = tracker.is_duplicate(patch2)
        assert is_dup is False

    def test_near_identical_patch_rejected(self):
        """几乎完全相同（> 0.9）的 patch 应被拦截。"""
        tracker = RejectedPatchTracker()
        patch1 = _make_patch(search="old code here abc", replace="new code here xyz")
        tracker.record_rejection(patch1, "no_improvement")

        # nearly identical — only 1 char different
        patch2 = _make_patch(search="old code here abc", replace="new code here xy!")
        is_dup, _ = tracker.is_duplicate(patch2)
        assert is_dup is True

    def test_different_patch_passes(self):
        tracker = RejectedPatchTracker()
        patch1 = _make_patch(search="old code A", replace="new code A")
        tracker.record_rejection(patch1, "no_improvement")

        patch2 = _make_patch(
            file_path="other.py",
            search="completely different search block",
            replace="completely different replace block",
        )
        is_dup, _ = tracker.is_duplicate(patch2)
        assert is_dup is False

    def test_dedup_check_returns_eval_result(self):
        tracker = RejectedPatchTracker()
        patch1 = _make_patch(search="x", replace="y")
        tracker.record_rejection(patch1, "smoke_regression")

        patch2 = _make_patch(search="x", replace="y")
        result = dedup_check(patch2, tracker, baseline_score=50.0)
        assert result is not None
        assert result.accepted is False
        assert result.failure_reason == "dedup_rejected"

    def test_dedup_check_passes_novel_patch(self):
        tracker = RejectedPatchTracker()
        patch1 = _make_patch(search="aaa", replace="bbb")
        tracker.record_rejection(patch1, "no_improvement")

        patch2 = _make_patch(search="completely_unrelated", replace="brand_new_code")
        result = dedup_check(patch2, tracker, baseline_score=50.0)
        assert result is None
