"""Tests for noa.eval_guard — quality gate, dedup, progressive sampling."""

import pytest

from noa.core.protocol import DeltaPatch, DiffBlock
from noa.eval_guard import (
    RejectedPatchTracker,
    dedup_check,
    quality_gate,
    should_stop_early,
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

    def test_similar_patch_detected(self):
        tracker = RejectedPatchTracker(similarity_threshold=0.7)
        patch1 = _make_patch(search="old code here", replace="new code here")
        tracker.record_rejection(patch1, "no_improvement")

        # slightly different but similar
        patch2 = _make_patch(search="old code here", replace="new code here v2")
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


# ── Progressive Sampling ─────────────────────────────────────


class TestProgressiveSampling:
    def test_clear_win_accepts_early(self):
        result = should_stop_early(
            batch_score=0.55, baseline_score=0.50,
            batch_size=6, total_samples=20,
        )
        assert result == "accept"

    def test_clear_loss_rejects_early(self):
        result = should_stop_early(
            batch_score=0.45, baseline_score=0.50,
            batch_size=6, total_samples=20,
        )
        assert result == "reject"

    def test_ambiguous_returns_none(self):
        result = should_stop_early(
            batch_score=0.505, baseline_score=0.50,
            batch_size=6, total_samples=20,
        )
        assert result is None

    def test_high_ratio_always_decides(self):
        # 80%+ sample ratio → always decide
        result = should_stop_early(
            batch_score=0.501, baseline_score=0.50,
            batch_size=17, total_samples=20,
        )
        assert result == "accept"

        result = should_stop_early(
            batch_score=0.499, baseline_score=0.50,
            batch_size=17, total_samples=20,
        )
        assert result == "reject"
