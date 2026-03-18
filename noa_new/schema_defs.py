"""Schema and template definitions for the experimental noa_new package."""

from __future__ import annotations

from copy import deepcopy

TRACE_SCHEMA_VERSION = "1.0"

L1_SLOT_TEMPLATES = {
    "system_prompt.md": """# system_prompt

## Role
你是 L1 优化器，只能通过 bash 和 execute_code 工作。

## Invariants
- 不碰 test_set
- 每步都要有证据，不做无验证修改
- 优先低风险改动，再做高风险改动
""",
    "loop_policy.md": """# loop_policy

- 每轮顺序：observe -> diagnose -> patch -> verify -> decide
- 连续 3 轮同一策略无提升时必须换策略
- 没有证据时不要直接 patch
""",
    "tool_policy.md": """# tool_policy

- bash 用于 ls/cat/rg/git/python 等环境探索与文件操作
- execute_code 用于并发评估、数据处理、批量分析
- 改代码前先用 bash 读取目标文件
""",
    "budget_policy.md": """# budget_policy

- 先做 cheap observe，再做更贵的 verify
- 小差异先用小样本验证，再决定是否扩大评估
- 不要把预算浪费在重复实验上
""",
    "eval_schema.md": """# eval_schema

主指标:
- accuracy

辅助指标:
- format_compliance
- avg_latency_sec
- failure_diversity
- confidence_calibration
""",
    "playbook.md": """# playbook

只追加经验，不重写历史。
""",
}

TRACE_SCHEMA = {
    "version": TRACE_SCHEMA_VERSION,
    "phases": {
        "observe": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "n_samples",
                "score",
                "failure_count",
                "failure_categories",
            ],
            "example": {
                "step": 1,
                "phase": "observe",
                "timestamp": "2026-03-17T10:00:00",
                "n_samples": 25,
                "score": 0.68,
                "failure_count": 8,
                "failure_categories": {
                    "wrong_answer": 5,
                    "no_answer": 2,
                    "format_error": 1,
                },
                "duration_sec": 45,
            },
        },
        "diagnose": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "attribution",
                "confidence",
                "evidence",
                "prior_attempts",
            ],
            "example": {
                "step": 1,
                "phase": "diagnose",
                "timestamp": "2026-03-17T10:02:00",
                "attribution": "prompts.py:L42",
                "component": "few_shot_examples",
                "confidence": "high",
                "evidence": "8/8 failures share the same format error",
                "prior_attempts": 0,
                "alternative_attributions": [
                    "components.py:L120 parser bug (low confidence)"
                ],
            },
        },
        "patch": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "patch_id",
                "target_file",
                "hypothesis",
                "diff_summary",
                "risk",
            ],
            "example": {
                "step": 1,
                "phase": "patch",
                "timestamp": "2026-03-17T10:05:00",
                "patch_id": "p1",
                "target_file": "prompts.py",
                "hypothesis": "adding an explicit format instruction fixes parse errors",
                "diff_summary": "add output-format guidance",
                "risk": "low",
                "expected_metrics_impact": {
                    "accuracy": "+0.05",
                    "format_compliance": "+0.10",
                },
            },
        },
        "verify": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "patch_id",
                "metrics",
                "baseline_metrics",
                "delta",
            ],
            "example": {
                "step": 1,
                "phase": "verify",
                "timestamp": "2026-03-17T10:07:00",
                "patch_id": "p1",
                "metrics": {
                    "accuracy": 0.72,
                    "format_compliance": 0.95,
                    "avg_latency_sec": 2.3,
                    "failure_diversity": 3,
                    "confidence_calibration": 0.81,
                },
                "baseline_metrics": {
                    "accuracy": 0.68,
                    "format_compliance": 0.85,
                },
                "delta": {
                    "accuracy": 0.04,
                    "format_compliance": 0.10,
                },
                "duration_sec": 60,
            },
        },
        "decide": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "patch_id",
                "decision",
                "reason",
            ],
            "example": {
                "step": 1,
                "phase": "decide",
                "timestamp": "2026-03-17T10:08:00",
                "patch_id": "p1",
                "decision": "commit",
                "reason": "accuracy improved without regressions",
                "pareto_status": "new_front",
            },
        },
        "strategy_change": {
            "required": [
                "step",
                "phase",
                "timestamp",
                "old_strategy",
                "new_strategy",
                "reason",
            ],
            "example": {
                "step": 3,
                "phase": "strategy_change",
                "timestamp": "2026-03-17T10:15:00",
                "old_strategy": "improving few-shot examples",
                "new_strategy": "fixing parser logic",
                "reason": "three rounds with diminishing returns",
                "strategy_tenure_at_switch": 3,
            },
        },
    },
}

L2_PATCH_SCHEMA = {
    "required": [
        "patch_id",
        "timestamp",
        "slot",
        "action",
        "rule",
        "evidence",
        "expected_effect",
        "rollback_condition",
        "risk",
    ],
    "allowed_slots": [
        "system_prompt",
        "loop_policy",
        "tool_policy",
        "budget_policy",
        "eval_schema",
        "playbook",
    ],
    "allowed_actions": ["append_rule", "modify_rule", "remove_rule"],
    "example": {
        "patch_id": "l2_p1",
        "timestamp": "2026-03-17T12:00:00",
        "slot": "loop_policy",
        "action": "append_rule",
        "rule": "连续 2 轮同一 attribution 无提升时，强制换目标文件",
        "evidence": "trace steps 3-5 all modified prompts.py:L42 with flat deltas",
        "diagnosis": "curvature",
        "expected_effect": "reduce wasted compute on flat regions",
        "rollback_condition": "next run accuracy drops > 3%",
        "risk": "low",
        "verified": None,
        "verification_result": None,
    },
}

DEFAULT_METRICS = {
    "accuracy": 0.0,
    "format_compliance": 0.0,
    "avg_latency_sec": 0.0,
    "failure_diversity": 0,
    "confidence_calibration": 0.0,
    "total_patches_tried": 0,
    "patches_accepted": 0,
    "patches_rejected": 0,
    "current_strategy": "baseline",
    "strategy_tenure": 0,
}


def default_metrics() -> dict:
    """Return a mutable metrics template."""
    return deepcopy(DEFAULT_METRICS)
