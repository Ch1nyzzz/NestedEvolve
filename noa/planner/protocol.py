"""Planner protocol types used by L1/L2 control-flow planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from noa.core.protocol import (
    DeltaPatch,
    Diagnosis,
    EvalResult,
    LayerContext,
    Trajectory,
)


ActionName = Literal[
    "observe", "analyze", "propose_patch", "evaluate_patch", "spawn_sublayer", "stop"
]


@dataclass
class PlannerBudget:
    max_steps: int = 20
    max_llm_calls: int = 80
    max_evals: int = 12
    max_no_improve_steps: int = 5
    target_delta: float = float("inf")
    max_spawn_calls: int = 2
    step_count: int = 0
    llm_calls_used: int = 0
    evals_used: int = 0
    spawn_calls_used: int = 0

    def reached_limit(self) -> bool:
        return (
            self.step_count >= self.max_steps
            or self.llm_calls_used >= self.max_llm_calls
            or self.evals_used >= self.max_evals
        )

    def to_summary(self) -> dict:
        return {
            "steps": f"{self.step_count}/{self.max_steps}",
            "llm_calls": f"{self.llm_calls_used}/{self.max_llm_calls}",
            "evals": f"{self.evals_used}/{self.max_evals}",
            "spawn_calls": f"{self.spawn_calls_used}/{self.max_spawn_calls}",
            "max_no_improve_steps": self.max_no_improve_steps,
            "target_delta": self.target_delta,
        }


@dataclass
class PlannerDecision:
    action: ActionName
    params: dict = field(default_factory=dict)
    reason: str = ""
    expected_gain: float = 0.0
    risk: str = ""


@dataclass
class ActionResult:
    action: ActionName
    ok: bool
    summary: str = ""
    payload: dict = field(default_factory=dict)
    llm_calls: int = 0
    eval_calls: int = 0
    error: str | None = None


@dataclass
class PlannerStepRecord:
    layer: str
    step: int
    decision: PlannerDecision
    result: ActionResult
    state_summary: dict

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "step": self.step,
            "decision": {
                "action": self.decision.action,
                "params": self.decision.params,
                "reason": self.decision.reason,
                "expected_gain": self.decision.expected_gain,
                "risk": self.decision.risk,
            },
            "result": {
                "action": self.result.action,
                "ok": self.result.ok,
                "summary": self.result.summary,
                "payload": self.result.payload,
                "llm_calls": self.result.llm_calls,
                "eval_calls": self.result.eval_calls,
                "error": self.result.error,
            },
            "state_summary": self.state_summary,
        }


@dataclass
class PlannerState:
    layer: str
    budget: PlannerBudget
    baseline_score: float = 0.0
    current_score: float = 0.0
    initial_observe_score: float | None = None
    last_observe_mean: float | None = None
    last_failure_count: int = 0
    last_with_intermediate: int = 0
    last_intermediate_coverage: float | None = None
    last_replay_count: int = 0
    trajectories: list[Trajectory] = field(default_factory=list)
    diagnosis: Diagnosis | None = None
    active_pattern: dict | None = None
    candidate_patch: DeltaPatch | None = None
    last_eval: EvalResult | None = None
    no_improve_steps: int = 0
    accepted_patches: int = 0
    history: list[dict] = field(default_factory=list)
    layer_context: LayerContext | None = None
    action_counts: dict[str, int] = field(
        default_factory=lambda: {
            "observe": 0,
            "analyze": 0,
            "propose_patch": 0,
            "evaluate_patch": 0,
            "spawn_sublayer": 0,
            "stop": 0,
        }
    )

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "baseline_score": round(self.baseline_score, 4),
            "current_score": round(self.current_score, 4),
            "delta": round(self.current_score - self.baseline_score, 4),
            "last_observe_mean": None
            if self.last_observe_mean is None
            else round(self.last_observe_mean, 4),
            "last_failure_count": self.last_failure_count,
            "last_with_intermediate": self.last_with_intermediate,
            "last_intermediate_coverage": None
            if self.last_intermediate_coverage is None
            else round(self.last_intermediate_coverage, 4),
            "last_replay_count": self.last_replay_count,
            "n_trajectories": len(self.trajectories),
            "has_diagnosis": self.diagnosis is not None,
            "n_patterns": len(self.diagnosis.failure_patterns) if self.diagnosis else 0,
            "has_patch": self.candidate_patch is not None
            and bool(self.candidate_patch.diffs),
            "accepted_patches": self.accepted_patches,
            "no_improve_steps": self.no_improve_steps,
            "action_counts": dict(self.action_counts),
            "budget": self.budget.to_summary(),
        }
