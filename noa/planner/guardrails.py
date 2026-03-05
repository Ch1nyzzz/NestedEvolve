"""Decision guardrails for planner actions."""

from __future__ import annotations

from noa.planner.protocol import PlannerDecision, PlannerState


_ACTIONS = {
    "observe",
    "analyze",
    "propose_patch",
    "evaluate_patch",
    "parallel_optimize",
    "spawn_sublayer",
    "stop",
}
_MIN_INTERMEDIATE_COVERAGE_FOR_PATCH = 0.95


def fallback_action(state: PlannerState) -> str:
    if not state.trajectories:
        return "observe"
    if state.diagnosis is None:
        return "analyze"
    if state.candidate_patch is None or not state.candidate_patch.diffs:
        return "propose_patch"
    return "evaluate_patch"


def validate_decision(
    state: PlannerState, decision: PlannerDecision
) -> PlannerDecision:
    action = decision.action
    if action not in _ACTIONS:
        decision.action = fallback_action(state)
        decision.reason = f"invalid action '{action}', rewritten by guardrail"
        decision.params = {}
        return decision

    if action == "analyze" and not state.trajectories:
        decision.action = "observe"
        decision.reason = "guardrail: analyze requires trajectories"
        decision.params = {}
        return decision

    if action == "propose_patch" and (
        state.diagnosis is None or not state.diagnosis.failure_patterns
    ):
        decision.action = "analyze"
        decision.reason = "guardrail: propose_patch requires diagnosis patterns"
        decision.params = {}
        return decision

    if action == "propose_patch":
        cov = state.last_intermediate_coverage
        if cov is not None and cov < _MIN_INTERMEDIATE_COVERAGE_FOR_PATCH:
            decision.action = "observe"
            decision.reason = (
                f"guardrail: intermediate coverage {cov:.0%} < {_MIN_INTERMEDIATE_COVERAGE_FOR_PATCH:.0%}; "
                "collect/repair trajectories before patching"
            )
            if not isinstance(decision.params, dict):
                decision.params = {}
            decision.params.setdefault("n_samples", 20)
            return decision

    if action == "parallel_optimize" and (
        state.diagnosis is None or not state.diagnosis.failure_patterns
    ):
        decision.action = "analyze"
        decision.reason = "guardrail: parallel_optimize requires diagnosis patterns"
        decision.params = {}
        return decision

    if action == "parallel_optimize":
        cov = state.last_intermediate_coverage
        if cov is not None and cov < _MIN_INTERMEDIATE_COVERAGE_FOR_PATCH:
            decision.action = "observe"
            decision.reason = (
                f"guardrail: intermediate coverage {cov:.0%} < {_MIN_INTERMEDIATE_COVERAGE_FOR_PATCH:.0%}; "
                "collect/repair trajectories before patching"
            )
            decision.params = {}
            return decision

    if action == "evaluate_patch" and (
        state.candidate_patch is None or not state.candidate_patch.diffs
    ):
        decision.action = "propose_patch"
        decision.reason = "guardrail: evaluate_patch requires candidate patch"
        decision.params = {}
        return decision

    # propose_patch 后必须 evaluate，不允许跳过
    if (
        action == "stop"
        and state.candidate_patch is not None
        and state.candidate_patch.diffs
    ):
        decision.action = "evaluate_patch"
        decision.reason = "guardrail: must evaluate pending patch before stopping"
        decision.params = {}
        return decision

    # stop 前：优先尝试 spawn_sublayer（L2 元优化）
    if action == "stop":
        can_spawn = (
            state.layer_context is not None
            and state.layer_context.can_spawn_sublayer()
            and state.no_improve_steps >= 2
        )
        if can_spawn:
            decision.action = "spawn_sublayer"
            decision.reason = (
                f"guardrail: spawn_sublayer available (no_improve={state.no_improve_steps}, "
                f"spawn_calls={state.budget.spawn_calls_used}/{state.layer_context.max_spawn_calls}), "
                "try meta-optimization before stopping"
            )
            decision.params = {}
            return decision

    # stop 前至少要 eval 3 次（或用完预算的 80%）
    if action == "stop":
        min_evals = min(3, state.budget.max_evals)
        budget_nearly_exhausted = state.budget.step_count >= int(
            state.budget.max_steps * 0.8
        ) or state.budget.evals_used >= int(state.budget.max_evals * 0.8)
        if state.budget.evals_used < min_evals and not budget_nearly_exhausted:
            decision.action = fallback_action(state)
            decision.reason = (
                f"guardrail: only {state.budget.evals_used}/{min_evals} evals done, "
                "too early to stop — continue exploring"
            )
            decision.params = {}
            return decision

    if action == "spawn_sublayer":
        can_spawn = (
            state.layer_context is not None
            and state.layer_context.can_spawn_sublayer()
            and state.no_improve_steps >= 2
        )
        if not can_spawn:
            decision.action = fallback_action(state)
            decision.reason = "guardrail: spawn_sublayer preconditions not met"
            decision.params = {}
            return decision

    if not isinstance(decision.params, dict):
        decision.params = {}
    return decision
