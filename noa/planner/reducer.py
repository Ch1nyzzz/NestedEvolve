"""Reducer updates planner state from action results."""

from __future__ import annotations

from noa.planner.protocol import ActionResult, PlannerDecision, PlannerState


def apply_result(
    state: PlannerState, decision: PlannerDecision, result: ActionResult
) -> PlannerState:
    if decision.action != "stop":
        state.budget.step_count += 1
        state.budget.llm_calls_used += max(0, int(result.llm_calls))
        state.budget.evals_used += max(0, int(result.eval_calls))
    state.action_counts[decision.action] = (
        state.action_counts.get(decision.action, 0) + 1
    )

    if decision.action == "observe" and result.ok:
        trajectories = result.payload.get("trajectories", [])
        state.trajectories = trajectories
        state.last_observe_mean = float(result.payload.get("mean_score", 0.0))
        state.last_failure_count = int(result.payload.get("failure_count", 0))
        state.last_with_intermediate = int(result.payload.get("with_intermediate", 0))
        if "intermediate_coverage" in result.payload:
            state.last_intermediate_coverage = float(
                result.payload.get("intermediate_coverage", 0.0)
            )
        state.last_replay_count = int(result.payload.get("replay_count", 0))
        if state.initial_observe_score is None:
            state.initial_observe_score = state.last_observe_mean
            state.baseline_score = state.last_observe_mean
            state.current_score = state.last_observe_mean

    elif decision.action == "analyze" and result.ok:
        state.diagnosis = result.payload.get("diagnosis")
        state.active_pattern = result.payload.get("active_pattern")

    elif decision.action == "propose_patch" and result.ok:
        state.candidate_patch = result.payload.get("patch")
        if result.payload.get("active_pattern"):
            state.active_pattern = result.payload.get("active_pattern")

    elif decision.action == "evaluate_patch" and result.ok:
        eval_result = result.payload.get("eval_result")
        if eval_result is not None:
            state.last_eval = eval_result
            if eval_result.accepted:
                state.current_score = float(eval_result.after_score)
                state.accepted_patches += 1
            if getattr(eval_result, "delta", 0.0) > 0:
                state.no_improve_steps = 0
            else:
                state.no_improve_steps += 1
        state.candidate_patch = None

    elif decision.action == "spawn_sublayer" and result.ok:
        state.trajectories = []
        state.diagnosis = None
        state.no_improve_steps = 0
        state.budget.spawn_calls_used += 1
        if state.layer_context is not None:
            state.layer_context.spawn_calls_used = state.budget.spawn_calls_used

    elif decision.action != "stop":
        state.no_improve_steps += 1

    history_item = {
        "step": state.budget.step_count,
        "action": decision.action,
        "reason": decision.reason,
        "expected_gain": decision.expected_gain,
        "ok": result.ok,
        "summary": result.summary,
        "error": result.error,
    }
    if result.payload:
        history_item["payload"] = _compact_payload(result.payload)
    state.history.append(history_item)
    return state


def should_stop(state: PlannerState) -> bool:
    if state.budget.reached_limit():
        return True
    # spawn_sublayer 可用时，不触发硬停止（让 guardrail 重定向到 spawn）
    can_spawn = (
        state.layer_context is not None
        and state.layer_context.can_spawn_sublayer()
        and state.no_improve_steps >= 2
    )
    if can_spawn:
        return False
    # 只有在至少尝试过 3 次 eval 后，no_improve 才能触发硬停止
    min_evals_before_stop = min(3, state.budget.max_evals)
    if (
        state.no_improve_steps >= state.budget.max_no_improve_steps
        and state.budget.evals_used >= min_evals_before_stop
    ):
        return True
    if (
        state.current_score - state.baseline_score
    ) >= state.budget.target_delta and state.accepted_patches > 0:
        return True
    return False


def _compact_payload(payload: dict) -> dict:
    compact = {}
    for key, val in payload.items():
        if key == "trajectories":
            compact["n_trajectories"] = len(val)
            continue
        if key == "with_intermediate":
            compact["with_intermediate"] = int(val)
            continue
        if key == "intermediate_coverage":
            compact["intermediate_coverage"] = float(val)
            continue
        if key == "replay_count":
            compact["replay_count"] = int(val)
            continue
        if key == "diagnosis" and val is not None:
            compact["diagnosis_summary"] = getattr(val, "summary", "")
            compact["n_patterns"] = len(getattr(val, "failure_patterns", []))
            continue
        if key == "patch" and val is not None:
            compact["n_diffs"] = len(getattr(val, "diffs", []))
            compact["patch_quality_score"] = getattr(val, "quality_score", 0.0)
            continue
        if key == "eval_result" and val is not None:
            compact["before"] = getattr(val, "before_score", 0.0)
            compact["after"] = getattr(val, "after_score", 0.0)
            compact["accepted"] = getattr(val, "accepted", False)
            compact["delta"] = getattr(val, "delta", 0.0)
            continue
        compact[key] = val
    return compact
