"""NOptimizer — planner-driven L1 optimizer (no fixed workflow fallback)."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable

from noa.core.protocol import LayerContext, SystemDescription
from noa.planner.agent import PlannerAgent
from noa.planner.executors import ActionExecutor
from noa.planner.guardrails import validate_decision
from noa.planner.protocol import (
    PlannerBudget,
    PlannerDecision,
    PlannerState,
    PlannerStepRecord,
)
from noa.planner.reducer import apply_result, should_stop
from noa.planner.trace import PlannerTraceWriter
from noa.stages.initiator import initiate
from utils.llm import resolve_model


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sort_patterns_by_severity(patterns: list[dict]) -> list[dict]:
    """Utility kept for compatibility in external imports."""
    return sorted(
        patterns, key=lambda p: _SEVERITY_ORDER.get(p.get("severity", "low"), 2)
    )


class NOptimizer:
    """Planner-driven nested optimizer for L1 target systems."""

    def __init__(
        self,
        source_dir: str,
        target_factory: Callable[[str], object],
        dataset: list,
        eval_fn,
        *,
        max_steps: int = 20,
        max_iterations: int | None = None,  # legacy alias, mapped to max_steps
        n_samples: int = 20,
        eval_n_samples: int = 20,
        model: str = resolve_model("gpt-4.1-mini"),
        planner_model: str | None = None,
        system_description: str = "",
        score_fn,
        failure_threshold: float | None = None,
        max_tool_calls: int = 10,
        observer_max_tool_calls: int = 8,
        optimizer_max_tool_calls: int = 5,
        planner_budget: PlannerBudget | None = None,
        max_llm_calls: int = 120,
        max_evals: int = 20,
        max_no_improve_steps: int = 5,
        layer_context: LayerContext | None = None,
        observer_search_roots: list[str] | None = None,
        noa_dir: str | None = None,
        dataset_pickle_path: str | None = None,
    ):
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.n_samples = n_samples
        self.eval_n_samples = eval_n_samples
        self.model = model
        self.planner_model = planner_model or model
        self.system_description = system_description
        self.score_fn = score_fn
        self.failure_threshold = failure_threshold
        self.max_tool_calls = max_tool_calls
        self.observer_max_tool_calls = observer_max_tool_calls
        self.optimizer_max_tool_calls = optimizer_max_tool_calls

        effective_steps = max_iterations if max_iterations is not None else max_steps
        if planner_budget is None:
            max_spawn = layer_context.max_spawn_calls if layer_context else 2
            planner_budget = PlannerBudget(
                max_steps=effective_steps,
                max_llm_calls=max_llm_calls,
                max_evals=max_evals,
                max_no_improve_steps=max_no_improve_steps,
                target_delta=float("inf"),
                max_spawn_calls=max_spawn,
            )
        self.planner_budget = planner_budget
        self.layer_context = layer_context
        self.observer_search_roots = observer_search_roots
        self.noa_dir = noa_dir
        self.dataset_pickle_path = dataset_pickle_path

        self.target = target_factory(source_dir)
        self.sys_desc: SystemDescription | None = None
        self.history: list[dict] = []
        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(self) -> dict:
        """Run L1 planner loop and return optimization result."""
        print("\n[NOA] === Initiate ===")
        self.sys_desc = initiate(
            source_dir=self.source_dir,
            model=self.model,
            system_description=self.system_description,
        )
        print(f"[NOA] Workflow: {self.sys_desc.workflow_summary}")
        print(f"[NOA] Source files: {len(self.sys_desc.source_files)}")

        layer_label = f"l{self.layer_context.level}" if self.layer_context else "l1"
        state = PlannerState(
            layer=layer_label,
            budget=PlannerBudget(**asdict(self.planner_budget)),
            baseline_score=0.0,
            current_score=0.0,
            layer_context=self.layer_context,
        )
        planner = PlannerAgent(model=self.planner_model)
        executor = ActionExecutor(
            sys_desc=self.sys_desc,
            source_dir=self.source_dir,
            target_factory=self.target_factory,
            target=self.target,
            dataset=self.dataset,
            eval_fn=self.eval_fn,
            score_fn=self.score_fn,
            model=self.model,
            layer_context=self.layer_context,
            analyzer_max_tool_calls=self.max_tool_calls,
            observer_max_tool_calls=self.observer_max_tool_calls,
            observe_default_samples=self.n_samples,
            eval_default_samples=self.eval_n_samples,
            failure_threshold=self.failure_threshold,
            observer_search_roots=self.observer_search_roots,
            optimizer_max_tool_calls=self.optimizer_max_tool_calls,
            noa_dir=self.noa_dir,
            project_root=self._project_root,
            dataset_pickle_path=self.dataset_pickle_path,
        )

        run_tag = os.path.basename(os.path.abspath(self.source_dir))
        with PlannerTraceWriter(
            project_root=self._project_root, layer=layer_label, run_tag=run_tag
        ) as tracer:
            trace_path = tracer.path
            while True:
                if should_stop(state):
                    decision = PlannerDecision(
                        action="stop",
                        params={},
                        reason="stop condition reached (budget/no-improve/target)",
                        expected_gain=0.0,
                        risk="low",
                    )
                else:
                    decision = planner.decide(state)
                    state.budget.llm_calls_used += 1  # decision LLM cost
                    decision = validate_decision(state, decision)

                result = executor.run(decision.action, decision.params, state)
                state = apply_result(state, decision, result)

                record = PlannerStepRecord(
                    layer=layer_label,
                    step=state.budget.step_count,
                    decision=decision,
                    result=result,
                    state_summary=state.summary(),
                )
                tracer.write(record)

                # spawn_sublayer 成功且修改了 noa/ → 中断循环，通知 orchestrator 重启
                if (
                    decision.action == "spawn_sublayer"
                    and result.ok
                    and result.payload.get("noa_modified")
                ):
                    break

                if decision.action == "stop":
                    break

        self.history = state.history
        final_score = (
            state.current_score
            if state.current_score
            else (state.last_observe_mean or 0.0)
        )
        baseline = (
            state.baseline_score
            if state.baseline_score
            else (state.initial_observe_score or 0.0)
        )

        print("\n[NOA] === Done ===")
        print(f"[NOA] Baseline F1: {baseline:.2f}")
        print(f"[NOA] Final F1: {final_score:.2f}")
        print(f"[NOA] Planner steps: {state.budget.step_count}")
        print(f"[NOA] Accepted patches: {state.accepted_patches}")

        spawn_restart = any(
            h.get("action") == "spawn_sublayer"
            and h.get("payload", {}).get("noa_modified")
            for h in state.history
        )

        return {
            "final_score": final_score,
            "baseline_score": baseline,
            "iterations": state.budget.step_count,
            "accepted": state.accepted_patches,
            "history": self.history,
            "planner_steps": state.budget.step_count,
            "budget_usage": state.budget.to_summary(),
            "action_stats": state.action_counts,
            "planner_trace_path": trace_path,
            "spawn_restart": spawn_restart,
        }

    def __call__(self, **kwargs) -> dict:
        return self.run()


def _format_history(history: list[dict]) -> str:
    """Compact history formatter used by downstream prompts/logging."""
    if not history:
        return ""
    lines: list[str] = []
    for h in history:
        line = (
            f"step={h.get('step', '?')} action={h.get('action', '?')} ok={h.get('ok', '?')} "
            f"reason={h.get('reason', '')[:80]} summary={h.get('summary', '')[:120]}"
        )
        payload = h.get("payload")
        if payload:
            line += f" payload={str(payload)[:280]}"
        err = h.get("error")
        if err:
            line += f" error={str(err)[:160]}"
        lines.append(line)
    return "\n".join(lines)
