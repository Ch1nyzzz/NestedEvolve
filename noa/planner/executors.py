"""Action executors for planner-driven optimization loops (统一 L1/L2/L_n)."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

from noa.core.protocol import Diagnosis, LayerContext, Trajectory
from noa.eval_guard import (
    RejectedPatchTracker,
    dedup_check,
    quality_gate,
)
from noa.planner.protocol import ActionResult, PlannerState
from noa.stages.analyzer import analyze_incremental, _format_trajectory
from noa.stages.evaluator import evaluate
from noa.stages.initiator import collect_sources
from noa.stages.observer import observe_agentic
from noa.stages.optimizer import optimize_agentic
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from noa.tools.probe import ComponentProbe

log = logging.getLogger(__name__)


class BaseActionExecutor:
    def run(self, action: str, params: dict, state: PlannerState) -> ActionResult:
        method = getattr(self, f"_do_{action}", None)
        if method is None:
            return ActionResult(
                action=action, ok=False, error=f"Unknown action: {action}"
            )
        return method(params or {}, state)


class ActionExecutor(BaseActionExecutor):
    """统一 action executor — 通过 LayerContext 注入层级差异。"""

    def __init__(
        self,
        *,
        sys_desc,
        source_dir: str,
        target_factory,
        target,
        dataset: list,
        eval_fn,
        score_fn,
        model: str,
        layer_context: LayerContext | None = None,
        analyzer_max_tool_calls: int = 10,
        observer_max_tool_calls: int = 8,
        observe_default_samples: int = 20,
        eval_default_samples: int = 20,
        failure_threshold: float | None = None,
        observer_search_roots: list[str] | None = None,
        optimizer_max_tool_calls: int = 5,
        noa_dir: str | None = None,
        project_root: str | None = None,
        dataset_pickle_path: str | None = None,
    ):
        self.sys_desc = sys_desc
        self.source_dir = source_dir
        self.target_factory = target_factory
        self.target = target
        self.dataset = dataset
        self.eval_fn = eval_fn
        self.score_fn = score_fn
        self.model = model
        self.layer_context = layer_context
        self.analyzer_max_tool_calls = analyzer_max_tool_calls
        self.observer_max_tool_calls = observer_max_tool_calls
        self.observe_default_samples = observe_default_samples
        self.eval_default_samples = eval_default_samples
        self.failure_threshold = failure_threshold
        self.observer_search_roots = observer_search_roots
        self.optimizer_max_tool_calls = optimizer_max_tool_calls
        self.failure_pool = None
        self.rejected_patch_tracker = RejectedPatchTracker()
        self.noa_dir = noa_dir
        self.project_root = project_root
        self.dataset_pickle_path = dataset_pickle_path

    def _layer_context_text(self) -> str:
        if self.layer_context is None:
            return ""
        return self.layer_context.to_prompt_context()

    def _do_observe(self, params: dict, state: PlannerState) -> ActionResult:
        n_samples = int(params.get("n_samples", self.observe_default_samples))
        seed = int(params.get("seed", 42 + state.budget.step_count))
        trajectories = observe_agentic(
            self.target,
            self.dataset,
            n_samples=n_samples,
            seed=seed,
            score_fn=self.score_fn,
            model=self.model,
            source_dir=self.source_dir,
            search_roots=self.observer_search_roots,
            max_tool_calls=self.observer_max_tool_calls,
            layer_context=self._layer_context_text(),
            layer_context_obj=self.layer_context,
            # L2+ 元优化层的 target 是 optimizer subprocess，intermediate 是优化结果 dict
            # 而非 L0 组件级输出，无需检查 component_names
            required_intermediate_keys=(
                None
                if self.layer_context and self.layer_context.level >= 2
                else getattr(self.sys_desc, "component_names", None)
            ),
        )
        if not trajectories:
            return ActionResult(
                action="observe", ok=False, error="No trajectories collected"
            )

        mean_score = sum(t.f1 for t in trajectories) / len(trajectories) * 100
        with_intermediate = sum(
            1 for t in trajectories if bool(getattr(t, "intermediate_complete", False))
        )
        intermediate_coverage = with_intermediate / len(trajectories)
        replay_count = sum(
            1 for t in trajectories if getattr(t, "trace_source", "") == "replayed"
        )
        ft = self.failure_threshold
        if ft is None:
            scores = sorted(t.f1 for t in trajectories)
            ft = scores[len(scores) // 2] if scores else 0.5
        failures = sum(1 for t in trajectories if t.f1 < ft)
        self.sys_desc.source_files = collect_sources(self.source_dir)
        return ActionResult(
            action="observe",
            ok=True,
            summary=(
                f"Collected {len(trajectories)} trajectories, mean={mean_score:.2f}, failures={failures}, "
                f"intermediate={intermediate_coverage:.0%}, replayed={replay_count}"
            ),
            payload={
                "trajectories": trajectories,
                "mean_score": mean_score,
                "failure_count": failures,
                "failure_threshold": ft,
                "with_intermediate": with_intermediate,
                "intermediate_coverage": intermediate_coverage,
                "replay_count": replay_count,
            },
            llm_calls=1,
        )

    def _do_analyze(self, params: dict, state: PlannerState) -> ActionResult:
        if not state.trajectories:
            return ActionResult(
                action="analyze", ok=False, error="No trajectories to analyze"
            )

        probe = None
        try:
            p = ComponentProbe(self.target, self.sys_desc)
            if p.components:
                probe = p
        except Exception:
            pass

        diagnosis, self.failure_pool = analyze_incremental(
            self.sys_desc,
            state.trajectories,
            model=self.model,
            failure_threshold=self.failure_threshold,
            past_attempts=_history_for_prompt(state.history),
            pool=self.failure_pool,
            top_n=int(params.get("top_n", 5)),
            probe=probe,
            max_tool_calls=self.analyzer_max_tool_calls,
            layer_context=self._layer_context_text(),
        )
        active_pattern = (
            diagnosis.failure_patterns[0] if diagnosis.failure_patterns else None
        )
        return ActionResult(
            action="analyze",
            ok=True,
            summary=diagnosis.summary,
            payload={
                "diagnosis": diagnosis,
                "active_pattern": active_pattern,
                "pool_size": len(self.failure_pool)
                if self.failure_pool is not None
                else 0,
            },
            llm_calls=1,
        )

    def _do_propose_patch(self, params: dict, state: PlannerState) -> ActionResult:
        if state.diagnosis is None or not state.diagnosis.failure_patterns:
            return ActionResult(
                action="propose_patch", ok=False, error="No diagnosis patterns"
            )
        idx = int(params.get("pattern_index", 0))
        idx = max(0, min(idx, len(state.diagnosis.failure_patterns) - 1))
        pattern = state.diagnosis.failure_patterns[idx]
        single_diag = Diagnosis(
            failure_patterns=[pattern],
            summary=pattern.get("pattern", "unknown"),
            raw_analysis=state.diagnosis.raw_analysis,
        )
        # 采样与 active pattern 相关的 failure trajectory，让 Optimizer 看到原始 intermediate
        traj_text = _sample_trajectories_for_optimizer(state.trajectories, pattern, n=3)
        patch = optimize_agentic(
            self.sys_desc,
            single_diag,
            model=self.model,
            past_attempts=_history_for_prompt(state.history),
            layer_context=self._layer_context_text(),
            max_tool_calls=self.optimizer_max_tool_calls,
            trajectory_samples=traj_text,
        )
        return ActionResult(
            action="propose_patch",
            ok=bool(patch.diffs),
            summary=f"Patch diffs={len(patch.diffs)} quality={patch.quality_score:.2f}",
            payload={
                "patch": patch,
                "active_pattern": pattern,
            },
            llm_calls=2,
            error=None if patch.diffs else "Optimizer returned empty patch",
        )

    def _do_evaluate_patch(self, params: dict, state: PlannerState) -> ActionResult:
        if state.candidate_patch is None or not state.candidate_patch.diffs:
            return ActionResult(
                action="evaluate_patch", ok=False, error="No candidate patch"
            )

        patch = state.candidate_patch

        # --- Pre-filter 1: Quality Gate ---
        gate_result = quality_gate(patch, baseline_score=state.current_score)
        if gate_result is not None:
            log.info(
                f"[EvalGuard] Quality gate rejected: "
                f"quality_score={patch.quality_score:.2f}"
            )
            self.rejected_patch_tracker.record_rejection(patch, "quality_gate")
            return ActionResult(
                action="evaluate_patch",
                ok=True,
                summary=f"REJECTED (quality gate: {patch.quality_score:.2f}) {gate_result.before_score:.2f} -> {gate_result.after_score:.2f}",
                payload={"eval_result": gate_result},
                eval_calls=0,  # 没有实际执行 pipeline
            )

        # --- Pre-filter 2: Rejected Patch Dedup ---
        dedup_result = dedup_check(
            patch, self.rejected_patch_tracker, baseline_score=state.current_score
        )
        if dedup_result is not None:
            log.info("[EvalGuard] Dedup gate rejected: similar to previously rejected patch")
            return ActionResult(
                action="evaluate_patch",
                ok=True,
                summary=f"REJECTED (dedup) {dedup_result.before_score:.2f} -> {dedup_result.after_score:.2f}",
                payload={"eval_result": dedup_result},
                eval_calls=0,
            )

        # --- Full cascade evaluation (with progressive sampling) ---
        n_samples = int(params.get("n_samples", self.eval_default_samples))
        seed = int(params.get("seed", 43))
        result = evaluate(
            source_files=self.sys_desc.source_files,
            source_dir=self.source_dir,
            patch=patch,
            dataset=self.dataset,
            eval_fn=self.eval_fn,
            target_factory=self.target_factory,
            baseline_score=state.current_score,
            n_samples=n_samples,
            seed=seed,
            layer_context=self.layer_context,
            progressive=True,
        )

        # 记录被 reject 的 patch
        if not result.accepted:
            self.rejected_patch_tracker.record_rejection(
                patch, result.failure_reason or "no_improvement"
            )

        if result.accepted:
            self.target = self.target_factory(self.source_dir)
            self.sys_desc.source_files = collect_sources(self.source_dir)

        return ActionResult(
            action="evaluate_patch",
            ok=True,
            summary=f"{'ACCEPTED' if result.accepted else 'REJECTED'} {result.before_score:.2f} -> {result.after_score:.2f}",
            payload={"eval_result": result},
            eval_calls=1,
        )

    def _do_spawn_sublayer(self, params: dict, state: PlannerState) -> ActionResult:
        if self.layer_context is None or not self.layer_context.can_spawn_sublayer():
            return ActionResult(
                action="spawn_sublayer",
                ok=False,
                error="Cannot spawn sublayer: depth/budget limit reached or no layer_context",
            )
        child_level = self.layer_context.level + 1
        if not self.noa_dir or not self.project_root:
            return ActionResult(
                action="spawn_sublayer",
                ok=True,
                summary=(
                    f"Sublayer spawn requested (used={self.layer_context.spawn_calls_used}/"
                    f"{self.layer_context.max_spawn_calls})"
                ),
                payload={
                    "spawn_calls_used": self.layer_context.spawn_calls_used,
                    "noa_modified": False,
                },
            )
        log.info(
            f"[Spawn] Starting L{child_level} child optimizer (noa_dir={self.noa_dir})"
        )

        try:
            child_result = self._run_child_optimizer(child_level, state)
            noa_modified = child_result.get("accepted", 0) > 0
            return ActionResult(
                action="spawn_sublayer",
                ok=True,
                summary=f"L{child_level} done: score={child_result.get('final_score', 0):.2f}, "
                f"accepted={child_result.get('accepted', 0)}, noa_modified={noa_modified}",
                payload={
                    "noa_modified": noa_modified,
                    "child_score": child_result.get("final_score", 0),
                    "child_accepted": child_result.get("accepted", 0),
                    "child_steps": child_result.get("planner_steps", 0),
                    "spawn_calls_used": self.layer_context.spawn_calls_used,
                },
            )
        except Exception as e:
            log.error(f"[Spawn] L{child_level} crashed: {e}", exc_info=True)
            return ActionResult(
                action="spawn_sublayer",
                ok=False,
                error=f"L{child_level} child crashed: {str(e)[:500]}",
                payload={
                    "noa_modified": False,
                    "spawn_calls_used": self.layer_context.spawn_calls_used,
                },
            )

    def _run_child_optimizer(self, child_level: int, state: PlannerState) -> dict:
        """同进程运行 L2+ child optimizer，修改 noa_dir 中的代码。"""
        from noa.engine import NOptimizer  # 延迟导入避免循环

        noa_dir = self.noa_dir
        project_root = self.project_root

        # 确保 dataset_pickle_path 存在
        dpp = self.dataset_pickle_path
        if not dpp:
            cache_dir = os.path.join(project_root, ".noa_cache")
            dpp = serialize_dataset(self.dataset, cache_dir=cache_dir)

        # child target_factory: 调 subprocess mini-L1 评估 noa/ 修改效果
        def child_target_factory(noa_source_dir):
            def target(question):
                result = run_layer_subprocess(
                    noa_dir=noa_source_dir,
                    project_root=project_root,
                    target_source_dir=self.source_dir,
                    dataset_pickle_path=dpp,
                    layer_level=1,
                    max_steps=8,
                    n_samples=10,
                    eval_n_samples=10,
                    max_llm_calls=40,
                    max_evals=4,
                    max_no_improve_steps=3,
                    model=self.model,
                    isolate_source=True,
                )
                return SimpleNamespace(
                    answer=str(result.get("final_score", 0)),
                    intermediate=result,
                )

            return target

        # child eval_fn: 多次运行 target 取平均分
        def child_eval_fn(target, dataset):
            scores = []
            details = []
            for ex in dataset:
                result = target(ex.question)
                s = float(result.answer) if result.answer else 0.0
                scores.append(s)
                details.append({"question": ex.question, "f1": s})
            avg = sum(scores) / len(scores) if scores else 0.0
            return {"score": avg, "details": details}

        # child score_fn: L2 的 prediction 是分数字符串
        def child_score_fn(prediction: str, _ground_truth: str) -> float:
            try:
                s = float(str(prediction).strip())
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(100.0, s)) / 100.0

        # child dataset: 虚拟数据条目（L2 评估的是 mini-L1 运行效果）
        child_dataset = [
            SimpleNamespace(question="opt_run_1", answer="100"),
            SimpleNamespace(question="opt_run_2", answer="100"),
        ]

        # child layer_context
        child_layer_context = LayerContext(
            layer_id=f"L{child_level}",
            level=child_level,
            writable_root=noa_dir,
            readable_roots=[noa_dir, self.source_dir],
            parent_history=state.history[-20:],
            max_depth=self.layer_context.max_depth,
            max_spawn_calls=max(0, self.layer_context.max_spawn_calls - 1),
        )

        log.info(
            f"[Spawn] Instantiating L{child_level} NOptimizer (source_dir={noa_dir})"
        )
        child = NOptimizer(
            source_dir=noa_dir,
            target_factory=child_target_factory,
            dataset=child_dataset,
            eval_fn=child_eval_fn,
            max_steps=12,
            n_samples=2,
            eval_n_samples=2,
            model=self.model,
            score_fn=child_score_fn,
            max_llm_calls=60,
            max_evals=8,
            max_no_improve_steps=4,
            layer_context=child_layer_context,
            observer_search_roots=[noa_dir],
        )
        return child.run()

    def _do_stop(self, params: dict, state: PlannerState) -> ActionResult:
        return ActionResult(action="stop", ok=True, summary="Planner requested stop")


# --- Backward compatibility aliases ---
# 保持旧 import 可用（engine.py 等引用）
L1ActionExecutor = ActionExecutor
L2ActionExecutor = ActionExecutor


def _history_for_prompt(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for h in history[-8:]:
        line = f"step={h.get('step', '?')} action={h.get('action', '?')} ok={h.get('ok', '?')} summary={h.get('summary', '')}"
        payload = h.get("payload", {})
        if payload:
            line += f" payload={payload}"
        lines.append(line)
    return "\n".join(lines)


def _sample_trajectories_for_optimizer(
    trajectories: list[Trajectory],
    pattern: dict,
    n: int = 3,
) -> str:
    """采样与 pattern 相关的 failure trajectory，格式化为 Optimizer 可读文本。"""
    if not trajectories:
        return "(no trajectories)"
    # 按 F1 升序排列，优先选最差的 failure
    failures = sorted(trajectories, key=lambda t: t.f1)
    # 优先选包含 affected_component 关键词的 trajectory
    affected = pattern.get("affected_component", "").lower()
    selected: list[Trajectory] = []
    for t in failures:
        if len(selected) >= n:
            break
        if not t.intermediate:
            continue
        # 检查 intermediate 中是否有与 affected_component 相关的空值或异常
        inter_str = str(t.intermediate).lower()
        if affected and any(kw in inter_str for kw in affected.split()):
            selected.append(t)
    # 补足到 n 条
    for t in failures:
        if len(selected) >= n:
            break
        if t not in selected and t.intermediate:
            selected.append(t)
    if not selected:
        return "(no trajectories with intermediate data)"
    parts = []
    for i, t in enumerate(selected):
        parts.append(_format_trajectory(t, label=f"FAIL-{i+1}"))
    return "\n---\n".join(parts)
