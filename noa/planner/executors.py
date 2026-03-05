"""Action executors for planner-driven optimization loops (统一 L1/L2/L_n)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from noa.core.protocol import (
    DeltaPatch,
    Diagnosis,
    EvalResult,
    LayerContext,
    Trajectory,
)
from noa.planner.protocol import ActionResult, PlannerState
from noa.stages.analyzer import analyze_incremental, _format_trajectory
from noa.stages.evaluator import (
    evaluate,
    generate_eval_feedback,
    extract_patch_regions,
    merge_patches,
)
from noa.stages.initiator import collect_sources
from noa.stages.observer import observe_agentic
from noa.stages.optimizer import optimize_agentic
from noa.subprocess_runner import run_layer_subprocess, serialize_dataset
from noa.tools.probe import ComponentProbe
from noa.diff_utils import apply_diffs_in_memory, commit_to_source, write_to_temp_dir

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
        spawn_config: dict | None = None,
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
        self.noa_dir = noa_dir
        self.project_root = project_root
        self.dataset_pickle_path = dataset_pickle_path
        self.spawn_config = spawn_config or {}

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

        # 构建 eval feedback 上下文
        eval_feedback_text = ""
        if state.eval_feedback_history:
            import json as _json

            recent = state.eval_feedback_history[-3:]
            eval_feedback_text = _json.dumps(recent, ensure_ascii=False, indent=1)[
                :3000
            ]

        diagnosis, self.failure_pool = analyze_incremental(
            self.sys_desc,
            state.trajectories,
            model=self.model,
            failure_threshold=self.failure_threshold,
            past_attempts=_history_for_prompt(state.history),
            pool=self.failure_pool,
            top_n=int(params.get("top_n", 10)),
            probe=probe,
            max_tool_calls=self.analyzer_max_tool_calls,
            layer_context=self._layer_context_text(),
            eval_feedback=eval_feedback_text,
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

        n_samples = int(params.get("n_samples", self.eval_default_samples))
        seed = int(params.get("seed", 43))
        result = evaluate(
            source_files=self.sys_desc.source_files,
            source_dir=self.source_dir,
            patch=state.candidate_patch,
            dataset=self.dataset,
            eval_fn=self.eval_fn,
            target_factory=self.target_factory,
            baseline_score=state.current_score,
            n_samples=n_samples,
            seed=seed,
            layer_context=self.layer_context,
        )
        if result.accepted:
            self.target = self.target_factory(self.source_dir)
            self.sys_desc.source_files = collect_sources(self.source_dir)

        # 生成结构化反馈
        feedback_llm_calls = 0
        if result.details and state.baseline_details:
            result.feedback = generate_eval_feedback(
                result, state.baseline_details, model=self.model
            )
            if result.feedback:
                feedback_llm_calls = 1

        return ActionResult(
            action="evaluate_patch",
            ok=True,
            summary=f"{'ACCEPTED' if result.accepted else 'REJECTED'} {result.before_score:.2f} -> {result.after_score:.2f}",
            payload={"eval_result": result},
            eval_calls=1,
            llm_calls=feedback_llm_calls,
        )

    def _do_parallel_optimize(self, params: dict, state: PlannerState) -> ActionResult:
        """并行多 pattern 多候选优化：Generate → Cascade Eval → Greedy Combine。"""
        if state.diagnosis is None or not state.diagnosis.failure_patterns:
            return ActionResult(
                action="parallel_optimize", ok=False, error="No diagnosis patterns"
            )

        n_patterns = min(
            int(params.get("n_patterns", 3)), len(state.diagnosis.failure_patterns)
        )
        n_variants = max(1, int(params.get("n_variants", 2)))
        temperatures = [0.0, 0.4, 0.7][:n_variants]
        patterns = state.diagnosis.failure_patterns[:n_patterns]

        log.info(
            "[ParallelOpt] Phase 1: Generating %d patterns × %d variants = %d candidates",
            n_patterns,
            n_variants,
            n_patterns * n_variants,
        )

        # --- Phase 1: Parallel Generation ---
        candidates: list[tuple[dict, "DeltaPatch"]] = []  # (pattern, patch)

        def _generate_one(pattern: dict, temp: float):
            single_diag = Diagnosis(
                failure_patterns=[pattern],
                summary=pattern.get("pattern", "unknown"),
                raw_analysis=state.diagnosis.raw_analysis if state.diagnosis else "",
            )
            traj_text = _sample_trajectories_for_optimizer(
                state.trajectories, pattern, n=3
            )
            patch = optimize_agentic(
                self.sys_desc,
                single_diag,
                model=self.model,
                past_attempts=_history_for_prompt(state.history),
                layer_context=self._layer_context_text(),
                max_tool_calls=self.optimizer_max_tool_calls,
                trajectory_samples=traj_text,
                temperature=temp,
            )
            return (pattern, patch)

        tasks = [(p, t) for p in patterns for t in temperatures]
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            futures = {pool.submit(_generate_one, p, t): (p, t) for p, t in tasks}
            for future in as_completed(futures):
                try:
                    pattern, patch = future.result()
                    if patch.diffs:
                        candidates.append((pattern, patch))
                except Exception:
                    log.warning("[ParallelOpt] Generation failed", exc_info=True)

        if not candidates:
            return ActionResult(
                action="parallel_optimize",
                ok=False,
                error="All candidates failed to generate valid patches",
                llm_calls=len(tasks) * 2,
            )

        log.info(
            "[ParallelOpt] Phase 2: Evaluating %d valid candidates", len(candidates)
        )

        # --- Phase 2: Parallel Cascade Evaluation ---
        # Stage 0: 静态冲突预检（标记信息，不淘汰）
        source_files = self.sys_desc.source_files
        n_smoke = 3
        n_medium = min(10, self.eval_default_samples)
        seed = int(params.get("seed", 44))
        import random as _random

        rng = _random.Random(seed)
        sampled = rng.sample(
            self.dataset, min(self.eval_default_samples, len(self.dataset))
        )

        scored: list[tuple[dict, "DeltaPatch", list, float, dict]] = []

        def _eval_one(pattern, patch):
            import shutil

            modified = apply_diffs_in_memory(source_files, patch.diffs)
            if not modified:
                return None
            temp_dir = write_to_temp_dir(modified, self.source_dir)
            try:
                target = self.target_factory(temp_dir)
                # Stage 1: Smoke
                smoke_result = self.eval_fn(target, sampled[:n_smoke])
                smoke_score = smoke_result["score"]
                if smoke_score < state.current_score * 0.8:
                    return None
                # Stage 2: Medium eval
                medium_result = self.eval_fn(target, sampled[:n_medium])
                medium_score = medium_result["score"]
                delta = medium_score - state.current_score
                if delta <= 0:
                    return None
                return (pattern, patch, modified, delta, medium_result)
            except Exception:
                log.warning("[ParallelOpt] Eval failed", exc_info=True)
                return None
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        with ThreadPoolExecutor(max_workers=min(6, len(candidates))) as pool:
            futures = {
                pool.submit(_eval_one, p, patch): (p, patch) for p, patch in candidates
            }
            for future in as_completed(futures):
                r = future.result()
                if r is not None:
                    scored.append(r)

        if not scored:
            return ActionResult(
                action="parallel_optimize",
                ok=True,
                summary=f"0/{len(candidates)} candidates passed evaluation",
                payload={"accepted": 0, "candidates_total": len(candidates)},
                llm_calls=len(tasks) * 2,
                eval_calls=1,
            )

        scored.sort(key=lambda x: -x[3])
        log.info(
            "[ParallelOpt] Phase 3: Greedy combining %d passing candidates", len(scored)
        )

        # --- Phase 3: Greedy Combination ---
        selected: list[tuple[dict, "DeltaPatch", list, float]] = []
        occupied: dict[str, set[tuple[int, int]]] = {}

        for pattern, patch, modified, delta, _ in scored:
            regions = extract_patch_regions(patch, source_files)
            conflicts = False
            for file_path, start, end in regions:
                if file_path in occupied:
                    for occ_s, occ_e in occupied[file_path]:
                        if start <= occ_e and end >= occ_s:
                            conflicts = True
                            break
                if conflicts:
                    break
            if not conflicts:
                selected.append((pattern, patch, modified, delta))
                for file_path, start, end in regions:
                    occupied.setdefault(file_path, set()).add((start, end))

        if not selected:
            return ActionResult(
                action="parallel_optimize",
                ok=True,
                summary="All candidates conflict with each other",
                payload={"accepted": 0},
                llm_calls=len(tasks) * 2,
                eval_calls=1,
            )

        # 尝试合并所有选中 patch
        if len(selected) == 1:
            # 单个 patch 直接提交
            _, patch, _, delta = selected[0]
            commit_to_source(
                apply_diffs_in_memory(source_files, patch.diffs),
                self.source_dir,
                layer_context=self.layer_context,
            )
            self.target = self.target_factory(self.source_dir)
            self.sys_desc.source_files = collect_sources(self.source_dir)

            # 生成反馈
            eval_result = EvalResult(
                before_score=state.current_score,
                after_score=state.current_score + delta,
                accepted=True,
                patch=patch,
                delta=delta,
            )
            feedback_cost = 0
            if state.baseline_details:
                eval_result.feedback = generate_eval_feedback(
                    eval_result, state.baseline_details, model=self.model
                )
                if eval_result.feedback:
                    feedback_cost = 1

            return ActionResult(
                action="parallel_optimize",
                ok=True,
                summary=f"1 patch accepted, delta=+{delta:.2f}",
                payload={
                    "accepted_count": 1,
                    "combined_delta": delta,
                    "patterns_fixed": [selected[0][0].get("pattern", "")],
                    "eval_result": eval_result,
                    "candidates_total": len(candidates),
                    "candidates_passing": len(scored),
                },
                llm_calls=len(tasks) * 2 + feedback_cost,
                eval_calls=1,
            )

        # 多 patch 组合
        combined = merge_patches(source_files, [s[1] for s in selected])
        if combined is None:
            # 合并失败，退化到单最优
            _, patch, _, delta = selected[0]
            commit_to_source(
                apply_diffs_in_memory(source_files, patch.diffs),
                self.source_dir,
                layer_context=self.layer_context,
            )
            self.target = self.target_factory(self.source_dir)
            self.sys_desc.source_files = collect_sources(self.source_dir)
            return ActionResult(
                action="parallel_optimize",
                ok=True,
                summary=f"Merge failed, best single patch accepted, delta=+{delta:.2f}",
                payload={
                    "accepted_count": 1,
                    "combined_delta": delta,
                    "patterns_fixed": [selected[0][0].get("pattern", "")],
                    "candidates_total": len(candidates),
                },
                llm_calls=len(tasks) * 2,
                eval_calls=1,
            )

        # Full eval 组合效果
        import shutil as _shutil

        temp_dir = write_to_temp_dir(combined, self.source_dir)
        try:
            target = self.target_factory(temp_dir)
            full_result = self.eval_fn(target, sampled)
            combined_delta = full_result["score"] - state.current_score

            if combined_delta > 0:
                # 组合提升，一次性提交
                commit_to_source(
                    combined, self.source_dir, layer_context=self.layer_context
                )
                self.target = self.target_factory(self.source_dir)
                self.sys_desc.source_files = collect_sources(self.source_dir)

                eval_result = EvalResult(
                    before_score=state.current_score,
                    after_score=state.current_score + combined_delta,
                    accepted=True,
                    delta=combined_delta,
                    details=full_result.get("details", []),
                )
                feedback_cost = 0
                if state.baseline_details:
                    eval_result.feedback = generate_eval_feedback(
                        eval_result, state.baseline_details, model=self.model
                    )
                    if eval_result.feedback:
                        feedback_cost = 1

                return ActionResult(
                    action="parallel_optimize",
                    ok=True,
                    summary=f"{len(selected)} patches combined, delta=+{combined_delta:.2f}",
                    payload={
                        "accepted_count": len(selected),
                        "combined_delta": combined_delta,
                        "patterns_fixed": [s[0].get("pattern", "") for s in selected],
                        "eval_result": eval_result,
                        "candidates_total": len(candidates),
                        "candidates_passing": len(scored),
                    },
                    llm_calls=len(tasks) * 2 + feedback_cost,
                    eval_calls=1,
                )
            else:
                # 组合退化 → 逐个尝试，提交最优单个
                best_patch, best_delta, best_pattern = None, 0, None
                for pattern, patch, modified, delta in selected:
                    if delta > best_delta:
                        best_delta = delta
                        best_patch = patch
                        best_pattern = pattern

                if best_patch is not None:
                    commit_to_source(
                        apply_diffs_in_memory(source_files, best_patch.diffs),
                        self.source_dir,
                        layer_context=self.layer_context,
                    )
                    self.target = self.target_factory(self.source_dir)
                    self.sys_desc.source_files = collect_sources(self.source_dir)
                    return ActionResult(
                        action="parallel_optimize",
                        ok=True,
                        summary=f"Combo degraded, best single accepted, delta=+{best_delta:.2f}",
                        payload={
                            "accepted_count": 1,
                            "combined_delta": best_delta,
                            "patterns_fixed": [best_pattern.get("pattern", "")],
                            "combo_failed": True,
                        },
                        llm_calls=len(tasks) * 2,
                        eval_calls=1,
                    )
                return ActionResult(
                    action="parallel_optimize",
                    ok=True,
                    summary="Combo degraded, no single patch viable",
                    payload={"accepted_count": 0, "combo_failed": True},
                    llm_calls=len(tasks) * 2,
                    eval_calls=1,
                )
        finally:
            _shutil.rmtree(temp_dir, ignore_errors=True)

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
            child_accepted = int(child_result.get("accepted", 0) or 0)
            child_spawn_restart = bool(child_result.get("spawn_restart"))
            # 子层即使本层 accepted=0，也可能通过更深层 spawn 修改了 noa/
            noa_modified = child_accepted > 0 or child_spawn_restart
            return ActionResult(
                action="spawn_sublayer",
                ok=True,
                summary=f"L{child_level} done: score={child_result.get('final_score', 0):.2f}, "
                f"accepted={child_accepted}, spawn_restart={child_spawn_restart}, noa_modified={noa_modified}",
                payload={
                    "noa_modified": noa_modified,
                    "child_score": child_result.get("final_score", 0),
                    "child_accepted": child_accepted,
                    "child_spawn_restart": child_spawn_restart,
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
        ml1 = self.spawn_config.get("mini_l1", {})

        def child_target_factory(noa_source_dir):
            def target(question):
                result = run_layer_subprocess(
                    noa_dir=noa_source_dir,
                    project_root=project_root,
                    target_source_dir=self.source_dir,
                    dataset_pickle_path=dpp,
                    layer_level=1,
                    max_steps=ml1.get("max_steps", 8),
                    n_samples=ml1.get("n_samples", 10),
                    eval_n_samples=ml1.get("eval_n_samples", 10),
                    max_llm_calls=ml1.get("max_llm_calls", 40),
                    max_evals=ml1.get("max_evals", 4),
                    max_no_improve_steps=ml1.get("max_no_improve_steps", 3),
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

        # child score_fn: 归一化到目标分数
        reasonable_target = min(100, state.current_score + 20)

        def child_score_fn(prediction: str, ground_truth: str) -> float:
            try:
                pred = float(str(prediction).strip())
                target = float(str(ground_truth).strip())
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, pred / target)) if target > 0 else 0.0

        # child dataset: 虚拟数据条目（L2 评估的是 mini-L1 运行效果）
        child_dataset = [
            SimpleNamespace(question="opt_run_1", answer=str(reasonable_target)),
            SimpleNamespace(question="opt_run_2", answer=str(reasonable_target)),
            SimpleNamespace(question="opt_run_3", answer=str(reasonable_target)),
        ]

        # child layer_context
        structured_history = _build_parent_history(state)
        parent_summary = _build_parent_summary(state)

        child_layer_context = LayerContext(
            layer_id=f"L{child_level}",
            level=child_level,
            writable_root=noa_dir,
            readable_roots=[noa_dir, self.source_dir],
            parent_history=structured_history,
            parent_summary=parent_summary,
            max_depth=self.layer_context.max_depth,
            max_spawn_calls=max(0, self.layer_context.max_spawn_calls - 1),
        )

        log.info(
            f"[Spawn] Instantiating L{child_level} NOptimizer (source_dir={noa_dir})"
        )
        l2 = self.spawn_config.get("l2", {})
        child = NOptimizer(
            source_dir=noa_dir,
            target_factory=child_target_factory,
            dataset=child_dataset,
            eval_fn=child_eval_fn,
            max_steps=l2.get("max_steps", 12),
            n_samples=l2.get("n_samples", 2),
            eval_n_samples=l2.get("eval_n_samples", 2),
            model=self.model,
            score_fn=child_score_fn,
            max_llm_calls=l2.get("max_llm_calls", 60),
            max_evals=l2.get("max_evals", 8),
            max_no_improve_steps=l2.get("max_no_improve_steps", 4),
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


def _build_parent_history(state: PlannerState) -> list[dict]:
    """将 L1 history 聚合为 L2 to_prompt_context() 期望的迭代级结构。"""
    result = []
    iteration = 0
    current_diagnosis = ""
    current_rationale = ""
    current_diffs = []

    for h in state.history:
        action = h.get("action", "")
        payload = h.get("payload", {})

        if action == "analyze":
            current_diagnosis = payload.get("diagnosis_summary", "") or h.get(
                "summary", ""
            )

        elif action == "propose_patch":
            current_rationale = payload.get("rationale", "") or h.get("reason", "")
            current_diffs = payload.get("diff_summaries", [])

        elif action in ("evaluate_patch", "parallel_optimize"):
            iteration += 1
            result.append(
                {
                    "iteration": iteration,
                    "accepted": payload.get("accepted", False),
                    "before": payload.get("before", 0),
                    "after": payload.get("after", 0),
                    "diagnosis": current_diagnosis,
                    "rationale": current_rationale,
                    "diffs": current_diffs,
                    "error": h.get("error"),
                    "failure_reason": payload.get("eval_feedback", {}).get(
                        "next_focus", ""
                    ),
                }
            )
            current_diagnosis = ""
            current_rationale = ""
            current_diffs = []

    return result[-20:]


def _build_parent_summary(state: PlannerState) -> str:
    """生成 L1 优化历程摘要。"""
    initial = state.initial_observe_score or state.baseline_score
    current = state.current_score
    accepted = state.accepted_patches
    total_evals = state.budget.evals_used
    no_improve = state.no_improve_steps

    last_diagnosis = ""
    for h in reversed(state.history):
        if h.get("action") == "analyze" and h.get("ok"):
            last_diagnosis = h.get("payload", {}).get("diagnosis_summary", "")
            break

    rejected = [
        h.get("summary", "")[:100]
        for h in state.history
        if h.get("action") in ("evaluate_patch", "parallel_optimize")
        and not h.get("payload", {}).get("accepted", False)
    ]

    lines = [
        f"Initial: {initial:.2f}, Current: {current:.2f}, Delta: {current - initial:+.2f}",
        f"Accepted: {accepted}/{total_evals} evals, No-improve streak: {no_improve}",
    ]
    if last_diagnosis:
        lines.append(f"Bottleneck: {last_diagnosis[:300]}")
    if rejected:
        lines.append(f"Rejections ({len(rejected)}): " + "; ".join(rejected[-3:]))
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
