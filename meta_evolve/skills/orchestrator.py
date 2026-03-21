"""SkillOrchestrator：在 evolver 外部编排分层 skills / anti-skills。"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np

from ..tasks.profiles import TaskProfile
from ..tasks.system_description import build_system_description
from .filesystem import (
    STARTER_SKILL_ID,
    load_hook_for_skill,
    load_managed_skills,
    run_managed_hook,
)
from .generator import SkillGenerator
from .library import SkillLibrary
from .proposer import SkillProposer
from .materializer import materialize_skill
from .models import GeneratedSkill, SkillAxis, SkillLevel
from .trajectory import OrchestratedResult, RunTrajectory, SegmentResult, SelectionEvent


_SELECT_SYSTEM = """You are an evolution optimization strategy orchestrator. Your task is to select the most suitable skills from the available layered skill list for the current state.

Rules:
- Prioritize skills matching the current task profile / phase / diagnostic tags
- General skills represent principles, template skills represent "how to solve this type of problem", ephemeral skills represent "what to do this step"
- Select at most 0-3 positive skills (don't be greedy)
- Only request generating a new skill when evidence clearly shows existing skills are insufficient
- Request generating at most 1 new skill
- Output in JSON format"""


class SkillOrchestrator:
    """在 evolver 外部编排 skills 的执行。框架无关。"""

    def __init__(
        self,
        library: SkillLibrary,
        generator: SkillGenerator,
        task_profile: TaskProfile,
        adapter,
        llm=None,
        config: dict | None = None,
        fresh: bool = False,
    ):
        self.library = library
        self.generator = generator
        self.proposer = SkillProposer(llm or generator.llm)
        self.profile = task_profile
        self.adapter = adapter
        self.llm = llm or generator.llm
        self.config = config or {}
        orchestrator_cfg = self.config.get("orchestrator", {})
        self.max_active_skills = max(
            1, int(orchestrator_cfg.get("max_active_skills", 3))
        )
        self.generation_interval_iters = max(
            1, int(orchestrator_cfg.get("generation_interval_iters", 5))
        )
        self.max_generate_per_window = max(
            0, int(orchestrator_cfg.get("max_generate_per_window", 1))
        )
        self.max_active_anti_skills = max(
            1, int(orchestrator_cfg.get("max_active_anti_skills", 2))
        )
        self.force_generation_stagnation = max(
            1, int(orchestrator_cfg.get("force_generation_stagnation", 3))
        )
        self.batch_size = max(
            1, int(orchestrator_cfg.get("batch_size", 8))
        )
        self._last_generation_iter = -999
        self._active_search_policy: dict[str, Any] = {}
        self._managed_skills, self._skill_hooks = load_managed_skills(
            include_archived=not fresh,
        )
        # 试用期制 re-selection
        self.selection_trial_iters = max(
            1, int(orchestrator_cfg.get("selection_trial_iters", 3))
        )
        self._cached_selection: tuple[list, list, list, list] | None = None
        self._cached_skill_context: dict[str, Any] | None = None
        self._iters_since_last_select = 0

    async def run(
        self,
        task,
        n_iterations: int = 50,
    ) -> OrchestratedResult:
        self._sys_desc = build_system_description(task, self.profile)

        trajectory = RunTrajectory(
            task_name=task.name,
            task_profile=self.profile,
        )
        all_skills_used = set()
        all_anti_skills_used = set()
        prev_best = task.baseline_score
        stagnation = 0
        population = None
        error_counts = Counter()

        self._ensure_system_skills()

        # 单个 SegmentResult 用于内部追踪（兼容 trajectory 结构）
        seg_result = SegmentResult(
            best_score=prev_best,
            initial_score=prev_best,
        )

        for iteration in range(n_iterations):
            traj_view = _with_current_segment(trajectory, seg_result)
            obs = _build_observation(
                task,
                self.profile,
                self._sys_desc,
                traj_view,
                prev_best,
                stagnation,
                0,
                error_counts,
                population,
                total_iters_completed=iteration,
            )
            obs["_sys_desc"] = self._sys_desc
            obs["_manifest"] = _build_data_manifest(traj_view, population)
            obs["_trajectory"] = traj_view
            obs["_population"] = population
            if self._task_needs_starter(task.name):
                diag = obs.setdefault("diagnostic_packet", {})
                tags = list(diag.get("diagnostic_tags", []))
                if "starter_analysis" not in tags:
                    tags.append("starter_analysis")
                diag["diagnostic_tags"] = tags

            allow_generation = self._should_attempt_generation(obs)
            force_generation = (
                allow_generation
                and stagnation >= self.force_generation_stagnation
            )

            # 试用期制：每组 skill 至少跑 selection_trial_iters 轮
            trial_expired = self._iters_since_last_select >= self.selection_trial_iters
            needs_reselect = (
                self._cached_selection is None  # 首次
                or trial_expired              # 试用期到期
                or force_generation           # 强制生成打断试用期
            )

            if needs_reselect:
                (
                    active_skills,
                    active_anti_skills,
                    generated_skills,
                    generated_anti_skills,
                ) = await self._select_and_generate(
                    obs,
                    task.name,
                    iteration,
                    allow_generation=allow_generation if trial_expired or force_generation else False,
                    force_generation=force_generation,
                    force_skill_ids=(
                        [STARTER_SKILL_ID] if self._task_needs_starter(task.name) else []
                    ),
                )
                active_ids = [s.skill_id for s in active_skills]
                anti_ids = [s.skill_id for s in active_anti_skills]
                generated_ids = [s.skill_id for s in generated_skills]
                generated_anti_ids = [s.skill_id for s in generated_anti_skills]
                all_skills_used.update(active_ids)
                all_anti_skills_used.update(anti_ids)
                if generated_skills or generated_anti_skills:
                    self._last_generation_iter = iteration

                dynamic_guidance, search_policy = await self._resolve_skill_guidance(
                    active_skills,
                    task=task,
                    obs=obs,
                )
                skill_context = _merge_guidance(
                    active_skills,
                    active_anti_skills,
                    phase=obs["diagnostic_packet"].get("phase", "early"),
                    diagnostics=obs["diagnostic_packet"].get("diagnostic_tags", []),
                    dynamic_positive=dynamic_guidance,
                )

                # 缓存 + 重置试用期计数器
                self._cached_selection = (active_skills, active_anti_skills, generated_skills, generated_anti_skills)
                self._cached_skill_context = skill_context
                self._iters_since_last_select = 0
                if search_policy:
                    self._active_search_policy = search_policy
                    print(f"    [search-policy] switched to: {search_policy.get('name', '?')}")
            else:
                # 试用期内：复用缓存的 skill 选择，但允许 generation
                active_skills, active_anti_skills, _prev_gen, _prev_anti_gen = self._cached_selection
                skill_context = self._cached_skill_context
                active_ids = [s.skill_id for s in active_skills]
                anti_ids = [s.skill_id for s in active_anti_skills]
                generated_ids = []
                generated_anti_ids = []
                self._iters_since_last_select += 1
                remaining = self.selection_trial_iters - self._iters_since_last_select

                # 试用期内仍可生成新 skill（不改变当前选择）
                if allow_generation:
                    gen_axis_name = self._infer_generation_axis(obs)
                    gen_axis = SkillAxis(gen_axis_name)
                    new_skill = await self._generate_layered_skill(
                        gen_axis, obs, task.name, iteration,
                    )
                    if new_skill is not None:
                        self._last_generation_iter = iteration
                        generated_ids = [new_skill.skill_id]

                print(f"    [skill-select] iter={iteration} trial period ({remaining} iters remaining), reusing cached")

            iter_result = await self.adapter.run_iteration(
                task=task,
                batch_size=self.batch_size,
                skill_context=skill_context,
                population=population,
                initial_best_score=prev_best,
                iteration_label=f"{iteration+1}/{n_iterations}",
                search_policy=self._active_search_policy,
            )
            self._merge_window_result(
                seg_result,
                iter_result,
                iteration,
                active_ids,
                anti_ids,
                generated_ids,
                generated_anti_ids,
                obs["diagnostic_packet"],
            )

            improvement = iter_result.best_score - prev_best
            outcome_type = (
                "effective" if improvement > 1e-6
                else "regressive" if improvement < -1e-6
                else "neutral"
            )
            phase = obs["diagnostic_packet"].get("phase", "early")
            co_active = len(active_skills) + len(active_anti_skills)
            for skill in active_skills + active_anti_skills:
                self.library.record_activation(
                    skill.skill_id,
                    improvement,
                    task_name=task.name,
                    phase=phase,
                    stagnation=stagnation,
                    error_rate=obs.get("error_rate", 0.0),
                    co_active_count=co_active,
                    outcome_type=outcome_type,
                )

            for rec in iter_result.trajectory:
                if rec.error_summary:
                    error_counts[_classify_error(rec.error_summary)] += 1
                if rec.outcome_type:
                    error_counts[rec.outcome_type] += 1

            if (
                hasattr(iter_result, "population")
                and iter_result.population is not None
            ):
                population = iter_result.population
            new_best = max(prev_best, iter_result.best_score)
            if new_best > prev_best + 1e-6:
                stagnation = 0
            else:
                stagnation += 1
            prev_best = new_best

        trajectory.segments.append(seg_result)
        self.library.prune()
        trajectory.skills_used = sorted(all_skills_used)
        trajectory.anti_skills_used = sorted(all_anti_skills_used)
        trajectory.total_improvement = prev_best - task.baseline_score

        return OrchestratedResult(
            trajectory=trajectory,
            evidence_snapshot=self.library.get_evidence_summary(),
            final_best_score=prev_best,
        )

    async def _select_and_generate(
        self,
        obs: dict,
        task_name: str,
        generation: int,
        allow_generation: bool,
        force_generation: bool = False,
        force_skill_ids: list[str] | None = None,
    ) -> tuple[list[GeneratedSkill], list[GeneratedSkill], list[GeneratedSkill], list[GeneratedSkill]]:
        diag = obs.get("diagnostic_packet", {})
        task_tags = diag.get("task_tags", [])
        phase = diag.get("phase", "early")
        diagnostic_tags = diag.get("diagnostic_tags", [])

        context_candidates = self.library.retrieve_for_context(
            task_name=task_name,
            task_tags=task_tags,
            phase=phase,
            diagnostics=diagnostic_tags,
            top_k=max(self.max_active_skills * 3, 6),
        )
        catalog = [
            item
            for item in self.library.get_catalog()
            if item["skill_id"] in {s.skill_id for s in context_candidates}
        ]
        obs["skill_history"] = self.library.get_skill_history(task_name)
        obs["library_size"] = len(catalog)

        selected_ids: list[str] = []
        generate_axes: list[str] = []
        edit_target_id: str | None = None
        if catalog or allow_generation:
            selected_ids, generate_axes, edit_target_id = await self._llm_select(
                catalog,
                obs,
                allow_generation=allow_generation,
                force_generation=force_generation,
            )
        if force_generation and not generate_axes and not edit_target_id:
            axis = self._infer_generation_axis(obs)
            generate_axes = [axis]
            print(
                f"    [skill-gen] force_generation triggered (stagnation={obs.get('stagnation', 0)}), injecting axis={axis}"
            )

        active: list[GeneratedSkill] = []
        ordered_ids = list(force_skill_ids or []) + selected_ids
        seen_ids: set[str] = set()
        for sid in ordered_ids:
            if sid in seen_ids or sid not in self.library.skills:
                continue
            active.append(self.library.skills[sid])
            seen_ids.add(sid)
            if len(active) >= self.max_active_skills:
                break
        active_anti = self.library.retrieve_anti_skills(
            task_name=task_name,
            task_tags=task_tags,
            phase=phase,
            diagnostics=diagnostic_tags,
            top_k=self.max_active_anti_skills,
        )
        generated: list[GeneratedSkill] = []
        generated_anti: list[GeneratedSkill] = []

        # 处理 edit（优先于 generate）
        if edit_target_id and edit_target_id in self.library.skills:
            target_skill = self.library.skills[edit_target_id]
            print(f"    [skill-edit] LLM chose to edit: {edit_target_id}")
            obs["_edit_target"] = target_skill
            edit_proposal = await self.proposer.propose(target_skill.axis, target_skill.level, obs)
            edited = await self.generator.edit(target_skill, obs, task_name, generation,
                                               proposal=edit_proposal)
            if edited:
                self._register_generated_skill(edited)
                active.append(edited)
                generated.append(edited)
                print(f"    [skill-edit] created edited skill: {edited.skill_id}")
            # edit 和 generate 互斥，edit 优先
            generate_axes = []

        if allow_generation:
            generate_axes = generate_axes[: self.max_generate_per_window]
        else:
            generate_axes = []

        for axis_name in generate_axes:
            if len(active) >= self.max_active_skills:
                break
            try:
                axis = SkillAxis(axis_name)
            except ValueError:
                continue
            new_skill = await self._generate_layered_skill(axis, obs, task_name, generation)
            if new_skill:
                self._register_generated_skill(new_skill)
                active.append(new_skill)
                generated.append(new_skill)
            if _should_generate_anti_skill(obs):
                anti = await self.generator.generate_anti_skill(
                    axis,
                    obs,
                    task_name,
                    generation,
                    source_skill_ids=[new_skill.skill_id] if new_skill else [],
                )
                if anti:
                    self._register_generated_skill(anti)
                    active_anti = (active_anti + [anti])[: self.max_active_anti_skills]
                    generated_anti.append(anti)

        return active, active_anti, generated, generated_anti

    def _ensure_system_skills(self) -> None:
        for managed in self._managed_skills:
            existing = self.library.skills.get(managed.skill_id)
            if existing is None:
                self.library.register(managed)
                continue
            existing.description = managed.description
            existing.guidance = managed.guidance
            existing.axis = managed.axis
            existing.level = managed.level
            existing.polarity = managed.polarity
            existing.applicable_stages = managed.applicable_stages
            existing.trigger_diagnostics = managed.trigger_diagnostics
            existing.hook_name = managed.hook_name
            existing.hook_mode = managed.hook_mode
            existing.skill_path = managed.skill_path
            existing.hook_path = managed.hook_path
            existing.hook_entrypoint = managed.hook_entrypoint
            existing.protected = managed.protected

    def _register_generated_skill(self, skill: GeneratedSkill) -> None:
        materialized = materialize_skill(skill)
        self.library.register(materialized)
        hook = load_hook_for_skill(materialized)
        if hook is not None:
            self._skill_hooks[materialized.skill_id] = hook

    def _task_needs_starter(self, task_name: str) -> bool:
        return not self.library.has_task_artifact(task_name, STARTER_SKILL_ID)

    async def _resolve_skill_guidance(
        self,
        active_skills: list[GeneratedSkill],
        *,
        task,
        obs: dict,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        search_policy: dict[str, Any] = {}
        for skill in active_skills:
            if not skill.is_executable or not skill.hook_name:
                continue
            hook_result = await self._execute_skill_hook_full(skill, task=task, obs=obs)
            if hook_result is None:
                continue
            guidance = hook_result.get("guidance")
            if isinstance(guidance, dict):
                resolved[skill.skill_id] = guidance
            policy = hook_result.get("search_policy")
            if isinstance(policy, dict):
                search_policy.update(policy)
        return resolved, search_policy

    async def _execute_skill_hook_full(
        self,
        skill: GeneratedSkill,
        *,
        task,
        obs: dict,
    ) -> dict[str, Any] | None:
        """执行 hook 并返回完整结果（含 guidance + strategy_overrides）。"""
        if not skill.hook_name:
            return None

        artifact_key = skill.skill_id
        cached = self.library.get_task_artifact(task.name, artifact_key)
        if cached and skill.hook_mode == "task_once":
            if isinstance(cached, dict) and cached.get("guidance"):
                print(
                    f"    [skill-hook] reuse task artifact for {task.name} via {skill.skill_id}"
                )
                return cached

        hook = self._skill_hooks.get(skill.skill_id)
        if hook is None:
            return None

        result = await run_managed_hook(
            hook,
            {
                "skill": skill,
                "task": task,
                "obs": obs,
                "sys_desc": self._sys_desc,
                "llm": self.llm,
                "library": self.library,
            },
        )
        if not isinstance(result, dict):
            return None
        if skill.hook_mode == "task_once":
            self.library.set_task_artifact(task.name, artifact_key, result)
        return result

    async def _generate_layered_skill(
        self,
        axis: SkillAxis,
        obs: dict,
        task_name: str,
        generation: int,
    ) -> GeneratedSkill | None:
        """分层生成 skill，强制 general-first 约束。

        规则：
        - task/template 级 skill 必须从 general 衍生
        - 如果该 axis 没有 general skill，先生成 general 再衍生 template
        - 如果已有 general，直接衍生 template

        流程：proposer.propose() → proposer.brainstorm() → generator（容错回退）
        """
        # 查该 axis 已有的 general skills
        general_refs = self.library.retrieve(
            axis=axis,
            level=SkillLevel.GENERAL,
            top_k=3,
        )

        # 没有 general → 先生成 general
        if not general_refs:
            print(f"    [skill-gen] no general skill for axis={axis.value}, generating general first")
            proposal = await self.proposer.propose(axis, SkillLevel.GENERAL, obs)
            brainstorm = await self.proposer.brainstorm(axis, SkillLevel.GENERAL, obs, proposal) if proposal else None
            general = await self.generator.distill_general(axis, obs, task_name, generation,
                                                           proposal=proposal, brainstorm=brainstorm)
            if general:
                self._register_generated_skill(general)
                general_refs = [general]
                print(f"    [skill-gen] general created: {general.skill_id}")
            else:
                print(f"    [skill-gen] general generation failed, skipping")
                return None

        # 从 general 衍生 template（贴合当前 task）
        print(f"    [skill-gen] deriving template from general: {[s.skill_id for s in general_refs]}")
        proposal = await self.proposer.propose(axis, SkillLevel.TEMPLATE, obs)
        brainstorm = await self.proposer.brainstorm(axis, SkillLevel.TEMPLATE, obs, proposal) if proposal else None
        template = await self.generator.compile_template(
            axis,
            obs,
            task_name,
            generation,
            source_skill_ids=[s.skill_id for s in general_refs],
            source_skills=general_refs,
            proposal=proposal,
            brainstorm=brainstorm,
        )
        return template

    @staticmethod
    def _merge_window_result(
        seg_result: SegmentResult,
        window_result: SegmentResult,
        step_offset: int,
        active_ids: list[str],
        anti_ids: list[str],
        generated_ids: list[str],
        generated_anti_ids: list[str],
        diagnostic_packet: dict,
    ) -> None:
        for rec in window_result.trajectory:
            rec.step += step_offset
            rec.activated_skills = list(active_ids)
            rec.active_anti_skills = list(anti_ids)
            rec.diagnostic_tags = list(diagnostic_packet.get("diagnostic_tags", []))
            seg_result.trajectory.append(rec)

        seg_result.best_score = max(seg_result.best_score, window_result.best_score)
        seg_result.skill_context_used = seg_result.skill_context_used or bool(
            active_ids or anti_ids
        )
        seg_result.activated_skills = sorted(
            set(seg_result.activated_skills) | set(active_ids)
        )
        seg_result.active_anti_skills = sorted(
            set(seg_result.active_anti_skills) | set(anti_ids)
        )
        seg_result.selection_events.append(
            SelectionEvent(
                iteration=step_offset,
                batch_size=len(window_result.trajectory),
                activated_skills=list(active_ids),
                active_anti_skills=list(anti_ids),
                generated_skills=list(generated_ids),
                generated_anti_skills=list(generated_anti_ids),
            )
        )

        if window_result.population_snapshot is not None:
            seg_result.population_snapshot = window_result.population_snapshot
        if window_result.island_stats is not None:
            seg_result.island_stats = window_result.island_stats
        if window_result.error_details:
            if seg_result.error_details is None:
                seg_result.error_details = []
            for detail in window_result.error_details:
                shifted = dict(detail)
                shifted["step"] = shifted.get("step", 0) + step_offset
                seg_result.error_details.append(shifted)
        seg_result.diagnostic_packet = diagnostic_packet

    @staticmethod
    def _infer_generation_axis(obs: dict[str, Any]) -> str:
        diag = obs.get("diagnostic_packet", {})
        tags = set(diag.get("diagnostic_tags", []))
        error_rate = obs.get("error_rate", 0.0)
        stagnation = obs.get("stagnation", 0)
        # 错误率高 → 需要诊断失败原因
        if error_rate > 0.3 or tags & {"syntax_failures", "runtime_failures"}:
            return "diagnosis"
        # 长期停滞 → 需要换视角
        if stagnation >= 5 or tags & {"convergence", "low_diversity"}:
            return "reflection"
        # 默认 → 调整搜索策略
        return "strategy"

    def _should_attempt_generation(self, obs: dict[str, Any]) -> bool:
        if self.max_generate_per_window <= 0:
            return False

        completed_iters = int(obs.get("total_iters_completed", 0))
        interval = self.generation_interval_iters

        # 首次生成在 interval 个 iteration 之后
        if completed_iters < interval:
            return False

        iters_since_last = completed_iters - self._last_generation_iter
        if iters_since_last < interval:
            return False

        return self._has_generation_evidence(obs)

    @staticmethod
    def _has_generation_evidence(obs: dict[str, Any]) -> bool:
        diag = obs.get("diagnostic_packet", {})
        if obs.get("recent_errors"):
            return True
        if obs.get("stagnation", 0) > 0:
            return True
        if obs.get("score_regressed"):
            return True
        if obs.get("score_trajectory"):
            return True
        if diag.get("failure_breakdown"):
            return True
        pop_summary = obs.get("population_summary", {})
        return pop_summary.get("size", 0) > 0

    async def _llm_select(
        self,
        catalog: list[dict],
        obs: dict,
        allow_generation: bool,
        force_generation: bool = False,
    ) -> tuple[list[str], list[str], str | None]:
        if catalog:
            lines = []
            for i, item in enumerate(catalog):
                line = f"  [{i+1}] {item['skill_id']} ({item['level']}/{item['axis']}): {item['description']}"
                if item["activations"] > 0:
                    line += (
                        f" [avg_imp={item['avg_improvement']:.4f}, "
                        f"activations={item['activations']}, "
                        f"outcomes={item.get('outcome_summary', 'n/a')}]"
                    )
                lines.append(line)
            catalog_text = "\n".join(lines)
        else:
            catalog_text = "  [none]"

        if force_generation:
            generation_block = """## Available Generation Axes (max 1) — GENERATION STRONGLY RECOMMENDED
The population has been stagnating for multiple iterations. Existing skills have failed to break through.
You SHOULD generate a new skill to introduce fresh approaches. Pick the most suitable axis:
- diagnosis: suitable when errors are frequent, many zero scores, need to locate failure patterns/bottlenecks/root causes
- strategy: suitable when current direction is correct but improvement slowing, need to adjust search intensity or convergence
- reflection: suitable when continuously stagnating, current paradigm clearly stuck, need to switch algorithmic perspective"""
        elif allow_generation:
            generation_block = """## Available Generation Axes (max 1)
- diagnosis: suitable when errors are frequent, many zero scores, need to locate failure patterns/bottlenecks/root causes
- strategy: suitable when current direction is correct but improvement slowing, need to adjust search intensity or convergence
- reflection: suitable when continuously stagnating, current paradigm clearly stuck, need to switch algorithmic perspective

Only generate when existing skills are clearly insufficient; if evidence is lacking or existing skills already cover the problem, leave generate as empty array."""
        else:
            generation_block = (
                "## New Skill Generation\n"
                "Not in generation window, new skill generation not allowed. Leave generate as empty array."
            )

        # edit 选项（任何时候都可以 edit）
        generation_block += """

## Edit Existing Skill (alternative to generating new)
If an existing skill SHOULD have addressed the current failures but didn't,
you can edit it instead of creating a new one. This avoids skill bloat.
- Set "edit": "skill_id_to_edit" to indicate which skill to improve
- Only edit when you can identify a specific gap in the skill's guidance
- Prefer edit over create when the skill's core idea is sound but its coverage is incomplete
- Set "edit": null if no edit is needed"""

        dyn = obs.get("step_dynamics", {})
        diag = obs.get("diagnostic_packet", {})
        diagnostics: list[str] = list(diag.get("diagnostic_tags", []))
        stagnant_ratio = dyn.get("stagnant_ratio", 0)
        if stagnant_ratio >= 0.8:
            diagnostics.append("severe_stagnation")
        elif stagnant_ratio >= 0.5:
            diagnostics.append("moderate_stagnation")

        user_msg = f"""## Current Evolution State
- Task: {obs.get('task_name', 'unknown')}
- Current Best Score: {obs.get('best_score', 0):.6f} (baseline: {obs.get('baseline_score', 0):.6f})
- Segment: {obs.get('generation', 0)}
- Completed Iterations: {obs.get('total_iters_completed', 0)}
- Task Profile: {diag.get('task_tags', [])}
- Current Phase: {diag.get('phase', 'early')}
- Diagnostic Tags: {diagnostics}
- Failure Distribution: {diag.get('failure_breakdown', {})}
- Population State: {diag.get('population', {})}

## Iteration-level Dynamics (each iteration = batch_size parallel candidates)
- Total iterations: {dyn.get('total_iters', 0)}, Total evaluations: {dyn.get('total_evals', 0)}
- Early improvement rate: {dyn.get('early_rate', 0):.6f}/iteration
- Recent improvement rate: {dyn.get('late_rate', 0):.6f}/iteration
- Deceleration: {dyn.get('deceleration', 0):+.6f}
- Recent stagnation ratio: {dyn.get('stagnant_ratio', 0):.0%}
- Iterations since last improvement: {dyn.get('iters_since_last_improvement', 0)} iterations

## Available Skills
{catalog_text}

{generation_block}

Select 0-{self.max_active_skills} most suitable skills, output JSON:
{{"select": ["skill_id_1", "skill_id_2"], "generate": ["axis_if_needed"], "edit": "skill_id_to_edit_or_null", "reason": "brief reason"}}"""

        try:
            response = await self.llm.generate(
                _SELECT_SYSTEM,
                user_msg,
                temperature=0.2,
                max_tokens=4096,
            )
            selected, generate, edit_id = self._parse_selection(
                response, {item["skill_id"] for item in catalog}
            )
            print(
                f"    [skill-select] iter={obs.get('total_iters_completed',0)} "
                f"allow_gen={allow_generation} selected={selected} generate={generate} edit={edit_id} raw={response[:200]}"
            )
            if not selected and not generate and not edit_id and catalog:
                selected, generate, edit_id = self._heuristic_select(catalog, allow_generation)
                print(
                    f"    [skill-select] heuristic fallback: selected={selected} generate={generate}"
                )
            return selected, generate, edit_id
        except Exception as e:
            print(
                f"    [skill-select] step={obs.get('total_iters_completed',0)} EXCEPTION: {e}"
            )
            return self._heuristic_select(catalog, allow_generation)

    @staticmethod
    def _parse_selection(
        response: str,
        valid_ids: set[str],
    ) -> tuple[list[str], list[str], str | None]:
        m = re.search(r"\{[\s\S]*\}", response)
        if not m:
            return [], [], None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return [], [], None

        selected = [sid for sid in data.get("select", []) if sid in valid_ids]
        generate = [
            a
            for a in data.get("generate", [])
            if a in ("reflection", "diagnosis", "strategy")
        ][:1]
        edit_id = data.get("edit")
        if isinstance(edit_id, str) and edit_id in valid_ids:
            pass  # valid edit target
        else:
            edit_id = None
        return selected, generate, edit_id

    @staticmethod
    def _heuristic_select(
        catalog: list[dict],
        allow_generation: bool,
    ) -> tuple[list[str], list[str], str | None]:
        by_score = sorted(
            catalog,
            key=lambda c: (
                c.get("avg_improvement", 0.0),
                c.get("activations", 0),
                1 if c.get("level") == "ephemeral" else 0,
            ),
            reverse=True,
        )
        if by_score:
            return [by_score[0]["skill_id"]], [], None
        if allow_generation:
            return [], ["strategy"], None
        return [], [], None


def _merge_guidance(
    skills: list[GeneratedSkill],
    anti_skills: list[GeneratedSkill],
    phase: str,
    diagnostics: list[str],
    dynamic_positive: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {"positive": {}, "negative": {}}
    dynamic_positive = dynamic_positive or {}
    for skill in skills:
        override = dynamic_positive.get(skill.skill_id)
        guidance = dict(skill.guidance)
        if override:
            guidance.update(override)
        context["positive"][skill.skill_id] = {
            "skill_id": skill.skill_id,
            "axis": skill.axis.value,
            "level": skill.level.value,
            "formatted": skill.format_for_prompt(
                phase,
                diagnostics,
                guidance_override=override,
            ),
            **guidance,
        }
    for skill in anti_skills:
        context["negative"][skill.skill_id] = {
            "skill_id": skill.skill_id,
            "axis": skill.axis.value,
            "level": skill.level.value,
            "formatted": skill.format_for_prompt(phase, diagnostics),
            **skill.guidance,
        }
    return context


def format_skill_context(skill_context: dict) -> str:
    if not skill_context:
        return ""

    parts = ["## Strategy Guidance\n", "Compiled skill context based on current evolution state:\n"]

    positives = skill_context.get("positive", {})
    negatives = skill_context.get("negative", {})
    if positives:
        parts.append("### Positive Skills")
        for data in positives.values():
            parts.append(data.get("formatted", ""))
    if negatives:
        parts.append("\n### Anti-Skills")
        for data in negatives.values():
            parts.append(data.get("formatted", ""))
    return "\n".join(parts)


def _build_observation(
    task,
    profile: TaskProfile,
    sys_desc,
    trajectory,
    best_score,
    stagnation,
    generation,
    error_counts,
    population=None,
    total_iters_completed: int = 0,
):
    score_trajectory = [seg.best_score for seg in trajectory.segments]
    recent_errors = []
    recent_records = []
    for seg in trajectory.segments[-2:]:
        recent_records.extend(seg.trajectory)
        for rec in seg.trajectory:
            if rec.error_summary:
                recent_errors.append(rec.error_summary)

    total_evals = max(len(recent_records), 1)
    error_rate = len(recent_errors) / total_evals
    score_regressed = (
        len(score_trajectory) >= 2 and score_trajectory[-1] < score_trajectory[-2]
    )

    pop_summary = {"size": 0, "mean_score": 0.0, "score_variance": 0.0}
    if population and hasattr(population, "scores"):
        pop_scores = population.scores()
        if len(pop_scores) > 0:
            pop_summary = {
                "size": len(pop_scores),
                "mean_score": float(np.mean(pop_scores)),
                "score_variance": float(np.var(pop_scores)),
                "percentile_25": float(np.percentile(pop_scores, 25)),
                "percentile_75": float(np.percentile(pop_scores, 75)),
                "min_score": float(np.min(pop_scores)),
                "max_score": float(np.max(pop_scores)),
            }
    elif trajectory.segments:
        last_scores = [r.score for r in trajectory.segments[-1].trajectory]
        if last_scores:
            pop_summary = {
                "size": len(last_scores),
                "mean_score": float(np.mean(last_scores)),
                "score_variance": float(np.var(last_scores)),
            }

    best_code = ""
    if population and hasattr(population, "best"):
        best_ind = population.best()
        if best_ind:
            best_code = best_ind.code

    step_dynamics = _compute_step_dynamics(trajectory, best_score)
    diagnostic_packet = _build_diagnostic_packet(
        task=task,
        profile=profile,
        sys_desc=sys_desc,
        trajectory=trajectory,
        recent_records=recent_records,
        error_counts=error_counts,
        pop_summary=pop_summary,
        step_dynamics=step_dynamics,
        stagnation=stagnation,
        score_regressed=score_regressed,
        population=population,
    )

    return {
        "task_name": task.name,
        "baseline_score": task.baseline_score,
        "best_score": best_score,
        "stagnation": stagnation,
        "generation": generation,
        "total_iters_completed": total_iters_completed,
        "score_trajectory": score_trajectory,
        "recent_errors": recent_errors[-10:],
        "error_rate": error_rate,
        "error_counts": dict(error_counts),
        "score_regressed": score_regressed,
        "population_summary": pop_summary,
        "best_code_snippet": best_code,
        "step_dynamics": step_dynamics,
        "diagnostic_packet": diagnostic_packet,
    }


def _build_diagnostic_packet(
    task,
    profile: TaskProfile,
    sys_desc,
    trajectory: RunTrajectory,
    recent_records: list,
    error_counts: Counter,
    pop_summary: dict,
    step_dynamics: dict,
    stagnation: int,
    score_regressed: bool,
    population=None,
) -> dict[str, Any]:
    failure_breakdown = Counter(error_counts)
    effective_edits = []
    regressive_edits = []
    no_effect_edits = []
    for rec in recent_records[-20:]:
        item = {
            "step": rec.step,
            "parent_id": rec.parent_id,
            "parent_score": rec.parent_score,
            "score": rec.score,
            "outcome": rec.outcome_type,
            "diff_summary": rec.diff_summary,
        }
        if rec.outcome_type in ("global_improvement", "local_improvement"):
            effective_edits.append(item)
        elif rec.outcome_type == "regression":
            regressive_edits.append(item)
        elif rec.outcome_type in ("no_effect", "no_change"):
            no_effect_edits.append(item)

    status = "healthy"
    convergence = pop_summary.get("score_variance", 0.0) < 1e-6 and pop_summary.get("size", 0) > 1
    if step_dynamics.get("stagnant_ratio", 0) >= 0.8 or stagnation > 0:
        status = "stagnating"
    elif convergence:
        status = "converging"
    elif score_regressed:
        status = "regressing"
    elif pop_summary.get("score_variance", 0.0) > 0.05:
        status = "diverse"

    phase = "early"
    total_steps = step_dynamics.get("total_evals", 0)
    if status == "stagnating":
        phase = "stalled"
    elif total_steps >= 12:
        phase = "late"
    elif total_steps >= 4:
        phase = "mid"

    diagnostic_tags: list[str] = []
    if status == "stagnating":
        diagnostic_tags.append("stagnation")
    if convergence:
        diagnostic_tags.append("convergence")
    if pop_summary.get("score_variance", 0.0) < 1e-4 and pop_summary.get("size", 0) > 1:
        diagnostic_tags.append("low_diversity")
    if failure_breakdown.get("syntax", 0) or failure_breakdown.get("syntax_error", 0):
        diagnostic_tags.append("syntax_failures")
    if failure_breakdown.get("runtime", 0) or failure_breakdown.get("runtime_error", 0):
        diagnostic_tags.append("runtime_failures")
    if regressive_edits:
        diagnostic_tags.append("regression")
    if no_effect_edits:
        diagnostic_tags.append("no_effect")
    if effective_edits:
        diagnostic_tags.append("has_effective_edits")

    return {
        "phase": phase,
        "task_tags": profile.tags(),
        "diagnostic_tags": diagnostic_tags,
        "evaluator": sys_desc.evaluator_contract,
        "population": {
            "status": status,
            "convergence": convergence,
            **pop_summary,
        },
        "failure_breakdown": dict(failure_breakdown),
        "effective_edits": effective_edits[-5:],
        "regressive_edits": regressive_edits[-5:],
        "no_effect_edits": no_effect_edits[-5:],
        "iters_since_last_improvement": step_dynamics.get("iters_since_last_improvement", 0),
        "task_name": task.name,
    }


def _compute_step_dynamics(trajectory: RunTrajectory, current_best: float) -> dict:
    """按 iteration 粒度计算动态指标。

    每个 iteration 产出 batch_size 个 sample，取该 iteration 的最高分作为代表。
    """
    # 收集所有 sample scores，按 iteration 分组
    all_scores: list[float] = []
    for seg in trajectory.segments:
        for rec in seg.trajectory:
            all_scores.append(rec.score)

    if not all_scores:
        return {
            "total_evals": 0,
            "total_iters": 0,
            "improvement_rate": 0.0,
            "deceleration": 0.0,
            "iters_since_last_improvement": 0,
            "stagnant_ratio": 0.0,
            "recent_scores": [],
        }

    # 按 iteration 分组（通过 rec.iteration 字段或按 step_records 的 iteration）
    iter_best: list[float] = []
    cur_iter_scores: list[float] = []
    cur_iter_id = None
    for seg in trajectory.segments:
        for rec in seg.trajectory:
            rec_iter = getattr(rec, "iteration", None)
            if rec_iter is not None and rec_iter != cur_iter_id:
                if cur_iter_scores:
                    iter_best.append(max(cur_iter_scores))
                cur_iter_scores = []
                cur_iter_id = rec_iter
            cur_iter_scores.append(rec.score)
    if cur_iter_scores:
        iter_best.append(max(cur_iter_scores))

    # fallback：如果没有 iteration 信息，按 batch_size=8 分组
    if len(iter_best) <= 1 and len(all_scores) > 1:
        batch_size = 8
        iter_best = []
        for i in range(0, len(all_scores), batch_size):
            chunk = all_scores[i:i + batch_size]
            iter_best.append(max(chunk))

    if len(iter_best) < 2:
        return {
            "total_evals": len(all_scores),
            "total_iters": len(iter_best),
            "improvement_rate": 0.0,
            "deceleration": 0.0,
            "iters_since_last_improvement": 0,
            "stagnant_ratio": 0.0,
            "recent_scores": iter_best,
        }

    # 按 iteration 计算 best_so_far
    best_so_far: list[float] = []
    running_best = iter_best[0]
    for s in iter_best:
        running_best = max(running_best, s)
        best_so_far.append(running_best)

    n = len(best_so_far)
    improvements: list[bool] = [False]
    for i in range(1, n):
        rel_gain = (best_so_far[i] - best_so_far[i - 1]) / max(best_so_far[i - 1], 1e-9)
        improvements.append(rel_gain > 0.001)

    window = min(n, 10)
    recent_improvements = improvements[-window:]
    stagnant_ratio = 1.0 - sum(recent_improvements) / len(recent_improvements)

    iters_since_last = 0
    for imp in reversed(improvements):
        if imp:
            break
        iters_since_last += 1

    mid = n // 2
    if mid > 0 and n > mid:
        early_gain = best_so_far[mid] - best_so_far[0]
        late_gain = best_so_far[-1] - best_so_far[mid]
        early_rate = early_gain / mid
        late_rate = late_gain / (n - mid)
        deceleration = early_rate - late_rate
    else:
        early_rate = 0.0
        late_rate = 0.0
        deceleration = 0.0

    return {
        "total_evals": len(all_scores),
        "total_iters": n,
        "improvement_rate": (best_so_far[-1] - best_so_far[0]) / n,
        "early_rate": round(early_rate, 6),
        "late_rate": round(late_rate, 6),
        "deceleration": round(deceleration, 6),
        "iters_since_last_improvement": iters_since_last,
        "stagnant_ratio": round(stagnant_ratio, 3),
        "recent_best_so_far": [round(s, 6) for s in best_so_far[-10:]],
    }


def _with_current_segment(
    trajectory: RunTrajectory,
    current_segment: SegmentResult | None,
) -> RunTrajectory:
    segments = list(trajectory.segments)
    if current_segment is not None and current_segment.trajectory:
        segments.append(current_segment)
    return RunTrajectory(
        task_name=trajectory.task_name,
        task_profile=trajectory.task_profile,
        segments=segments,
        total_improvement=trajectory.total_improvement,
        skills_used=list(trajectory.skills_used),
        anti_skills_used=list(trajectory.anti_skills_used),
    )


def _classify_error(error: str) -> str:
    err = error.lower()
    if "syntax" in err:
        return "syntax"
    if "timeout" in err:
        return "timeout"
    if "no_code" in err:
        return "extraction"
    if "memory" in err:
        return "memory"
    return "runtime"


def _should_generate_anti_skill(obs: dict[str, Any]) -> bool:
    diag = obs.get("diagnostic_packet", {})
    failure_breakdown = diag.get("failure_breakdown", {})
    if not failure_breakdown:
        return False
    total_failures = sum(failure_breakdown.values())
    if total_failures < 2:
        return False
    return any(
        key in failure_breakdown
        for key in ("regression", "runtime_error", "syntax_error", "no_effect")
    )


# ============================================================
# Data Manifest：让 LLM 知道有哪些数据可看
# ============================================================


def _build_data_manifest(
    trajectory: RunTrajectory,
    population=None,
) -> list[tuple[str, str, str]]:
    manifest: list[tuple[str, str, str]] = []

    error_count = 0
    for seg in trajectory.segments[-2:]:
        if seg.error_details:
            error_count += len(seg.error_details)
    if error_count > 0:
        manifest.append(
            (
                "error_traces",
                "Full error messages and context from recent failures",
                f"{error_count} errors",
            )
        )

    transition_count = 0
    for seg in trajectory.segments[-2:]:
        transition_count += len(seg.trajectory)
    if transition_count > 0:
        manifest.append(
            (
                "edit_outcomes",
                "Recent parent→child edit outcome attribution (improve/regress/no_effect)",
                f"{transition_count} transitions",
            )
        )

    if population and population.size() > 0:
        manifest.append(
            (
                "population_codes",
                "Code and scores of all population individuals",
                f"{population.size()} individuals",
            )
        )

    if population and population.size() >= 2:
        manifest.append(
            (
                "best_vs_worst",
                "Best vs worst individual code comparison",
                "2 programs",
            )
        )

    last_seg = trajectory.segments[-1] if trajectory.segments else None
    if last_seg and last_seg.island_stats and len(last_seg.island_stats) > 1:
        manifest.append(
            (
                "score_by_island",
                "Score distribution statistics per island",
                f"{len(last_seg.island_stats)} islands",
            )
        )

    stag_steps = 0
    for seg in trajectory.segments:
        if seg.improvement <= 1e-6:
            stag_steps += len(seg.trajectory)
    if stag_steps > 0:
        manifest.append(
            (
                "stagnation_history",
                "Results of all attempts during stagnation (scores, errors)",
                f"{stag_steps} attempts",
            )
        )

    return manifest


def _expand_data_request(
    keys: list[str],
    trajectory: RunTrajectory,
    population,
    max_chars: int = 32000,
) -> str:
    parts: list[str] = []
    budget = max_chars

    def _add(title: str, content: str):
        nonlocal budget
        if budget <= 0:
            return
        block = f"\n### {title}\n{content[:budget]}\n"
        parts.append(block)
        budget -= len(block)

    if "error_traces" in keys:
        lines = []
        for seg in trajectory.segments[-2:]:
            if seg.error_details:
                for e in seg.error_details[-6:]:
                    lines.append(
                        f"step={e['step']} island={e.get('island_id', '?')} "
                        f"score={e.get('score', 0):.4f}\n  error: {e['error_full']}"
                    )
        _add("Error Traces", "\n".join(lines) if lines else "(no error records)")

    if "edit_outcomes" in keys:
        lines = []
        for seg in trajectory.segments[-2:]:
            for rec in seg.trajectory[-10:]:
                lines.append(
                    f"step={rec.step} outcome={rec.outcome_type} parent={rec.parent_id} "
                    f"parent_score={rec.parent_score} score={rec.score:.4f}\n"
                    f"  diff={rec.diff_summary or 'n/a'}"
                )
        _add("Edit Outcomes", "\n".join(lines) if lines else "(no edit results)")

    if "population_codes" in keys and population:
        lines = []
        for ind in population.all_sorted()[:12]:
            lines.append(
                f"[island={ind.island_id} score={ind.score:.6f}]\n"
                f"```python\n{ind.code[:400]}\n```"
            )
        _add("Population Codes", "\n".join(lines))

    if "best_vs_worst" in keys and population and population.size() >= 2:
        sorted_inds = population.all_sorted()
        best, worst = sorted_inds[0], sorted_inds[-1]
        content = (
            f"**Best** (score={best.score:.6f}, island={best.island_id}):\n"
            f"```python\n{best.code[:600]}\n```\n\n"
            f"**Worst** (score={worst.score:.6f}, island={worst.island_id}):\n"
            f"```python\n{worst.code[:600]}\n```"
        )
        _add("Best vs Worst", content)

    if "score_by_island" in keys:
        last_seg = trajectory.segments[-1] if trajectory.segments else None
        if last_seg and last_seg.island_stats:
            lines = []
            for isl_id, stats in sorted(last_seg.island_stats.items()):
                lines.append(
                    f"Island {isl_id}: size={stats['size']}, best={stats['best']:.6f}, mean={stats['mean']:.6f}"
                )
            _add("Score by Island", "\n".join(lines))

    if "stagnation_history" in keys:
        lines = []
        for seg in trajectory.segments:
            if seg.improvement <= 1e-6:
                for rec in seg.trajectory[-5:]:
                    line = (
                        f"step={rec.step} score={rec.score:.4f} Δ={rec.delta_score:.4f} outcome={rec.outcome_type}"
                    )
                    if rec.error_summary:
                        line += f" err={rec.error_summary[:80]}"
                    lines.append(line)
        _add(
            "Stagnation History", "\n".join(lines[-20:]) if lines else "(no stagnation records)"
        )

    return "".join(parts)
