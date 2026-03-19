"""SkillOrchestrator：在 evolver 外部编排分层 skills / anti-skills。"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np

from .skill import GeneratedSkill, SkillAxis, SkillLevel, SkillPolarity
from .skill_generator import SkillGenerator
from .skill_library import SkillLibrary
from .system_description import build_system_description
from .task_profile import TaskProfile
from .trajectory import OrchestratedResult, RunTrajectory, SegmentResult, SelectionEvent


_SELECT_SYSTEM = """你是一个进化优化的策略编排器。你的任务是从可用的 layered skill 列表中选择最适合当前状态的 skills。

规则：
- 优先选择与当前 task profile / phase / 诊断标签匹配的 skill
- general skill 表示原则，template skill 表示“这类问题怎么做”，ephemeral skill 表示“这一步怎么做”
- 最多选 0-3 个正向 skills（不要贪多）
- 只有在当前证据明确显示现有 skill 不足时，才请求生成新 skill
- 最多只请求生成 1 个新 skill
- 输出 JSON 格式"""


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
    ):
        self.library = library
        self.generator = generator
        self.profile = task_profile
        self.adapter = adapter
        self.llm = llm or generator.llm
        self.config = config or {}
        orchestrator_cfg = self.config.get("orchestrator", {})
        self.max_active_skills = max(
            1, int(orchestrator_cfg.get("max_active_skills", 3))
        )
        self.selection_interval_steps = max(
            1, int(orchestrator_cfg.get("selection_interval_steps", 1))
        )
        self.generation_interval_steps = max(
            1, int(orchestrator_cfg.get("generation_interval_steps", 10))
        )
        self.max_generate_per_window = max(
            0, int(orchestrator_cfg.get("max_generate_per_window", 1))
        )
        self.max_active_anti_skills = max(
            1, int(orchestrator_cfg.get("max_active_anti_skills", 2))
        )
        self._last_generation_step = -999

    async def run(
        self,
        task,
        n_segments: int = 3,
        steps_per_segment: int = 10,
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
        completed_steps = 0

        if not self.library.skills:
            await self._bootstrap_skills(task)

        for seg_idx in range(n_segments):
            segment_initial_best = prev_best
            seg_result = SegmentResult(
                best_score=segment_initial_best,
                initial_score=segment_initial_best,
            )
            segment_steps_completed = 0

            while segment_steps_completed < steps_per_segment:
                traj_view = _with_current_segment(trajectory, seg_result)
                obs = _build_observation(
                    task,
                    self.profile,
                    self._sys_desc,
                    traj_view,
                    prev_best,
                    stagnation,
                    seg_idx,
                    error_counts,
                    population,
                    total_steps_completed=completed_steps,
                )
                obs["_sys_desc"] = self._sys_desc
                obs["_manifest"] = _build_data_manifest(traj_view, population)
                obs["_trajectory"] = traj_view
                obs["_population"] = population

                allow_generation = self._should_attempt_generation(obs)
                (
                    active_skills,
                    active_anti_skills,
                    generated_skills,
                    generated_anti_skills,
                ) = await self._select_and_generate(
                    obs,
                    task.name,
                    completed_steps,
                    allow_generation=allow_generation,
                )
                active_ids = [s.skill_id for s in active_skills]
                anti_ids = [s.skill_id for s in active_anti_skills]
                generated_ids = [s.skill_id for s in generated_skills]
                generated_anti_ids = [s.skill_id for s in generated_anti_skills]
                all_skills_used.update(active_ids)
                all_anti_skills_used.update(anti_ids)
                if generated_skills or generated_anti_skills:
                    self._last_generation_step = completed_steps

                skill_context = _merge_guidance(
                    active_skills,
                    active_anti_skills,
                    phase=obs["diagnostic_packet"].get("phase", "early"),
                    diagnostics=obs["diagnostic_packet"].get("diagnostic_tags", []),
                )

                window_steps = min(
                    self.selection_interval_steps,
                    steps_per_segment - segment_steps_completed,
                )
                window_result = await self.adapter.run_segment(
                    task=task,
                    n_steps=window_steps,
                    skill_context=skill_context,
                    population=population,
                    initial_best_score=prev_best,
                )
                self._merge_window_result(
                    seg_result,
                    window_result,
                    segment_steps_completed,
                    active_ids,
                    anti_ids,
                    generated_ids,
                    generated_anti_ids,
                    obs["diagnostic_packet"],
                )

                improvement = window_result.best_score - prev_best
                phase = obs["diagnostic_packet"].get("phase", "early")
                for skill in active_skills + active_anti_skills:
                    self.library.record_activation(
                        skill.skill_id,
                        improvement,
                        task_name=task.name,
                        phase=phase,
                    )

                for rec in window_result.trajectory:
                    if rec.error_summary:
                        error_counts[_classify_error(rec.error_summary)] += 1
                    if rec.outcome_type:
                        error_counts[rec.outcome_type] += 1

                if (
                    hasattr(window_result, "population")
                    and window_result.population is not None
                ):
                    population = window_result.population
                prev_best = max(prev_best, window_result.best_score)
                completed_steps += len(window_result.trajectory)
                segment_steps_completed += len(window_result.trajectory)

            trajectory.segments.append(seg_result)
            if seg_result.best_score > segment_initial_best + 1e-6:
                stagnation = 0
            else:
                stagnation += 1

            print(
                f"  [Seg {seg_idx}] best={seg_result.best_score:.6f} "
                f"Δ={seg_result.improvement:+.6f} skills={seg_result.activated_skills} anti={seg_result.active_anti_skills}"
            )

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
        if catalog or allow_generation:
            selected_ids, generate_axes = await self._llm_select(
                catalog,
                obs,
                allow_generation=allow_generation,
            )

        active = [
            self.library.skills[sid]
            for sid in selected_ids
            if sid in self.library.skills
        ][: self.max_active_skills]
        active_anti = self.library.retrieve_anti_skills(
            task_name=task_name,
            task_tags=task_tags,
            phase=phase,
            diagnostics=diagnostic_tags,
            top_k=self.max_active_anti_skills,
        )
        generated: list[GeneratedSkill] = []
        generated_anti: list[GeneratedSkill] = []

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
                self.library.register(new_skill)
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
                    self.library.register(anti)
                    active_anti = (active_anti + [anti])[: self.max_active_anti_skills]
                    generated_anti.append(anti)

        return active, active_anti, generated, generated_anti

    async def _generate_layered_skill(
        self,
        axis: SkillAxis,
        obs: dict,
        task_name: str,
        generation: int,
    ) -> GeneratedSkill | None:
        diag = obs.get("diagnostic_packet", {})
        phase = diag.get("phase", "early")
        if phase == "early":
            return await self.generator.distill_general(axis, obs, task_name, generation)
        if phase in ("mid", "stalled"):
            general_refs = self.library.retrieve(
                axis=axis,
                task_name=task_name,
                task_tags=diag.get("task_tags", []),
                phase=phase,
                diagnostics=diag.get("diagnostic_tags", []),
                level=SkillLevel.GENERAL,
                top_k=2,
            )
            return await self.generator.compile_template(
                axis,
                obs,
                task_name,
                generation,
                source_skill_ids=[s.skill_id for s in general_refs],
            )
        template_refs = self.library.retrieve(
            axis=axis,
            task_name=task_name,
            task_tags=diag.get("task_tags", []),
            phase=phase,
            diagnostics=diag.get("diagnostic_tags", []),
            level=SkillLevel.TEMPLATE,
            top_k=2,
        )
        return await self.generator.instantiate_ephemeral(
            axis,
            obs,
            task_name,
            generation,
            source_skill_ids=[s.skill_id for s in template_refs],
        )

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
                start_step=step_offset + 1,
                n_steps=len(window_result.trajectory),
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

    async def _bootstrap_skills(self, task) -> None:
        print("  [bootstrap] 空 skill 库，生成分层种子 skills...")
        obs = {
            "task_name": task.name,
            "baseline_score": task.baseline_score,
            "best_score": task.baseline_score,
            "stagnation": 0,
            "generation": 0,
            "total_steps_completed": 0,
            "score_trajectory": [],
            "recent_errors": [],
            "error_rate": 0,
            "population_summary": {"size": 0},
            "best_code_snippet": task.initial_code[:500],
            "step_dynamics": {"total_steps": 0},
            "diagnostic_packet": {
                "phase": "early",
                "task_tags": self.profile.tags(),
                "diagnostic_tags": ["bootstrap"],
                "failure_breakdown": {},
                "population": {"status": "cold_start"},
                "evaluator": self._sys_desc.evaluator_contract,
            },
            "_sys_desc": self._sys_desc,
        }
        seeds = [
            (SkillAxis.REFLECTION, SkillLevel.GENERAL),
            (SkillAxis.DIAGNOSIS, SkillLevel.TEMPLATE),
            (SkillAxis.STRATEGY, SkillLevel.TEMPLATE),
        ]
        for axis, level in seeds:
            try:
                skill = await self.generator.generate(
                    axis, obs, task.name, generation=0, level=level
                )
                if skill:
                    self.library.register(skill)
                    print(
                        f"  [bootstrap] 生成种子 skill: {skill.skill_id} — {skill.description[:80]}"
                    )
            except Exception as e:
                print(f"  [bootstrap] {axis.value}/{level.value} 生成失败: {e}")

    def _should_attempt_generation(self, obs: dict[str, Any]) -> bool:
        if self.max_generate_per_window <= 0:
            return False

        completed_steps = int(obs.get("total_steps_completed", 0))
        interval = self.generation_interval_steps
        first_trigger = interval // 2
        if completed_steps < first_trigger:
            return False

        stagnation = obs.get("stagnation", 0)
        effective_interval = interval // 2 if stagnation > 0 else interval

        steps_since_last = completed_steps - self._last_generation_step
        if steps_since_last < effective_interval:
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
    ) -> tuple[list[str], list[str]]:
        if catalog:
            catalog_text = "\n".join(
                f"  [{i+1}] {item['skill_id']} ({item['level']}/{item['axis']}): {item['description']}"
                + (
                    f" [avg_improvement={item['avg_improvement']:.4f}, activations={item['activations']}]"
                    if item["activations"] > 0
                    else ""
                )
                for i, item in enumerate(catalog)
            )
        else:
            catalog_text = "  [none]"

        if allow_generation:
            generation_block = """## 可生成的方向（最多生成 1 个）
- diagnosis: 适合错误频发、零分多、需要定位失败模式/瓶颈/根因的时候
- strategy: 适合当前方向基本正确但提升放缓，需要调搜索强度或收敛方式的时候
- reflection: 适合连续停滞、当前范式明显卡死、需要切换算法视角的时候

只有当现有 skills 明显不够用时才生成；如果证据不足或已有 skills 已覆盖问题，就让 generate 为空数组。"""
        else:
            generation_block = (
                "## 新 skill 生成\n"
                "当前不在生成窗口内，不允许生成新 skill。请让 generate 为空数组。"
            )

        dyn = obs.get("step_dynamics", {})
        diag = obs.get("diagnostic_packet", {})
        diagnostics: list[str] = list(diag.get("diagnostic_tags", []))
        stagnant_ratio = dyn.get("stagnant_step_ratio", 0)
        if stagnant_ratio >= 0.8:
            diagnostics.append("severe_stagnation")
        elif stagnant_ratio >= 0.5:
            diagnostics.append("moderate_stagnation")

        user_msg = f"""## 当前进化状态
- 任务: {obs.get('task_name', 'unknown')}
- 当前最高分: {obs.get('best_score', 0):.6f} (基线: {obs.get('baseline_score', 0):.6f})
- Segment: {obs.get('generation', 0)}
- 已完成步数: {obs.get('total_steps_completed', 0)}
- 任务画像: {diag.get('task_tags', [])}
- 当前阶段: {diag.get('phase', 'early')}
- 诊断标签: {diagnostics}
- 失败分布: {diag.get('failure_breakdown', {})}
- population 状态: {diag.get('population', {})}

## 步级动态指标
- 早期改进速率: {dyn.get('early_rate', 0):.6f}/步
- 近期改进速率: {dyn.get('late_rate', 0):.6f}/步
- 减速量: {dyn.get('deceleration', 0):+.6f}
- 最近 20 步停滞比: {dyn.get('stagnant_step_ratio', 0):.0%}
- 距上次改进: {dyn.get('steps_since_last_improvement', 0)} 步

## 可用 Skills
{catalog_text}

{generation_block}

请选择 0-{self.max_active_skills} 个最合适的 skill，输出 JSON:
{{"select": ["skill_id_1", "skill_id_2"], "generate": ["axis_if_needed"], "reason": "简短说明"}}"""

        try:
            response = await self.llm.generate(
                _SELECT_SYSTEM,
                user_msg,
                temperature=0.2,
                max_tokens=4096,
            )
            selected, generate = self._parse_selection(
                response, {item["skill_id"] for item in catalog}
            )
            print(
                f"    [skill-select] step={obs.get('total_steps_completed',0)} "
                f"allow_gen={allow_generation} selected={selected} generate={generate} raw={response[:200]}"
            )
            if not selected and not generate and catalog:
                selected, generate = self._heuristic_select(catalog, allow_generation)
                print(
                    f"    [skill-select] heuristic fallback: selected={selected} generate={generate}"
                )
            return selected, generate
        except Exception as e:
            print(
                f"    [skill-select] step={obs.get('total_steps_completed',0)} EXCEPTION: {e}"
            )
            return self._heuristic_select(catalog, allow_generation)

    @staticmethod
    def _parse_selection(
        response: str,
        valid_ids: set[str],
    ) -> tuple[list[str], list[str]]:
        m = re.search(r"\{[\s\S]*\}", response)
        if not m:
            return [], []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return [], []

        selected = [sid for sid in data.get("select", []) if sid in valid_ids]
        generate = [
            a
            for a in data.get("generate", [])
            if a in ("reflection", "diagnosis", "strategy")
        ][:1]
        return selected, generate

    @staticmethod
    def _heuristic_select(
        catalog: list[dict],
        allow_generation: bool,
    ) -> tuple[list[str], list[str]]:
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
            return [by_score[0]["skill_id"]], []
        if allow_generation:
            return [], ["strategy"]
        return [], []


def _merge_guidance(
    skills: list[GeneratedSkill],
    anti_skills: list[GeneratedSkill],
    phase: str,
    diagnostics: list[str],
) -> dict[str, Any]:
    context: dict[str, Any] = {"positive": {}, "negative": {}}
    for skill in skills:
        context["positive"][skill.skill_id] = {
            "skill_id": skill.skill_id,
            "axis": skill.axis.value,
            "level": skill.level.value,
            "formatted": skill.format_for_prompt(phase, diagnostics),
            **skill.guidance,
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

    parts = ["## 策略指导\n", "以下是基于当前进化状态编译出的技能上下文：\n"]

    positives = skill_context.get("positive", {})
    negatives = skill_context.get("negative", {})
    if positives:
        parts.append("### 正向 Skills")
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
    total_steps_completed: int = 0,
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
    if trajectory.segments:
        last_scores = [r.score for r in trajectory.segments[-1].trajectory if r.score > 0]
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
            best_code = best_ind.code[:500]

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
        "total_steps_completed": total_steps_completed,
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
    if step_dynamics.get("stagnant_step_ratio", 0) >= 0.8 or stagnation > 0:
        status = "stagnating"
    elif convergence:
        status = "converging"
    elif score_regressed:
        status = "regressing"
    elif pop_summary.get("score_variance", 0.0) > 0.05:
        status = "diverse"

    phase = "early"
    total_steps = step_dynamics.get("total_steps", 0)
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
        "steps_since_last_improvement": step_dynamics.get("steps_since_last_improvement", 0),
        "task_name": task.name,
    }


def _compute_step_dynamics(trajectory: RunTrajectory, current_best: float) -> dict:
    all_scores: list[float] = []
    for seg in trajectory.segments:
        for rec in seg.trajectory:
            all_scores.append(rec.score)

    if len(all_scores) < 2:
        return {
            "total_steps": len(all_scores),
            "improvement_rate": 0.0,
            "recent_improvement_rate": 0.0,
            "deceleration": 0.0,
            "steps_since_last_improvement": 0,
            "stagnant_step_ratio": 0.0,
            "recent_step_scores": [],
        }

    best_so_far: list[float] = []
    running_best = 0.0
    for s in all_scores:
        running_best = max(running_best, s)
        best_so_far.append(running_best)

    n = len(best_so_far)
    improvements: list[bool] = [False]
    for i in range(1, n):
        rel_gain = (best_so_far[i] - best_so_far[i - 1]) / max(best_so_far[i - 1], 1e-9)
        improvements.append(rel_gain > 0.001)

    window = min(n, 20)
    recent_improvements = improvements[-window:]
    stagnant_ratio = 1.0 - sum(recent_improvements) / len(recent_improvements)

    steps_since_last = 0
    for imp in reversed(improvements):
        if imp:
            break
        steps_since_last += 1

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

    recent_bsf = best_so_far[-10:]

    return {
        "total_steps": n,
        "improvement_rate": (best_so_far[-1] - best_so_far[0]) / n,
        "early_rate": round(early_rate, 6),
        "late_rate": round(late_rate, 6),
        "deceleration": round(deceleration, 6),
        "steps_since_last_improvement": steps_since_last,
        "stagnant_step_ratio": round(stagnant_ratio, 3),
        "recent_best_so_far": [round(s, 6) for s in recent_bsf],
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
                "最近失败的完整错误信息和上下文",
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
                "最近 parent→child 改动的结果归因（improve/regress/no_effect）",
                f"{transition_count} transitions",
            )
        )

    if population and population.size() > 0:
        manifest.append(
            (
                "population_codes",
                "种群所有个体的代码和分数",
                f"{population.size()} individuals",
            )
        )

    if population and population.size() >= 2:
        manifest.append(
            (
                "best_vs_worst",
                "最优 vs 最差个体的代码对比",
                "2 programs",
            )
        )

    last_seg = trajectory.segments[-1] if trajectory.segments else None
    if last_seg and last_seg.island_stats and len(last_seg.island_stats) > 1:
        manifest.append(
            (
                "score_by_island",
                "每个岛屿的分数分布统计",
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
                "停滞期间所有尝试的结果（分数、错误）",
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
        _add("Error Traces", "\n".join(lines) if lines else "（无错误记录）")

    if "edit_outcomes" in keys:
        lines = []
        for seg in trajectory.segments[-2:]:
            for rec in seg.trajectory[-10:]:
                lines.append(
                    f"step={rec.step} outcome={rec.outcome_type} parent={rec.parent_id} "
                    f"parent_score={rec.parent_score} score={rec.score:.4f}\n"
                    f"  diff={rec.diff_summary or 'n/a'}"
                )
        _add("Edit Outcomes", "\n".join(lines) if lines else "（无编辑结果）")

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
            f"**最优** (score={best.score:.6f}, island={best.island_id}):\n"
            f"```python\n{best.code[:600]}\n```\n\n"
            f"**最差** (score={worst.score:.6f}, island={worst.island_id}):\n"
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
            "Stagnation History", "\n".join(lines[-20:]) if lines else "（无停滞记录）"
        )

    return "".join(parts)
