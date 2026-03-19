"""SkillOrchestrator：在 evolver 外部编排动态生成的 skills。

激活机制（类似 Claude Code tool selection）：
1. 把所有 skill 的 one-liner description 列给 LLM
2. LLM 根据当前状态选择激活哪些
3. 被选中的 skill 加载完整 guidance 注入 prompt
4. 没选中的 skill 只以 description 形式存在，不消耗 token

skill 生成受证据门控和 cadence 控制，不会在空库时盲目 bootstrap。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np

from .skill import GeneratedSkill, SkillAxis
from .skill_generator import SkillGenerator
from .skill_library import SkillLibrary
from .system_description import build_system_description
from .task_profile import TaskProfile
from .trajectory import OrchestratedResult, RunTrajectory, SegmentResult, SelectionEvent


_SELECT_SYSTEM = """你是一个进化优化的策略编排器。你的任务是从可用的 skill 列表中选择最适合当前状态的 skills。

规则：
- 选 0-3 个最相关的 skill（不要贪多）
- 如果当前状态良好且无明显问题，可以只选 1 个 strategy skill，或者不新增 skill
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
        self._last_generation_step = -999  # 上次成功生成 skill 的步数

    async def run(
        self,
        task,
        n_segments: int = 3,
        steps_per_segment: int = 10,
    ) -> OrchestratedResult:
        """分段运行 evolver，segment 内按小窗口动态刷新已激活的 skills。"""
        # 一次性构建系统描述
        self._sys_desc = build_system_description(task, self.profile)

        trajectory = RunTrajectory(
            task_name=task.name,
            task_profile=self.profile,
        )
        all_skills_used = set()
        prev_best = task.baseline_score
        stagnation = 0
        population = None
        error_counts = Counter()
        completed_steps = 0

        # Bootstrap：空库冷启动时生成种子 skill
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
                    traj_view,
                    prev_best,
                    stagnation,
                    seg_idx,
                    error_counts,
                    population,
                    total_steps_completed=completed_steps,
                )
                # 注入系统描述、data manifest、原始数据引用
                obs["_sys_desc"] = self._sys_desc
                obs["_manifest"] = _build_data_manifest(traj_view, population)
                obs["_trajectory"] = traj_view
                obs["_population"] = population

                # === 1. LLM 从 catalog 中选择 skill（或请求生成新的）===
                allow_generation = self._should_attempt_generation(obs)
                active_skills, generated_skills = await self._select_and_generate(
                    obs,
                    task.name,
                    completed_steps,
                    allow_generation=allow_generation,
                )
                active_ids = [s.skill_id for s in active_skills]
                generated_ids = [s.skill_id for s in generated_skills]
                all_skills_used.update(active_ids)
                if generated_skills:
                    self._last_generation_step = completed_steps

                # === 2. 合并 guidance → skill_context ===
                skill_context = _merge_guidance(active_skills)

                # === 3. 执行当前选择窗口 ===
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
                    generated_ids,
                )

                # === 4. 更新 evidence 和状态 ===
                improvement = window_result.best_score - prev_best
                for skill in active_skills:
                    self.library.record_activation(skill.skill_id, improvement)

                for rec in window_result.trajectory:
                    if rec.error_summary:
                        error_counts[_classify_error(rec.error_summary)] += 1

                if (
                    hasattr(window_result, "population")
                    and window_result.population is not None
                ):
                    population = window_result.population
                prev_best = max(prev_best, window_result.best_score)
                completed_steps += len(window_result.trajectory)
                segment_steps_completed += len(window_result.trajectory)

            # === 5. 提交完整 segment ===
            trajectory.segments.append(seg_result)
            if seg_result.best_score > segment_initial_best + 1e-6:
                stagnation = 0
            else:
                stagnation += 1

            print(
                f"  [Seg {seg_idx}] best={seg_result.best_score:.6f} "
                f"Δ={seg_result.improvement:+.6f} skills={seg_result.activated_skills}"
            )

        self.library.prune()
        trajectory.skills_used = sorted(all_skills_used)
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
    ) -> tuple[list[GeneratedSkill], list[GeneratedSkill]]:
        """用 LLM 从 catalog 选择 skill，必要时请求生成新的。"""
        catalog = self.library.get_catalog()
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

        # 收集被选中的 skill
        active = [
            self.library.skills[sid]
            for sid in selected_ids
            if sid in self.library.skills
        ][: self.max_active_skills]
        generated: list[GeneratedSkill] = []

        # 生成 LLM 要求的新 axis skill（每个窗口最多一个）
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
            new_skill = await self.generator.generate(axis, obs, task_name, generation)
            if new_skill:
                self.library.register(new_skill)
                active.append(new_skill)
                generated.append(new_skill)

        return active, generated

    @staticmethod
    def _merge_window_result(
        seg_result: SegmentResult,
        window_result: SegmentResult,
        step_offset: int,
        active_ids: list[str],
        generated_ids: list[str],
    ) -> None:
        """将一个小窗口的执行结果并入 segment。"""
        for rec in window_result.trajectory:
            rec.step += step_offset
            rec.activated_skills = list(active_ids)
            seg_result.trajectory.append(rec)

        seg_result.best_score = max(seg_result.best_score, window_result.best_score)
        seg_result.skill_context_used = seg_result.skill_context_used or bool(
            active_ids
        )
        seg_result.activated_skills = sorted(
            set(seg_result.activated_skills) | set(active_ids)
        )
        seg_result.selection_events.append(
            SelectionEvent(
                start_step=step_offset + 1,
                n_steps=len(window_result.trajectory),
                activated_skills=list(active_ids),
                generated_skills=list(generated_ids),
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

    async def _bootstrap_skills(self, task) -> None:
        """空库冷启动：生成 2 个种子 skill（REFLECTION + STRATEGY）。"""
        print("  [bootstrap] 空 skill 库，生成种子 skills...")
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
            "_sys_desc": self._sys_desc,
        }
        seed_axes = [SkillAxis.REFLECTION, SkillAxis.STRATEGY]
        for axis in seed_axes:
            try:
                skill = await self.generator.generate(
                    axis, obs, task.name, generation=0
                )
                if skill:
                    self.library.register(skill)
                    print(
                        f"  [bootstrap] 生成种子 skill: {skill.skill_id} — {skill.description[:80]}"
                    )
            except Exception as e:
                print(f"  [bootstrap] {axis.value} 生成失败: {e}")

    def _should_attempt_generation(self, obs: dict[str, Any]) -> bool:
        """放宽的门控：最小间隔替代严格 modulo，停滞时加速生成。"""
        if self.max_generate_per_window <= 0:
            return False

        completed_steps = int(obs.get("total_steps_completed", 0))
        interval = self.generation_interval_steps

        # 首次生成提前到 interval // 2（如 5 步而非 10 步）
        first_trigger = interval // 2
        if completed_steps < first_trigger:
            return False

        # 停滞时间隔减半
        stagnation = obs.get("stagnation", 0)
        effective_interval = interval // 2 if stagnation > 0 else interval

        steps_since_last = completed_steps - self._last_generation_step
        if steps_since_last < effective_interval:
            return False

        return self._has_generation_evidence(obs)

    @staticmethod
    def _has_generation_evidence(obs: dict[str, Any]) -> bool:
        """避免在完全无轨迹信息时生成 skill。"""
        if obs.get("recent_errors"):
            return True
        if obs.get("stagnation", 0) > 0:
            return True
        if obs.get("score_regressed"):
            return True
        if obs.get("score_trajectory"):
            return True
        pop_summary = obs.get("population_summary", {})
        if pop_summary.get("size", 0) > 0:
            return True
        return False

    async def _llm_select(
        self,
        catalog: list[dict],
        obs: dict,
        allow_generation: bool,
    ) -> tuple[list[str], list[str]]:
        """让 LLM 从 catalog 中选择 skill_ids + 请求生成的 axes。

        Returns: (selected_skill_ids, axes_to_generate)
        """
        if catalog:
            catalog_text = "\n".join(
                f"  [{i+1}] {item['skill_id']} ({item['axis']}): {item['description']}"
                + (
                    f" [avg_improvement={item['avg_improvement']:.4f}, "
                    f"activations={item['activations']}]"
                    if item["activations"] > 0
                    else ""
                )
                for i, item in enumerate(catalog)
            )
        else:
            catalog_text = "  [none]"

        if allow_generation:
            generation_block = """## 可生成的方向（最多生成 1 个）
- diagnosis: 适合错误频发、零分多、组件交互复杂或需要定位失败模式/瓶颈/根因的时候
- strategy: 适合当前方向基本正确但提升放缓，需要调搜索强度、explore/exploit 平衡或临近解精修的时候
- reflection: 适合连续停滞、当前范式明显卡死、需要大改思路或切换算法框架的时候

只有当现有 skills 明显不够用时才生成；如果证据不足或现有 skills 已覆盖问题，就让 generate 为空数组。"""
        else:
            generation_block = (
                "## 新 skill 生成\n"
                "当前不在生成窗口内，不允许生成新 skill。请让 generate 为空数组。"
            )

        dyn = obs.get("step_dynamics", {})
        # 诊断标签
        diagnostics: list[str] = []
        stagnant_ratio = dyn.get("stagnant_step_ratio", 0)
        decel = dyn.get("deceleration", 0)
        steps_since = dyn.get("steps_since_last_improvement", 0)
        if stagnant_ratio >= 0.8:
            diagnostics.append("CRITICAL: 近 20 步中 80%+ 无改进，严重停滞")
        elif stagnant_ratio >= 0.5:
            diagnostics.append("WARNING: 近 20 步中 50%+ 无改进，改进效率低")
        if decel > 0 and dyn.get("late_rate", 0) < dyn.get("early_rate", 1) * 0.2:
            diagnostics.append("WARNING: 改进速度已降至早期的 <20%，明显减速")
        if steps_since >= 10:
            diagnostics.append(f"WARNING: 已连续 {steps_since} 步无任何改进")

        dynamics_text = (
            f"- 总步数: {dyn.get('total_steps', 0)}\n"
            f"- 早期改进速率: {dyn.get('early_rate', 0):.6f}/步\n"
            f"- 近期改进速率: {dyn.get('late_rate', 0):.6f}/步\n"
            f"- 减速量: {decel:+.6f} (>0 = 在减速)\n"
            f"- 最近 20 步停滞比: {stagnant_ratio:.0%}\n"
            f"- 距上次改进: {steps_since} 步\n"
            f"- 近期 best-so-far: {dyn.get('recent_best_so_far', [])}"
        )
        diag_text = (
            "\n".join(f"  ⚠ {d}" for d in diagnostics)
            if diagnostics
            else "  （无异常）"
        )

        user_msg = f"""## 当前进化状态
- 任务: {obs.get('task_name', 'unknown')}
- 当前最高分: {obs.get('best_score', 0):.6f} (基线: {obs.get('baseline_score', 0):.6f})
- Segment: {obs.get('generation', 0)}
- 已完成步数: {obs.get('total_steps_completed', 0)}
- 错误率: {obs.get('error_rate', 0):.0%}
- 段级分数轨迹: {obs.get('score_trajectory', [])}
- 已有 skill 数: {obs.get('library_size', 0)}

## 步级动态指标
{dynamics_text}

## 自动诊断
{diag_text}

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
                f"allow_gen={allow_generation} selected={selected} generate={generate} "
                f"raw={response[:200]}"
            )
            # LLM 返回空/无效时 fallback 到 heuristic
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
        """LLM 失败时的 fallback：按 evidence 排序选 top。"""
        # 有 evidence 的按 avg_improvement 排序
        with_evidence = [c for c in catalog if c["activations"] > 0]
        with_evidence.sort(key=lambda c: c["avg_improvement"], reverse=True)

        if with_evidence:
            return [with_evidence[0]["skill_id"]], []

        # 全是新 skill，选最近生成的 strategy
        strategies = [c for c in catalog if c["axis"] == "strategy"]
        if strategies:
            return [strategies[-1]["skill_id"]], []

        if allow_generation:
            return [], ["strategy"]
        return [], []


def _merge_guidance(skills: list[GeneratedSkill]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for skill in skills:
        context[skill.axis.value] = {
            "skill_id": skill.skill_id,
            "formatted": skill.format_for_prompt(),
            **skill.guidance,
        }
    return context


def format_skill_context(skill_context: dict) -> str:
    """将 skill context 格式化为 prompt 中的可读文本。"""
    if not skill_context:
        return ""

    parts = ["## 策略指导\n"]
    parts.append("以下是基于当前进化状态的分析和策略建议：\n")

    for axis_name, data in skill_context.items():
        if isinstance(data, dict) and "formatted" in data:
            parts.append(data["formatted"])
        elif isinstance(data, dict):
            for key in (
                "idea",
                "failure_pattern",
                "fix_direction",
                "emphasis",
                "bottleneck",
            ):
                if data.get(key):
                    parts.append(f"- {data[key]}\n")

    return "\n".join(parts)


def _build_observation(
    task,
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
    for seg in trajectory.segments[-2:]:
        for rec in seg.trajectory:
            if rec.error_summary:
                recent_errors.append(rec.error_summary)

    total_evals = sum(len(seg.trajectory) for seg in trajectory.segments[-2:])
    error_rate = len(recent_errors) / max(total_evals, 1)
    score_regressed = (
        len(score_trajectory) >= 2 and score_trajectory[-1] < score_trajectory[-2]
    )

    pop_summary = {"size": 0, "mean_score": 0.0, "score_variance": 0.0}
    if trajectory.segments:
        last_scores = [
            r.score for r in trajectory.segments[-1].trajectory if r.score > 0
        ]
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

    # === 步级动态统计 ===
    step_dynamics = _compute_step_dynamics(trajectory, best_score)

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
    }


def _compute_step_dynamics(trajectory: RunTrajectory, current_best: float) -> dict:
    """从步级记录计算改进速度、减速率、停滞率等动态指标。"""
    # 收集所有步的 best-so-far 序列
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

    # best-so-far 序列
    best_so_far: list[float] = []
    running_best = 0.0
    for s in all_scores:
        running_best = max(running_best, s)
        best_so_far.append(running_best)

    n = len(best_so_far)

    # 每步是否产生了改进（相对提升 > 0.1%）
    improvements: list[bool] = [False]
    for i in range(1, n):
        rel_gain = (best_so_far[i] - best_so_far[i - 1]) / max(best_so_far[i - 1], 1e-9)
        improvements.append(rel_gain > 0.001)

    # 步级停滞率（最近 window 内没有改进的步占比）
    window = min(n, 20)
    recent_improvements = improvements[-window:]
    stagnant_ratio = 1.0 - sum(recent_improvements) / len(recent_improvements)

    # 距上次改进的步数
    steps_since_last = 0
    for imp in reversed(improvements):
        if imp:
            break
        steps_since_last += 1

    # 改进速度：前半段 vs 后半段的平均每步提升
    mid = n // 2
    if mid > 0 and n > mid:
        early_gain = best_so_far[mid] - best_so_far[0]
        late_gain = best_so_far[-1] - best_so_far[mid]
        early_rate = early_gain / mid
        late_rate = late_gain / (n - mid)
        deceleration = early_rate - late_rate  # >0 表示在减速
    else:
        early_rate = 0.0
        late_rate = 0.0
        deceleration = 0.0

    # 最近 10 步的 best-so-far（供 LLM 直接看趋势）
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
    """为 observation 构造一个包含当前 segment 部分轨迹的视图。"""
    segments = list(trajectory.segments)
    if current_segment is not None and current_segment.trajectory:
        segments.append(current_segment)
    return RunTrajectory(
        task_name=trajectory.task_name,
        task_profile=trajectory.task_profile,
        segments=segments,
        total_improvement=trajectory.total_improvement,
        skills_used=list(trajectory.skills_used),
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


# ============================================================
# Data Manifest：让 LLM 知道有哪些数据可看
# ============================================================


def _build_data_manifest(
    trajectory: RunTrajectory,
    population=None,
) -> list[tuple[str, str, str]]:
    """扫描可用数据 → (key, 描述, 数据量) 清单。"""
    manifest: list[tuple[str, str, str]] = []

    # error_traces
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

    # population_codes
    if population and population.size() > 0:
        manifest.append(
            (
                "population_codes",
                "种群所有个体的代码和分数",
                f"{population.size()} individuals",
            )
        )

    # best_vs_worst
    if population and population.size() >= 2:
        manifest.append(
            (
                "best_vs_worst",
                "最优 vs 最差个体的代码对比",
                "2 programs",
            )
        )

    # score_by_island
    last_seg = trajectory.segments[-1] if trajectory.segments else None
    if last_seg and last_seg.island_stats and len(last_seg.island_stats) > 1:
        manifest.append(
            (
                "score_by_island",
                "每个岛屿的分数分布统计",
                f"{len(last_seg.island_stats)} islands",
            )
        )

    # stagnation_history
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
    """根据 keys 展开详细数据为文本，预算上限 max_chars。"""
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
                    f"Island {isl_id}: size={stats['size']}, "
                    f"best={stats['best']:.6f}, mean={stats['mean']:.6f}"
                )
            _add("Score by Island", "\n".join(lines))

    if "stagnation_history" in keys:
        lines = []
        for seg in trajectory.segments:
            if seg.improvement <= 1e-6:
                for rec in seg.trajectory[-5:]:
                    line = (
                        f"step={rec.step} score={rec.score:.4f} Δ={rec.delta_score:.4f}"
                    )
                    if rec.error_summary:
                        line += f" err={rec.error_summary[:80]}"
                    lines.append(line)
        _add(
            "Stagnation History", "\n".join(lines[-20:]) if lines else "（无停滞记录）"
        )

    return "".join(parts)
