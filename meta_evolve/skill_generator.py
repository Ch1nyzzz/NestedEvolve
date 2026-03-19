"""SkillGenerator：LLM 驱动的分层 Skill / Anti-Skill 动态生成器。"""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import LLMClient
from .skill import GeneratedSkill, SkillAxis, SkillLevel, SkillPolarity


_LAYER_SCHEMAS: dict[tuple[SkillAxis, SkillLevel], str] = {
    (SkillAxis.REFLECTION, SkillLevel.GENERAL): """输出 JSON:
{
  "description": "一句话描述这个 general skill",
  "principle": "跨任务适用的高层原则",
  "heuristics": ["启发1", "启发2"],
  "transfer_rule": "什么情况下可以迁移使用",
  "idea": "核心思路"
}""",
    (SkillAxis.REFLECTION, SkillLevel.TEMPLATE): """输出 JSON:
{
  "description": "一句话描述这个 template skill",
  "pattern": "这类问题的模式概括",
  "actions": ["模板动作1", "模板动作2"],
  "stage_hint": "适用阶段",
  "idea": "该模板背后的关键想法"
}""",
    (SkillAxis.REFLECTION, SkillLevel.EPHEMERAL): """输出 JSON:
{
  "description": "一句话描述这个 ephemeral skill",
  "objective": "当前窗口的即时目标",
  "actions": ["立刻执行动作1", "立刻执行动作2"],
  "avoid": ["应避免的做法"],
  "idea": "当前突破口"
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.GENERAL): """输出 JSON:
{
  "description": "一句话描述这个 general diagnosis skill",
  "principle": "跨任务有效的诊断原则",
  "heuristics": ["诊断启发1", "诊断启发2"],
  "transfer_rule": "何时复用这个诊断原则",
  "root_cause": "常见根因"
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.TEMPLATE): """输出 JSON:
{
  "description": "一句话描述这个 diagnosis template",
  "pattern": "这一类失败/停滞的模式",
  "actions": ["诊断动作1", "诊断动作2"],
  "stage_hint": "适用阶段",
  "root_cause": "模板级根因",
  "bottleneck": "模板级瓶颈",
  "fix_direction": "通常修复方向"
}""",
    (SkillAxis.DIAGNOSIS, SkillLevel.EPHEMERAL): """输出 JSON:
{
  "description": "一句话描述这个即时诊断 skill",
  "objective": "当前要验证/排查什么",
  "actions": ["立即诊断动作1", "立即诊断动作2"],
  "avoid": ["别再做什么"],
  "failure_pattern": "当前主要失败模式",
  "root_cause": "当前最可能的根因",
  "bottleneck": "当前瓶颈",
  "fix_direction": "下一步修复方向",
  "intermediate_insight": "从当前轨迹提取出的洞察"
}""",
    (SkillAxis.STRATEGY, SkillLevel.GENERAL): """输出 JSON:
{
  "description": "一句话描述这个 general strategy skill",
  "principle": "跨任务有效的搜索原则",
  "heuristics": ["策略启发1", "策略启发2"],
  "transfer_rule": "何时复用",
  "mode": "exploration / exploitation / balanced"
}""",
    (SkillAxis.STRATEGY, SkillLevel.TEMPLATE): """输出 JSON:
{
  "description": "一句话描述这个 strategy template",
  "pattern": "这一类任务或阶段的搜索模式",
  "actions": ["模板动作1", "模板动作2"],
  "stage_hint": "适用阶段",
  "mode": "exploration / exploitation / balanced",
  "adjustments": ["参数/搜索调整建议"]
}""",
    (SkillAxis.STRATEGY, SkillLevel.EPHEMERAL): """输出 JSON:
{
  "description": "一句话描述这个 strategy ephemeral skill",
  "objective": "当前窗口的搜索目标",
  "actions": ["立即执行动作1", "立即执行动作2"],
  "avoid": ["应避免的做法"],
  "mode": "exploration / exploitation / balanced",
  "adjustments": ["当前窗口策略调整"],
  "emphasis": "对 LLM 的具体搜索指导"
}""",
}

_NEGATIVE_SCHEMA = """输出 JSON:
{
  "description": "一句话描述这个 anti-skill",
  "failure_pattern": "当前反复出现的失败模式",
  "forbidden_actions": ["不要再做的事1", "不要再做的事2"],
  "replacement_actions": ["替代策略1", "替代策略2"],
  "rationale": "为什么这些动作应被禁止"
}"""


class SkillGenerator:
    """LLM 驱动的 Skill 动态生成器。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._id_counter = 0

    async def generate(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        level: SkillLevel | None = None,
        source_skill_ids: list[str] | None = None,
    ) -> GeneratedSkill | None:
        """生成正向 skill，默认根据观察推断其层级。"""
        skill_level = level or self._infer_level(observation)
        guidance = await self._generate_guidance(
            axis=axis,
            level=skill_level,
            observation=observation,
        )
        if guidance is None:
            guidance = self._heuristic_fallback(axis, observation, skill_level)
        return self._build_skill(
            axis=axis,
            level=skill_level,
            polarity=SkillPolarity.POSITIVE,
            guidance=guidance,
            observation=observation,
            task_name=task_name,
            generation=generation,
            source_skill_ids=source_skill_ids or [],
        )

    async def generate_anti_skill(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str] | None = None,
    ) -> GeneratedSkill | None:
        """基于失败模式生成 anti-skill。"""
        guidance = await self._generate_guidance(
            axis=axis,
            level=SkillLevel.EPHEMERAL,
            observation=observation,
            polarity=SkillPolarity.NEGATIVE,
        )
        if guidance is None:
            guidance = self._heuristic_negative(axis, observation)
        return self._build_skill(
            axis=axis,
            level=SkillLevel.EPHEMERAL,
            polarity=SkillPolarity.NEGATIVE,
            guidance=guidance,
            observation=observation,
            task_name=task_name,
            generation=generation,
            source_skill_ids=source_skill_ids or [],
        )

    async def distill_general(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
    ) -> GeneratedSkill | None:
        return await self.generate(axis, observation, task_name, generation, level=SkillLevel.GENERAL)

    async def compile_template(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str] | None = None,
    ) -> GeneratedSkill | None:
        return await self.generate(
            axis,
            observation,
            task_name,
            generation,
            level=SkillLevel.TEMPLATE,
            source_skill_ids=source_skill_ids,
        )

    async def instantiate_ephemeral(
        self,
        axis: SkillAxis,
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str] | None = None,
    ) -> GeneratedSkill | None:
        return await self.generate(
            axis,
            observation,
            task_name,
            generation,
            level=SkillLevel.EPHEMERAL,
            source_skill_ids=source_skill_ids,
        )

    async def _generate_guidance(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        observation: dict[str, Any],
        polarity: SkillPolarity = SkillPolarity.POSITIVE,
    ) -> dict | None:
        sys_desc = observation.get("_sys_desc")
        manifest = observation.get("_manifest", [])
        data_request: list[str] = []
        if manifest:
            try:
                phase1_msg = self._build_phase1_message(
                    axis,
                    level,
                    observation,
                    manifest,
                    sys_desc,
                    polarity,
                )
                response1 = await self.llm.generate(
                    "你是一个进化优化数据分析师。根据可用数据源选择你需要查看的数据。",
                    phase1_msg,
                    temperature=0.3,
                    max_tokens=4096,
                )
                data_request = self._parse_data_request(response1)
            except Exception:
                data_request = []

        expanded_data = ""
        if data_request:
            traj = observation.get("_trajectory")
            pop = observation.get("_population")
            if traj:
                from .skill_orchestrator import _expand_data_request

                expanded_data = _expand_data_request(data_request, traj, pop)

        schema = _NEGATIVE_SCHEMA if polarity == SkillPolarity.NEGATIVE else _LAYER_SCHEMAS[(axis, level)]
        user_msg = self._build_phase2_message(
            axis=axis,
            level=level,
            obs=observation,
            output_schema=schema,
            sys_desc=sys_desc,
            expanded_data=expanded_data,
            polarity=polarity,
        )
        system_msg = self._system_prompt(axis, level, polarity)
        try:
            response = await self.llm.generate(
                system_msg,
                user_msg,
                temperature=0.4,
                max_tokens=4096,
            )
            return self._parse_json(response)
        except Exception:
            return None

    @staticmethod
    def _system_prompt(
        axis: SkillAxis,
        level: SkillLevel,
        polarity: SkillPolarity,
    ) -> str:
        if polarity == SkillPolarity.NEGATIVE:
            return (
                "你是一个进化失败模式分析专家。"
                "请基于诊断包提炼出应该被明确禁止的操作模式，并给出替代策略。"
            )
        return (
            "你是一个进化优化技能蒸馏专家。"
            f"请沿 {axis.value} 方向生成一个 {level.value} skill，"
            "要求可复用、可迁移，并严格依据诊断包。"
        )

    @staticmethod
    def _infer_level(observation: dict[str, Any]) -> SkillLevel:
        diag = observation.get("diagnostic_packet", {})
        total_steps = observation.get("total_steps_completed", 0)
        if diag.get("phase") in ("late", "stalled") or total_steps >= 12:
            return SkillLevel.EPHEMERAL
        if total_steps >= 4 or diag.get("population", {}).get("convergence"):
            return SkillLevel.TEMPLATE
        return SkillLevel.GENERAL

    def _build_skill(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        polarity: SkillPolarity,
        guidance: dict[str, Any],
        observation: dict[str, Any],
        task_name: str,
        generation: int,
        source_skill_ids: list[str],
    ) -> GeneratedSkill:
        description = guidance.pop(
            "description",
            f"[{polarity.value}:{level.value}:{axis.value}] auto-generated skill",
        )
        diag = observation.get("diagnostic_packet", {})
        self._id_counter += 1
        return GeneratedSkill(
            skill_id=(
                f"{polarity.value}_{level.value}_{axis.value}_{task_name}_{generation}_{self._id_counter}"
            ),
            axis=axis,
            level=level,
            polarity=polarity,
            description=description,
            guidance=guidance,
            source_observation=self._summarize_observation(observation),
            task_origin=task_name,
            generation=generation,
            task_tags=list(diag.get("task_tags", [])),
            applicable_stages=[diag.get("phase", "early")],
            trigger_diagnostics=list(diag.get("diagnostic_tags", [])),
            source_skill_ids=source_skill_ids,
        )

    def _build_phase1_message(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        obs: dict[str, Any],
        manifest: list[tuple[str, str, str]],
        sys_desc=None,
        polarity: SkillPolarity = SkillPolarity.POSITIVE,
    ) -> str:
        parts = []
        if sys_desc:
            parts.append(sys_desc.format_for_prompt())
            parts.append("")
        parts.append(self._format_obs_summary(obs))
        parts.append("")
        parts.append("## 可查看的数据源")
        for key, desc, amount in manifest:
            parts.append(f"- **{key}**: {desc} ({amount})")
        parts.append("")
        intent = "anti-skill" if polarity == SkillPolarity.NEGATIVE else f"{level.value} skill"
        parts.append(
            f"你需要沿 **{axis.value}** 方向生成一个 **{intent}**。选择你需要查看的数据源。"
        )
        parts.append(
            '输出 JSON: {"data_request": ["key1", "key2"], "reasoning": "为什么需要这些数据"}'
        )
        return "\n".join(parts)

    def _build_phase2_message(
        self,
        axis: SkillAxis,
        level: SkillLevel,
        obs: dict[str, Any],
        output_schema: str,
        sys_desc=None,
        expanded_data: str = "",
        polarity: SkillPolarity = SkillPolarity.POSITIVE,
    ) -> str:
        parts = []
        if sys_desc:
            parts.append(sys_desc.format_for_prompt())
            parts.append("")
        parts.append(self._format_obs_summary(obs))

        diagnostic_packet = obs.get("diagnostic_packet", {})
        if diagnostic_packet:
            parts.append("\n## 结构化诊断包")
            parts.append(json.dumps(diagnostic_packet, indent=2, ensure_ascii=False))

        if expanded_data:
            parts.append("\n## 你请求的详细数据")
            parts.append(expanded_data)

        history = obs.get("skill_history", [])
        if history:
            parts.append("\n## 已尝试的策略")
            for h in history[-6:]:
                eff = "+" if h.get("improvement", 0) > 0 else "-"
                parts.append(
                    f"  [{eff}] {h['polarity']}/{h['level']}/{h['axis']}: {h.get('summary', '')[:120]}"
                )

        mode = "anti-skill" if polarity == SkillPolarity.NEGATIVE else f"{level.value} skill"
        parts.append(
            f"\n请生成一个 {axis.value} 方向的 {mode}，要求直接基于上面的诊断包，不要泛泛而谈。"
        )
        parts.append(f"\n{output_schema}")
        return "\n".join(parts)

    @staticmethod
    def _format_obs_summary(obs: dict[str, Any]) -> str:
        parts = [
            "## 当前进化状态",
            f"- 任务: {obs.get('task_name', 'unknown')}",
            f"- 当前最高分: {obs.get('best_score', 0):.6f}",
            f"- 基线分: {obs.get('baseline_score', 0):.6f}",
            f"- Segment: {obs.get('generation', 0)}",
        ]
        traj = obs.get("score_trajectory", [])
        if traj:
            parts.append(
                f"- 段级分数轨迹: {' → '.join(f'{s:.4f}' for s in traj[-10:])}"
            )
        dyn = obs.get("step_dynamics", {})
        if dyn.get("total_steps", 0) > 0:
            parts.append(f"- 早期改进速率: {dyn.get('early_rate', 0):.6f}/步")
            parts.append(f"- 近期改进速率: {dyn.get('late_rate', 0):.6f}/步")
            parts.append(f"- 最近 20 步停滞比: {dyn.get('stagnant_step_ratio', 0):.0%}")
            parts.append(
                f"- 距上次改进: {dyn.get('steps_since_last_improvement', 0)} 步"
            )
        return "\n".join(parts)

    @staticmethod
    def _parse_data_request(response: str) -> list[str]:
        m = re.search(r"\{[\s\S]*\}", response)
        if not m:
            return []
        try:
            data = json.loads(m.group())
            req = data.get("data_request", [])
            return req if isinstance(req, list) else []
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _parse_json(response: str) -> dict | None:
        m = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{[\s\S]*\}", response)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _heuristic_fallback(axis: SkillAxis, obs: dict, level: SkillLevel) -> dict:
        diagnostic_packet = obs.get("diagnostic_packet", {})
        phase = diagnostic_packet.get("phase", "early")
        bottleneck = diagnostic_packet.get("population", {}).get("status", "unknown")
        if level == SkillLevel.GENERAL:
            return {
                "description": f"{axis.value} general: 面向 {phase} 阶段的通用原则",
                "principle": "先用诊断包定位高杠杆修改点，再扩大搜索或收敛。",
                "heuristics": [
                    "优先解释 evaluator 真正在奖励什么",
                    "区分报错、无效改动和真正有益改动",
                ],
                "transfer_rule": "适用于有明确评估信号且允许逐步迭代的任务",
            }
        if level == SkillLevel.TEMPLATE:
            return {
                "description": f"{axis.value} template: 针对 {bottleneck} 的模板策略",
                "pattern": f"当前群体表现为 {bottleneck}",
                "actions": [
                    "基于最近有效改动总结可复用操作模板",
                    "把失败模式转成明确禁止项",
                ],
                "stage_hint": phase,
            }
        return {
            "description": f"{axis.value} ephemeral: 当前窗口执行策略",
            "objective": "在当前窗口内提升 best-so-far，而不是做泛化总结",
            "actions": [
                "优先放大已验证有效的局部改动",
                "避免重复最近失败的修改模式",
            ],
            "avoid": ["不要在无诊断支持下随机大改"],
            "bottleneck": bottleneck,
            "fix_direction": "围绕最近有效 parent→child 改动继续推进",
        }

    @staticmethod
    def _heuristic_negative(axis: SkillAxis, obs: dict) -> dict:
        diag = obs.get("diagnostic_packet", {})
        failures = diag.get("failure_breakdown", {})
        dominant = max(failures.items(), key=lambda x: x[1])[0] if failures else "regression"
        return {
            "description": f"anti-{axis.value}: 避免 {dominant}",
            "failure_pattern": dominant,
            "forbidden_actions": [
                "不要重复最近已证明无效的局部修改路径",
                "不要在高错误率阶段继续加大无约束变异",
            ],
            "replacement_actions": [
                "优先回到最近成功 parent 的邻域继续搜索",
                "先修复硬错误，再做结构性探索",
            ],
            "rationale": "当前诊断显示这些动作会继续制造无效尝试或回归。",
        }

    @staticmethod
    def _summarize_observation(observation: dict[str, Any]) -> str:
        diag = observation.get("diagnostic_packet", {})
        return json.dumps(
            {
                "task": observation.get("task_name"),
                "best_score": observation.get("best_score"),
                "phase": diag.get("phase"),
                "diagnostic_tags": diag.get("diagnostic_tags", []),
                "failure_breakdown": diag.get("failure_breakdown", {}),
            },
            ensure_ascii=False,
        )
