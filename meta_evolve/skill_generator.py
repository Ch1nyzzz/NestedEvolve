"""SkillGenerator：LLM 驱动的 Skill 动态生成器。

观察进化轨迹 → 沿 axis 方向生成具体 skill（含 one-liner description）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import LLMClient
from .skill import GeneratedSkill, SkillAxis


_AXIS_PROMPTS: dict[SkillAxis, dict[str, str]] = {
    SkillAxis.REFLECTION: {
        "system": (
            "你是一个进化算法的范式分析专家。分析当前所有方案的共性局限，"
            "提出一个完全不同的高级算法思路来突破停滞。"
        ),
        "output_schema": """输出 JSON:
{
  "description": "一句话描述这个 skill 的适用场景和作用（供后续选择用）",
  "idea": "一句话描述核心思路",
  "detail": "详细的实现指导（2-5 句话）",
  "cautions": "实现时的注意事项",
  "approach_type": "算法类型标签"
}""",
    },
    SkillAxis.DIAGNOSIS: {
        "system": (
            "你是一个进化过程的全面诊断专家。职责：\n"
            "1. 失败分析：错误类型、超时原因、零分根因\n"
            "2. 中间输出分析：从 partial scores 中提取洞察，识别瓶颈\n"
            "3. 改进方向：基于诊断给出可操作指导"
        ),
        "output_schema": """输出 JSON:
{
  "description": "一句话描述这个 skill 的适用场景和作用（供后续选择用）",
  "failure_pattern": "主要失败模式（如果有）",
  "root_cause": "根本原因分析",
  "bottleneck": "当前阻碍分数提升的主要瓶颈",
  "intermediate_insight": "从中间结果/错误中提取的关键洞察",
  "fix_direction": "具体的修复/改进方向",
  "avoid": ["要避免的做法1", "要避免的做法2"],
  "severity": "high/medium/low"
}""",
    },
    SkillAxis.STRATEGY: {
        "system": (
            "你是一个进化搜索策略专家。根据当前进化阶段和轨迹，"
            "统一决定下一段的搜索策略：搜索模式、变异控制、种群参数。\n"
            "早期偏 explore 建立多样性，后期偏 exploit 精修高分方案。"
        ),
        "output_schema": """输出 JSON:
{
  "description": "一句话描述这个 skill 的适用场景和作用（供后续选择用）",
  "mode": "exploration / exploitation / balanced",
  "emphasis": "对 LLM 的具体搜索指导（1-3 句话）",
  "mutation_strength_hint": "high / medium / low",
  "prefer_diff": true/false,
  "adjustments": ["其他策略调整建议"],
  "param_hints": {},
  "reasoning": "为什么选择这个策略"
}""",
    },
}


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
    ) -> GeneratedSkill | None:
        """两阶段 skill 生成：Phase 1 看 manifest 选数据，Phase 2 基于展开数据生成 skill。"""
        prompt_cfg = _AXIS_PROMPTS[axis]
        sys_desc = observation.get("_sys_desc")
        manifest = observation.get("_manifest", [])

        # === Phase 1: 看 manifest → 输出 data_request ===
        data_request: list[str] = []
        if manifest:
            try:
                phase1_msg = self._build_phase1_message(
                    axis, observation, manifest, sys_desc
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

        # === 展开数据 ===
        expanded_data = ""
        if data_request:
            traj = observation.get("_trajectory")
            pop = observation.get("_population")
            if traj:
                from .skill_orchestrator import _expand_data_request

                expanded_data = _expand_data_request(data_request, traj, pop)

        # === Phase 2: 看展开数据 → 生成完整 skill ===
        user_msg = self._build_phase2_message(
            axis,
            observation,
            prompt_cfg["output_schema"],
            sys_desc,
            expanded_data,
        )
        try:
            response = await self.llm.generate(
                prompt_cfg["system"],
                user_msg,
                temperature=0.4,
                max_tokens=4096,
            )
            guidance = self._parse_json(response)
            if guidance is None:
                guidance = self._heuristic_fallback(axis, observation)
        except Exception:
            guidance = self._heuristic_fallback(axis, observation)

        # 提取 description，剩余作为 guidance
        description = guidance.pop(
            "description", f"[{axis.value}] auto-generated skill"
        )

        self._id_counter += 1
        return GeneratedSkill(
            skill_id=f"{axis.value}_{task_name}_{generation}_{self._id_counter}",
            axis=axis,
            description=description,
            guidance=guidance,
            source_observation=self._summarize_observation(observation),
            task_origin=task_name,
            generation=generation,
        )

    def _build_phase1_message(
        self,
        axis: SkillAxis,
        obs: dict[str, Any],
        manifest: list[tuple[str, str, str]],
        sys_desc=None,
    ) -> str:
        """Phase 1: 展示 manifest，让 LLM 选择需要的数据。"""
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
        parts.append(f"你需要沿 **{axis.value}** 方向分析。选择你需要查看的数据源。")
        parts.append(
            '输出 JSON: {"data_request": ["key1", "key2"], "reasoning": "为什么需要这些数据"}'
        )
        return "\n".join(parts)

    def _build_phase2_message(
        self,
        axis: SkillAxis,
        obs: dict[str, Any],
        output_schema: str,
        sys_desc=None,
        expanded_data: str = "",
    ) -> str:
        """Phase 2: 展示展开数据 + obs，生成完整 skill。"""
        parts = []
        if sys_desc:
            parts.append(sys_desc.format_for_prompt())
            parts.append("")

        parts.append(self._format_obs_summary(obs))

        if expanded_data:
            parts.append("\n## 你请求的详细数据")
            parts.append(expanded_data)

        # 已尝试策略历史
        history = obs.get("skill_history", [])
        if history:
            parts.append("\n## 已尝试的策略")
            for h in history[-5:]:
                eff = "+" if h.get("improvement", 0) > 0 else "-"
                parts.append(f"  [{eff}] {h['axis']}: {h.get('summary', '')[:100]}")

        best_code = obs.get("best_code_snippet", "")
        if (
            best_code
            and axis in (SkillAxis.REFLECTION, SkillAxis.DIAGNOSIS)
            and not expanded_data
        ):
            parts.append("\n## 当前最优代码（前 500 字符）")
            parts.append(f"```python\n{best_code[:500]}\n```")

        # 为 REFLECTION/DIAGNOSIS 注入评估器源码，帮 LLM 理解评分逻辑
        if axis in (SkillAxis.REFLECTION, SkillAxis.DIAGNOSIS) and sys_desc:
            eval_src = getattr(sys_desc, "evaluator_source", "")
            if eval_src:
                parts.append("\n## 评分函数源码（理解如何打分）")
                parts.append(f"```python\n{eval_src[:2000]}\n```")

        parts.append(f"\n{output_schema}")
        return "\n".join(parts)

    @staticmethod
    def _format_obs_summary(obs: dict[str, Any]) -> str:
        """共用的观察摘要。"""
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

        errors = obs.get("recent_errors", [])
        if errors:
            parts.append(f"- 最近错误: {len(errors)} 条")

        pop = obs.get("population_summary", {})
        if pop.get("size"):
            parts.append(
                f"- 种群: size={pop['size']}, "
                f"mean={pop.get('mean_score', 0):.4f}, "
                f"var={pop.get('score_variance', 0):.6f}"
            )

        # 步级动态指标
        dyn = obs.get("step_dynamics", {})
        if dyn.get("total_steps", 0) > 0:
            parts.append(f"- 早期改进速率: {dyn.get('early_rate', 0):.6f}/步")
            parts.append(f"- 近期改进速率: {dyn.get('late_rate', 0):.6f}/步")
            parts.append(f"- 最近 20 步停滞比: {dyn.get('stagnant_step_ratio', 0):.0%}")
            parts.append(
                f"- 距上次改进: {dyn.get('steps_since_last_improvement', 0)} 步"
            )
            recent = dyn.get("recent_best_so_far", [])
            if recent:
                parts.append(
                    f"- 近期 best-so-far: {' → '.join(f'{s:.4f}' for s in recent)}"
                )

        return "\n".join(parts)

    @staticmethod
    def _parse_data_request(response: str) -> list[str]:
        """从 Phase 1 响应中解析 data_request keys。"""
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
    def _heuristic_fallback(axis: SkillAxis, obs: dict) -> dict:
        stagnation = obs.get("stagnation", 0)
        error_rate = obs.get("error_rate", 0)
        generation = obs.get("generation", 0)

        if axis == SkillAxis.REFLECTION:
            return {
                "description": "停滞突破：尝试完全不同的算法范式",
                "idea": "尝试完全不同的算法范式",
                "detail": "当前方案已连续停滞，需要从根本上重新思考问题建模方式。",
                "cautions": "确保新方案仍然满足所有约束条件",
            }
        elif axis == SkillAxis.DIAGNOSIS:
            return {
                "description": f"诊断：错误率 {error_rate:.0%}，停滞 {stagnation} 段",
                "failure_pattern": f"错误率 {error_rate:.0%}",
                "root_cause": "需要更多数据分析",
                "bottleneck": "未能自动识别",
                "fix_direction": "降低变异幅度，使用更保守的修改策略",
                "avoid": [],
            }
        else:
            mode = "exploration" if (generation < 2 or stagnation >= 3) else "balanced"
            return {
                "description": f"策略调整：{mode} 模式",
                "mode": mode,
                "emphasis": "根据当前阶段自动选择",
                "mutation_strength_hint": "high" if stagnation >= 2 else "medium",
                "prefer_diff": stagnation < 2,
                "adjustments": [],
                "param_hints": {},
            }

    @staticmethod
    def _summarize_observation(obs: dict) -> str:
        parts = [
            f"score={obs.get('best_score', 0):.4f}",
            f"stag={obs.get('stagnation', 0)}",
            f"gen={obs.get('generation', 0)}",
        ]
        if obs.get("error_rate"):
            parts.append(f"err={obs['error_rate']:.0%}")
        return ", ".join(parts)
