"""SkillDistiller：跨任务的 skill 效果分析和路由优化。

不再调整预定义 trigger，而是：
1. 分析哪些 axis 在哪些任务类型上有效
2. 识别 skill 之间的协同/冲突模式
3. 调整 axis 检测器的灵敏度
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm_client import LLMClient
from .skill_library import SkillLibrary
from .trajectory import RunTrajectory


@dataclass
class DistillResult:
    """蒸馏结果。"""

    axis_effectiveness: dict[str, dict]  # axis → {task_type → score}
    observations: str = ""
    recommendations: list[str] = field(default_factory=list)


DISTILL_SYSTEM = """你是一个 meta-learning 分析器。分析多个任务上的进化轨迹和 skill 使用记录，总结：
1. 每个 axis（reflection/diagnosis/strategy）在什么类型任务上最有效
2. 哪些 axis 组合有协同效果
3. 哪些 skill 实例特别成功，它们的共性是什么

输出 JSON:
{
  "axis_effectiveness": {
    "reflection": {"effective_on": ["停滞严重的任务"], "avg_improvement": 0.05},
    "diagnosis": {"effective_on": ["高错误率任务"], "avg_improvement": 0.03},
    "strategy": {"effective_on": ["需要搜索策略调整的任务"], "avg_improvement": 0.04}
  },
  "observations": "分析总结",
  "recommendations": ["建议1", "建议2"]
}"""


class SkillDistiller:
    """跨任务 skill 效果分析。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def analyze(
        self,
        trajectories: list[RunTrajectory],
        library: SkillLibrary,
    ) -> DistillResult:
        """分析多条轨迹，生成蒸馏结果。"""
        evidence = library.get_evidence_summary()
        summaries = [traj.to_summary() for traj in trajectories]
        joined_summaries = "---\n".join(summaries)

        user_msg = f"""## Skill Evidence（按 axis 分组）
{json.dumps(evidence, indent=2, ensure_ascii=False)}

## 本轮轨迹
{joined_summaries}

请分析各 axis 的效果并给出建议。"""

        try:
            response = await self.llm.generate(
                DISTILL_SYSTEM, user_msg, temperature=0.3, max_tokens=1500
            )
            return self._parse(response)
        except Exception:
            return self._fallback(trajectories, library)

    def _parse(self, response: str) -> DistillResult:
        m = re.search(r"\{[\s\S]*\}", response)
        if not m:
            return DistillResult(axis_effectiveness={}, observations="parse failed")
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return DistillResult(axis_effectiveness={}, observations="json error")

        return DistillResult(
            axis_effectiveness=data.get("axis_effectiveness", {}),
            observations=data.get("observations", ""),
            recommendations=data.get("recommendations", []),
        )

    @staticmethod
    def _fallback(
        trajectories: list[RunTrajectory], library: SkillLibrary
    ) -> DistillResult:
        """纯统计 fallback。"""
        evidence = library.get_evidence_summary()
        axis_eff = {}
        for axis_name, skills in evidence.items():
            if skills:
                avg = sum(s["avg_improvement"] for s in skills) / len(skills)
                axis_eff[axis_name] = {"avg_improvement": avg, "n_skills": len(skills)}

        return DistillResult(
            axis_effectiveness=axis_eff,
            observations="statistical fallback",
        )
