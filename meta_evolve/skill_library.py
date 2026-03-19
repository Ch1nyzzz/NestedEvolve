"""Skill Library：动态生成的 skill 的存储、检索、evidence 管理。

不再从文件系统加载预定义 skill，而是在运行时由 SkillGenerator 生成并注册。
支持跨任务复用：在 task A 上生成的 skill 可以在 task B 上被检索到。
"""

from __future__ import annotations

import json
from pathlib import Path

from .skill import GeneratedSkill, SkillAxis, SkillEvidence


class SkillLibrary:
    """动态 Skill 的注册表。"""

    def __init__(self, persist_path: Path | None = None):
        # axis → list of skills（按 axis 组织，每个 axis 可以有多个 skill 实例）
        self.skills: dict[str, GeneratedSkill] = {}
        self._persist_path = persist_path

        # 加载持久化数据
        if persist_path and persist_path.exists():
            self._load(persist_path)

    def register(self, skill: GeneratedSkill):
        """注册一个新生成的 skill。"""
        self.skills[skill.skill_id] = skill

    def retrieve(
        self,
        axis: SkillAxis | None = None,
        task_name: str | None = None,
        top_k: int = 3,
    ) -> list[GeneratedSkill]:
        """检索最相关的 skills。

        优先级：
        1. 同 axis + 同 task 上效果好的
        2. 同 axis + 其他 task 上效果好的（跨任务迁移）
        3. 最新生成的（缺少 evidence 时的 fallback）
        """
        candidates = list(self.skills.values())

        if axis is not None:
            candidates = [s for s in candidates if s.axis == axis]

        # 排序：同 task + 高 avg_improvement 优先
        def _score(s: GeneratedSkill) -> float:
            task_bonus = 0.1 if (task_name and s.task_origin == task_name) else 0.0
            return s.evidence.avg_improvement * s.evidence.confidence + task_bonus

        candidates.sort(key=_score, reverse=True)
        return candidates[:top_k]

    def retrieve_by_axes(
        self,
        axes: list[SkillAxis],
        task_name: str | None = None,
        per_axis: int = 1,
    ) -> list[GeneratedSkill]:
        """每个 axis 检索 top-N 个 skill。"""
        result = []
        for axis in axes:
            result.extend(self.retrieve(axis=axis, task_name=task_name, top_k=per_axis))
        return result

    def record_activation(self, skill_id: str, improvement: float):
        """记录一次激活结果。"""
        if skill_id in self.skills:
            self.skills[skill_id].evidence.record(improvement)

    def get_skill_history(
        self, task_name: str | None = None, limit: int = 10
    ) -> list[dict]:
        """获取 skill 历史摘要（用于 SkillGenerator 避免重复）。"""
        items = list(self.skills.values())
        if task_name:
            items = [s for s in items if s.task_origin == task_name]
        # 按 ID（含时序信息）倒序
        items.sort(key=lambda s: s.skill_id, reverse=True)
        return [
            {
                "axis": s.axis.value,
                "summary": s.source_observation,
                "improvement": s.evidence.avg_improvement,
                "idea": s.guidance.get("idea", s.guidance.get("mode", ""))[:100],
            }
            for s in items[:limit]
        ]

    def get_catalog(self) -> list[dict]:
        """返回所有 skill 的 one-liner catalog（供 orchestrator LLM 选择用）。"""
        return [
            {
                "skill_id": s.skill_id,
                "axis": s.axis.value,
                "description": s.description,
                "activations": s.evidence.activations,
                "avg_improvement": s.evidence.avg_improvement,
            }
            for s in self.skills.values()
        ]

    def get_evidence_summary(self) -> dict:
        """返回所有 skill 的 evidence 摘要。"""
        by_axis: dict[str, list] = {}
        for s in self.skills.values():
            axis_name = s.axis.value
            if axis_name not in by_axis:
                by_axis[axis_name] = []
            by_axis[axis_name].append(
                {
                    "id": s.skill_id,
                    "task": s.task_origin,
                    "activations": s.evidence.activations,
                    "avg_improvement": s.evidence.avg_improvement,
                    "success_rate": s.evidence.success_rate,
                }
            )
        return by_axis

    def prune(self, max_per_axis: int = 20):
        """清理低效 skill，每个 axis 保留最优的 N 个。"""
        by_axis: dict[SkillAxis, list[GeneratedSkill]] = {}
        for s in self.skills.values():
            by_axis.setdefault(s.axis, []).append(s)

        keep_ids = set()
        for axis, skills in by_axis.items():
            # 保留 evidence 最好的 + 最新的（给新 skill 机会）
            scored = sorted(
                skills, key=lambda s: s.evidence.avg_improvement, reverse=True
            )
            for s in scored[:max_per_axis]:
                keep_ids.add(s.skill_id)
            # 也保留最新的几个（即使 evidence 不好）
            newest = sorted(skills, key=lambda s: s.skill_id, reverse=True)
            for s in newest[:3]:
                keep_ids.add(s.skill_id)

        self.skills = {sid: s for sid, s in self.skills.items() if sid in keep_ids}

    def save(self):
        """持久化到 JSON。"""
        if self._persist_path is None:
            return
        data = {}
        for sid, skill in self.skills.items():
            data[sid] = {
                "axis": skill.axis.value,
                "description": skill.description,
                "guidance": skill.guidance,
                "source_observation": skill.source_observation,
                "task_origin": skill.task_origin,
                "generation": skill.generation,
                "evidence": {
                    "activations": skill.evidence.activations,
                    "improvements": skill.evidence.improvements,
                },
            }
        self._persist_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self, path: Path):
        """从 JSON 加载。"""
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for sid, d in data.items():
            evidence = SkillEvidence(
                activations=d.get("evidence", {}).get("activations", 0),
                improvements=d.get("evidence", {}).get("improvements", []),
            )
            skill = GeneratedSkill(
                skill_id=sid,
                axis=SkillAxis(d["axis"]),
                description=d.get("description", ""),
                guidance=d["guidance"],
                source_observation=d.get("source_observation", ""),
                task_origin=d.get("task_origin", ""),
                generation=d.get("generation", 0),
                evidence=evidence,
            )
            self.skills[sid] = skill
