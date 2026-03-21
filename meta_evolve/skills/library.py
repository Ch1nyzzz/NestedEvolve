"""Skill Library：动态生成的 skill / anti-skill 的存储、检索、evidence 管理。"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    GeneratedSkill,
    SkillAxis,
    SkillEvidence,
    SkillLevel,
    SkillPolarity,
)


class SkillLibrary:
    """动态 Skill 的注册表。"""

    def __init__(
        self,
        persist_path: Path | None = None,
        task_artifact_path: Path | None = None,
    ):
        self.skills: dict[str, GeneratedSkill] = {}
        self.task_artifacts: dict[str, dict[str, dict]] = {}
        self._persist_path = persist_path
        self._task_artifact_path = task_artifact_path
        if persist_path and persist_path.exists():
            self._load(persist_path)
        if task_artifact_path and task_artifact_path.exists():
            self._load_task_artifacts(task_artifact_path)

    def register(self, skill: GeneratedSkill):
        """注册一个新生成的 skill。"""
        self.skills[skill.skill_id] = skill

    def get_task_artifact(self, task_name: str, hook_name: str) -> dict | None:
        return self.task_artifacts.get(task_name, {}).get(hook_name)

    def has_task_artifact(self, task_name: str, hook_name: str) -> bool:
        return hook_name in self.task_artifacts.get(task_name, {})

    def set_task_artifact(self, task_name: str, hook_name: str, payload: dict) -> None:
        self.task_artifacts.setdefault(task_name, {})[hook_name] = payload

    def retrieve(
        self,
        axis: SkillAxis | None = None,
        task_name: str | None = None,
        task_tags: list[str] | None = None,
        phase: str | None = None,
        diagnostics: list[str] | None = None,
        level: SkillLevel | None = None,
        polarity: SkillPolarity | None = SkillPolarity.POSITIVE,
        top_k: int = 3,
    ) -> list[GeneratedSkill]:
        """检索最相关的 skills。"""
        candidates = list(self.skills.values())
        if axis is not None:
            candidates = [s for s in candidates if s.axis == axis]
        if level is not None:
            candidates = [s for s in candidates if s.level == level]
        if polarity is not None:
            candidates = [s for s in candidates if s.polarity == polarity]

        def _score(s: GeneratedSkill) -> float:
            task_bonus = 0.15 if (task_name and s.task_origin == task_name) else 0.0
            context_match = s.matches_context(task_tags, phase, diagnostics)
            evidence = s.evidence.avg_improvement * max(s.evidence.confidence, 0.2)
            freshness = 0.01 * s.generation
            return evidence + task_bonus + context_match + freshness

        candidates.sort(key=_score, reverse=True)
        return candidates[:top_k]

    def retrieve_for_context(
        self,
        task_name: str,
        task_tags: list[str],
        phase: str,
        diagnostics: list[str],
        top_k: int = 6,
    ) -> list[GeneratedSkill]:
        return self.retrieve(
            task_name=task_name,
            task_tags=task_tags,
            phase=phase,
            diagnostics=diagnostics,
            polarity=SkillPolarity.POSITIVE,
            top_k=top_k,
        )

    def retrieve_anti_skills(
        self,
        task_name: str,
        task_tags: list[str],
        phase: str,
        diagnostics: list[str],
        top_k: int = 2,
    ) -> list[GeneratedSkill]:
        return self.retrieve(
            task_name=task_name,
            task_tags=task_tags,
            phase=phase,
            diagnostics=diagnostics,
            polarity=SkillPolarity.NEGATIVE,
            top_k=top_k,
        )

    def record_activation(
        self,
        skill_id: str,
        improvement: float,
        task_name: str | None = None,
        phase: str | None = None,
        stagnation: int = 0,
        error_rate: float = 0.0,
        co_active_count: int = 1,
    ):
        """记录一次激活结果。"""
        if skill_id in self.skills:
            self.skills[skill_id].evidence.record(
                improvement,
                task_name=task_name,
                phase=phase,
                stagnation=stagnation,
                error_rate=error_rate,
                co_active_count=co_active_count,
            )

    def get_skill_history(
        self, task_name: str | None = None, limit: int = 10
    ) -> list[dict]:
        """获取 skill 历史摘要（用于 SkillGenerator 避免重复）。"""
        items = list(self.skills.values())
        if task_name:
            items = [s for s in items if s.task_origin == task_name]
        items.sort(key=lambda s: s.skill_id, reverse=True)
        return [
            {
                "axis": s.axis.value,
                "level": s.level.value,
                "polarity": s.polarity.value,
                "summary": s.source_observation,
                "improvement": s.evidence.avg_improvement,
                "idea": (
                    s.guidance.get("idea")
                    or s.guidance.get("pattern")
                    or s.guidance.get("objective")
                    or s.guidance.get("failure_pattern")
                    or ""
                )[:100],
            }
            for s in items[:limit]
        ]

    def get_catalog(
        self,
        polarity: SkillPolarity | None = SkillPolarity.POSITIVE,
    ) -> list[dict]:
        """返回 skill catalog（供 orchestrator 选择）。"""
        items = self.skills.values()
        if polarity is not None:
            items = [s for s in items if s.polarity == polarity]
        return [
            {
                "skill_id": s.skill_id,
                "axis": s.axis.value,
                "level": s.level.value,
                "polarity": s.polarity.value,
                "description": s.description,
                "activations": s.evidence.activations,
                "avg_improvement": s.evidence.avg_improvement,
                "task_tags": s.task_tags,
                "trigger_diagnostics": s.trigger_diagnostics,
            }
            for s in items
        ]

    def get_evidence_summary(self) -> dict:
        """返回所有 skill 的 evidence 摘要。"""
        by_axis: dict[str, list] = {}
        for s in self.skills.values():
            axis_name = f"{s.polarity.value}:{s.axis.value}:{s.level.value}"
            by_axis.setdefault(axis_name, []).append(
                {
                    "id": s.skill_id,
                    "task": s.task_origin,
                    "task_tags": s.task_tags,
                    "activations": s.evidence.activations,
                    "avg_improvement": s.evidence.avg_improvement,
                    "success_rate": s.evidence.success_rate,
                }
            )
        return by_axis

    def prune(self, max_per_bucket: int = 20):
        """清理低效 skill，每个 polarity/axis/level 保留最优的 N 个。"""
        by_bucket: dict[tuple[SkillPolarity, SkillAxis, SkillLevel], list[GeneratedSkill]] = {}
        for s in self.skills.values():
            by_bucket.setdefault((s.polarity, s.axis, s.level), []).append(s)

        keep_ids = {sid for sid, skill in self.skills.items() if skill.protected}
        for _bucket, skills in by_bucket.items():
            scored = sorted(
                skills,
                key=lambda s: (
                    s.evidence.avg_improvement,
                    s.evidence.success_rate,
                    s.generation,
                ),
                reverse=True,
            )
            for s in scored[:max_per_bucket]:
                keep_ids.add(s.skill_id)
            newest = sorted(skills, key=lambda s: s.generation, reverse=True)
            for s in newest[:3]:
                keep_ids.add(s.skill_id)

        self.skills = {sid: s for sid, s in self.skills.items() if sid in keep_ids}

    def save(self):
        """持久化到 JSON。"""
        if self._persist_path is not None:
            data = {}
            for sid, skill in self.skills.items():
                data[sid] = {
                    "axis": skill.axis.value,
                    "level": skill.level.value,
                    "polarity": skill.polarity.value,
                    "description": skill.description,
                    "guidance": skill.guidance,
                    "source_observation": skill.source_observation,
                    "task_origin": skill.task_origin,
                    "generation": skill.generation,
                    "task_tags": skill.task_tags,
                    "applicable_stages": skill.applicable_stages,
                    "trigger_diagnostics": skill.trigger_diagnostics,
                    "source_skill_ids": skill.source_skill_ids,
                    "hook_name": skill.hook_name,
                    "hook_mode": skill.hook_mode,
                    "skill_path": skill.skill_path,
                    "hook_path": skill.hook_path,
                    "hook_entrypoint": skill.hook_entrypoint,
                    "protected": skill.protected,
                    "evidence": {
                        "activations": skill.evidence.activations,
                        "improvements": skill.evidence.improvements,
                        "matched_tasks": skill.evidence.matched_tasks,
                        "matched_phases": skill.evidence.matched_phases,
                        "stagnation_levels": skill.evidence.stagnation_levels,
                        "error_rates": skill.evidence.error_rates,
                        "co_active_counts": skill.evidence.co_active_counts,
                    },
                }
            self._persist_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        if self._task_artifact_path is not None:
            self._task_artifact_path.write_text(
                json.dumps(self.task_artifacts, indent=2, ensure_ascii=False)
            )

    def _load(self, path: Path):
        """从 JSON 加载。"""
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for sid, d in data.items():
            ev = d.get("evidence", {})
            evidence = SkillEvidence(
                activations=ev.get("activations", 0),
                improvements=ev.get("improvements", []),
                matched_tasks=ev.get("matched_tasks", []),
                matched_phases=ev.get("matched_phases", []),
                stagnation_levels=ev.get("stagnation_levels", []),
                error_rates=ev.get("error_rates", []),
                co_active_counts=ev.get("co_active_counts", []),
            )
            skill = GeneratedSkill(
                skill_id=sid,
                axis=SkillAxis(d.get("axis", "strategy")),
                level=SkillLevel(d.get("level", "ephemeral")),
                polarity=SkillPolarity(d.get("polarity", "positive")),
                description=d.get("description", ""),
                guidance=d.get("guidance", {}),
                source_observation=d.get("source_observation", ""),
                task_origin=d.get("task_origin", ""),
                generation=d.get("generation", 0),
                task_tags=d.get("task_tags", []),
                applicable_stages=d.get("applicable_stages", []),
                trigger_diagnostics=d.get("trigger_diagnostics", []),
                source_skill_ids=d.get("source_skill_ids", []),
                hook_name=d.get("hook_name"),
                hook_mode=d.get("hook_mode"),
                skill_path=d.get("skill_path"),
                hook_path=d.get("hook_path"),
                hook_entrypoint=d.get("hook_entrypoint"),
                protected=bool(d.get("protected", False)),
                evidence=evidence,
            )
            self.skills[sid] = skill

    def _load_task_artifacts(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self.task_artifacts = data
