"""Filesystem-backed managed skill loading."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import managed_skills_dir
from .models import GeneratedSkill, SkillAxis, SkillLevel, SkillPolarity

STARTER_SKILL_ID = "system_starter_target_analysis"

ManagedHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


def load_managed_skills() -> tuple[list[GeneratedSkill], dict[str, ManagedHook]]:
    skills: list[GeneratedSkill] = []
    hooks: dict[str, ManagedHook] = {}
    base_dir = managed_skills_dir()
    if not base_dir.exists():
        return skills, hooks

    for meta_path in sorted(base_dir.rglob("skill.json")):
        skill_dir = meta_path.parent
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        axis_name, level_name = _infer_axis_level_from_path(base_dir, skill_dir)
        skill_id = data["skill_id"]
        hook_file = data.get("hook_file")
        hook_path = skill_dir / hook_file if hook_file else None
        hook_entrypoint = data.get("hook_entrypoint", "run") if hook_file else None
        skill = GeneratedSkill(
            skill_id=skill_id,
            axis=SkillAxis(data.get("axis", axis_name)),
            level=SkillLevel(data.get("level", level_name)),
            polarity=SkillPolarity(data.get("polarity", "positive")),
            description=data.get("description", ""),
            guidance=data.get("guidance", {}),
            source_observation=data.get("source_observation", "managed_skill"),
            task_origin=data.get("task_origin", "global"),
            generation=int(data.get("generation", 0)),
            task_tags=data.get("task_tags", []),
            applicable_stages=data.get("applicable_stages", []),
            trigger_diagnostics=data.get("trigger_diagnostics", []),
            source_skill_ids=data.get("source_skill_ids", []),
            hook_name=data.get("hook_name", skill_id if hook_file else None),
            hook_mode=data.get("hook_mode"),
            skill_path=str(skill_dir),
            hook_path=str(hook_path) if hook_path else None,
            hook_entrypoint=hook_entrypoint,
            protected=bool(data.get("protected", True)),
        )
        skills.append(skill)
        if hook_path and hook_path.exists() and hook_entrypoint:
            hook = _load_hook(skill_id, hook_path, hook_entrypoint)
            if hook is not None:
                hooks[skill_id] = hook

    return skills, hooks


def _infer_axis_level_from_path(base_dir: Path, skill_dir: Path) -> tuple[str, str]:
    rel_parts = skill_dir.relative_to(base_dir).parts
    if len(rel_parts) >= 3:
        return rel_parts[0], rel_parts[1]
    return "diagnosis", "general"


def _load_hook(
    skill_id: str,
    hook_path: Path,
    entrypoint: str,
) -> ManagedHook | None:
    spec = importlib.util.spec_from_file_location(
        f"meta_evolve_managed_skill_{skill_id}",
        str(hook_path),
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, entrypoint, None)
    if fn is None or not callable(fn):
        return None
    return fn


def load_hook_for_skill(skill: GeneratedSkill) -> ManagedHook | None:
    if not skill.hook_path or not skill.hook_entrypoint:
        return None
    return _load_hook(skill.skill_id, Path(skill.hook_path), skill.hook_entrypoint)


def refresh_skills() -> int:
    """删除所有非 starter 的 managed skills，清除持久化数据。返回删除数量。"""
    import shutil
    from ..config import shared_skill_artifact_path, shared_skill_library_path

    # 1. 删除持久化 JSON
    removed = 0
    for p in [shared_skill_library_path(), shared_skill_artifact_path()]:
        if p.exists():
            p.unlink()
            removed += 1

    # 2. 删除非 protected 的 managed skill 目录
    base_dir = managed_skills_dir()
    if not base_dir.exists():
        return removed

    for meta_path in sorted(base_dir.rglob("skill.json")):
        skill_dir = meta_path.parent
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("protected", False):
            continue
        shutil.rmtree(skill_dir)
        removed += 1

    return removed


async def run_managed_hook(hook: ManagedHook, context: dict[str, Any]) -> dict[str, Any] | None:
    result = hook(context)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, dict) else None
