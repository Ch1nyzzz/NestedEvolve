"""Filesystem-backed managed skill loading."""

from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import managed_skills_dir, skill_archives_dir
from .models import GeneratedSkill, SkillAxis, SkillLevel, SkillPolarity

STARTER_SKILL_ID = "system_starter_target_analysis"

# status 三级：starter (始终可用) > archived (wrap 后收纳) > temporary (运行中生成)
STATUS_STARTER = "starter"
STATUS_ARCHIVED = "archived"
STATUS_TEMPORARY = "temporary"

ManagedHook = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


def _read_status(data: dict) -> str:
    """兼容旧格式：protected=True → starter，否则 temporary。"""
    if "status" in data:
        return data["status"]
    return STATUS_STARTER if data.get("protected", False) else STATUS_TEMPORARY


def load_managed_skills(
    include_archived: bool = True,
) -> tuple[list[GeneratedSkill], dict[str, ManagedHook]]:
    """加载 managed skills。

    include_archived=False 时只加载 starter，跳过 archived 和 temporary。
    include_archived=True 时加载 starter + archived + temporary。
    """
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

        status = _read_status(data)
        if not include_archived and status != STATUS_STARTER:
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
            protected=status == STATUS_STARTER,
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
    """删除 temporary skills，保留 starter 和 archived。清除持久化 JSON。"""
    from ..config import shared_skill_artifact_path, shared_skill_library_path

    removed = 0
    for p in [shared_skill_library_path(), shared_skill_artifact_path()]:
        if p.exists():
            p.unlink()
            removed += 1

    base_dir = managed_skills_dir()
    if not base_dir.exists():
        return removed

    for meta_path in sorted(base_dir.rglob("skill.json")):
        skill_dir = meta_path.parent
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = _read_status(data)
        if status == STATUS_TEMPORARY:
            shutil.rmtree(skill_dir)
            removed += 1

    return removed


def archive_run_skills(run_name: str) -> tuple[int, Path]:
    """把本次运行生成的 temporary skills 移动到 archive 目录。"""
    base_dir = managed_skills_dir()
    archive_dir = skill_archives_dir() / run_name
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    if not base_dir.exists():
        return moved, archive_dir
    for meta_path in sorted(base_dir.rglob("skill.json")):
        skill_dir = meta_path.parent
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = _read_status(data)
        if status != STATUS_TEMPORARY:
            continue
        rel = skill_dir.relative_to(base_dir)
        dst = archive_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(dst))
        moved += 1
    return moved, archive_dir


def merge_skills(run_name: str, as_archived: bool = True) -> int:
    """从 archive 目录复制 skills 到 managed_skills。

    as_archived=True 时标记为 archived（不会被 refresh 删除）。
    """
    archive_dir = skill_archives_dir() / run_name
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive not found: {archive_dir}")
    base_dir = managed_skills_dir()
    merged = 0
    for meta_path in sorted(archive_dir.rglob("skill.json")):
        skill_dir = meta_path.parent
        rel = skill_dir.relative_to(archive_dir)
        dst = base_dir / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(skill_dir), str(dst))
        # 更新 status
        if as_archived:
            dst_meta = dst / "skill.json"
            try:
                data = json.loads(dst_meta.read_text())
                data["status"] = STATUS_ARCHIVED
                data.pop("protected", None)
                dst_meta.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            except (OSError, json.JSONDecodeError):
                pass
        merged += 1
    return merged


def list_archives() -> list[tuple[str, int]]:
    """列出所有 archive 及其 skill 数。"""
    archives_dir = skill_archives_dir()
    result = []
    for d in sorted(archives_dir.iterdir()):
        if not d.is_dir():
            continue
        count = len(list(d.rglob("skill.json")))
        result.append((d.name, count))
    return result


async def run_managed_hook(hook: ManagedHook, context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        result = hook(context)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, dict) else None
    except Exception as e:
        print(f"    [skill-hook-err] {e}")
        return None
