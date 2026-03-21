"""Materialize generated skills into filesystem-managed bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import managed_skills_dir
from .models import GeneratedSkill


def materialize_skill(skill: GeneratedSkill) -> GeneratedSkill:
    skill_dir = managed_skills_dir() / skill.axis.value / skill.level.value / _slugify(skill.skill_id)
    skill_dir.mkdir(parents=True, exist_ok=True)

    hook_file = None
    hook_entrypoint = None
    if skill.hook_source:
        hook_file = "hook.py"
        hook_entrypoint = skill.hook_entrypoint or "run"
        (skill_dir / hook_file).write_text(skill.hook_source)
        skill.hook_path = str(skill_dir / hook_file)
        skill.hook_entrypoint = hook_entrypoint
    else:
        skill.hook_path = None
        skill.hook_entrypoint = None
        skill.hook_name = None

    metadata = {
        "skill_id": skill.skill_id,
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
        "hook_file": hook_file,
        "hook_entrypoint": hook_entrypoint,
        "protected": skill.protected,
        "status": "starter" if skill.protected else "temporary",
    }
    (skill_dir / "skill.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (skill_dir / "SKILL.md").write_text(_render_skill_doc(skill))
    skill.skill_path = str(skill_dir)
    return skill


def _render_skill_doc(skill: GeneratedSkill) -> str:
    kind = "executable" if skill.hook_source else "prompt-only"
    lines = [
        f"# {skill.skill_id}",
        "",
        f"- axis: `{skill.axis.value}`",
        f"- level: `{skill.level.value}`",
        f"- polarity: `{skill.polarity.value}`",
        f"- mode: `{kind}`",
        f"- task_origin: `{skill.task_origin}`",
        "",
        "## Summary",
        skill.description or "(no description)",
        "",
        "## Guidance",
        "```json",
        json.dumps(skill.guidance, indent=2, ensure_ascii=False),
        "```",
    ]
    if skill.hook_source:
        lines.extend(
            [
                "",
                "## Hook",
                f"- entrypoint: `{skill.hook_entrypoint or 'run'}`",
                f"- mode: `{skill.hook_mode or 'always'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "generated_skill"
