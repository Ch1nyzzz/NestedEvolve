"""Post-run skill wrapper: 用 LLM 整理、归纳、分级、裁剪本次运行生成的 skills。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..config import managed_skills_dir
from ..integrations.llm import LLMClient

_SYSTEM = (
    "You are an expert skill curator for an evolutionary code optimization framework. "
    "Your job is to review NEW skills from a run, compare them against the EXISTING skill library, "
    "and decide what to keep, merge, or prune — maintaining strict quality and zero redundancy."
)


def _load_existing_archived() -> list[dict]:
    """加载当前 managed_skills 中所有 archived skills。"""
    base = managed_skills_dir()
    skills = []
    if not base.exists():
        return skills
    for meta_path in sorted(base.rglob("skill.json")):
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = data.get("status", "starter" if data.get("protected") else "temporary")
        if status == "archived":
            skills.append(data)
    return skills


def _build_wrap_prompt(
    task_name: str,
    new_skills: list[dict],
    existing_skills: list[dict],
    run_summary: dict,
) -> str:
    parts = [
        "## Run Summary",
        f"- Task: {task_name}",
        f"- Final score: {run_summary.get('final_best', 'N/A')}",
        f"- Baseline: {run_summary.get('baseline', 'N/A')}",
        f"- New skills from this run: {len(new_skills)}",
        f"- Skills used: {run_summary.get('skills_used', [])}",
    ]

    # === 已有库 ===
    if existing_skills:
        parts.append(f"\n## EXISTING Skill Library ({len(existing_skills)} archived skills)")
        parts.append("These are curated skills already in the library. New skills must NOT duplicate these.\n")
        for i, s in enumerate(existing_skills):
            parts.append(
                f"### Existing-{i+1}: `{s['skill_id']}`"
                f"\n- level: {s.get('level')}, axis: {s.get('axis')}"
                f"\n- description: {s.get('description', '')}"
                f"\n- guidance.idea: {s.get('guidance', {}).get('idea', 'N/A')[:200]}"
            )
    else:
        parts.append("\n## EXISTING Skill Library: (empty)")

    # === 新 skills ===
    parts.append(f"\n## NEW Skills from This Run ({len(new_skills)} candidates)")
    for i, s in enumerate(new_skills):
        evidence = s.get("evidence", {})
        outcome = evidence.get("outcome_summary", "no data")
        avg_imp = evidence.get("avg_improvement", 0)
        activations = evidence.get("activations", 0)
        parts.append(
            f"\n### New-{i+1}: `{s['skill_id']}`"
            f"\n- axis: {s.get('axis')}, level: {s.get('level')}, polarity: {s.get('polarity')}"
            f"\n- description: {s.get('description', '')}"
            f"\n- guidance.idea: {s.get('guidance', {}).get('idea', 'N/A')}"
            f"\n- guidance.transfer_rule: {s.get('guidance', {}).get('transfer_rule', 'N/A')}"
            f"\n- evidence: activations={activations}, avg_improvement={avg_imp:.4f}, outcomes={outcome}"
        )

    parts.append("""
## Instructions

IMPORTANT: These skills guide an **LLM-based code evolver**. Skills tell the evolver HOW to mutate code.

For EACH new skill, decide ONE action:

1. **promote_to_general**: Extract a META-LEVEL principle about how the evolver should behave.
   - Must be about the **evolution/mutation strategy**, NOT the domain problem
   - Apply to ANY task (code optimization, math, systems, etc.)
   - If an existing general skill already covers this principle → prune instead

2. **keep_as_template**: Useful for this type of task. Keep domain-specific details.
   - If an existing template covers the same approach → check which is better:
     - If new is better → use `update_existing` instead
     - If existing is better → prune the new one

3. **update_existing**: The new skill overlaps with an existing archived skill, and MERGING them would produce a better skill than either alone.
   - Specify `existing_id` to indicate which existing skill to merge with
   - Provide `new_description` and `new_guidance` that COMBINE the best aspects of both:
     - Keep strengths from the existing skill
     - Incorporate new insights or improvements from the new skill
     - Remove weaknesses from either
   - The existing skill in the library will be replaced with this merged version
   - The new skill will be removed from the archive (its value is absorbed)

4. **prune**: Remove because:
   - Redundant with an existing archived skill (specify which)
   - Ineffective (no activations, negative outcomes)
   - Duplicate of another new skill in this batch

QUALITY RULES:
- The library must have ZERO redundancy. Two skills covering the same method = one must go.
- Negative/ephemeral skills: prune unless genuinely useful as anti-patterns.
- 0 activations or only negative outcomes → prune.
- When in doubt, prune. A small high-quality library beats a large noisy one.
- For general skills: actions must be instructions TO THE EVOLVER, not domain algorithms.

## Output (JSON)
```json
{
  "wrapped_skills": [
    {
      "action": "promote_to_general" | "keep_as_template" | "update_existing" | "prune",
      "original_id": "new skill_id",
      "existing_id": "existing skill_id to merge with (only for update_existing)",
      "reason": "why this action — for update_existing, explain what each skill contributes to the merge",
      "new_description": "rewritten description (for promote/keep/update)",
      "new_guidance": {
        "idea": "merged insight combining best of both (for update_existing) or standalone (for promote/keep)",
        "transfer_rule": "...",
        "actions": ["..."],
        "cautions": "..."
      }
    }
  ]
}
```""")

    return "\n".join(parts)


async def wrap_run_skills(
    task_name: str,
    archive_path: Path,
    run_summary: dict,
    llm: LLMClient,
) -> dict[str, Any]:
    """对一次运行归档的 skills 进行 LLM 整理，与已有库去重。"""
    new_skills = []
    for meta_path in sorted(archive_path.rglob("skill.json")):
        try:
            data = json.loads(meta_path.read_text())
            new_skills.append(data)
        except (OSError, json.JSONDecodeError):
            continue

    if not new_skills:
        return {"promoted": [], "kept": [], "updated": [], "pruned": [], "raw_response": ""}

    existing_skills = _load_existing_archived()
    prompt = _build_wrap_prompt(task_name, new_skills, existing_skills, run_summary)
    response = await llm.generate(_SYSTEM, prompt, temperature=0.2, max_tokens=8192)

    from .generator import SkillGenerator
    result = SkillGenerator._parse_json(response)
    if not result or "wrapped_skills" not in result:
        return {"promoted": [], "kept": [], "updated": [], "pruned": [], "raw_response": response}

    promoted, kept, updated, pruned = [], [], [], []
    for item in result["wrapped_skills"]:
        action = item.get("action", "prune")
        if action == "promote_to_general":
            promoted.append(item)
        elif action == "keep_as_template":
            kept.append(item)
        elif action == "update_existing":
            updated.append(item)
        else:
            pruned.append(item)

    return {
        "promoted": promoted,
        "kept": kept,
        "updated": updated,
        "pruned": pruned,
        "raw_response": response,
    }


def apply_wrap_result(
    archive_path: Path,
    wrap_result: dict[str, Any],
) -> dict[str, int]:
    """根据 wrap 结果处理 archive 和 managed_skills。

    - promoted → archive 中标记 archived + general
    - kept → archive 中标记 archived + template
    - updated → 更新 managed_skills 中对应的已有 skill
    - pruned → 从 archive 中删除
    """
    base_dir = managed_skills_dir()

    # 建索引: archive 中的 skill_id → meta_path
    new_id_to_path: dict[str, Path] = {}
    for meta_path in archive_path.rglob("skill.json"):
        try:
            data = json.loads(meta_path.read_text())
            new_id_to_path[data["skill_id"]] = meta_path
        except (OSError, json.JSONDecodeError):
            continue

    # 建索引: managed_skills 中的 skill_id → meta_path
    existing_id_to_path: dict[str, Path] = {}
    for meta_path in base_dir.rglob("skill.json"):
        try:
            data = json.loads(meta_path.read_text())
            existing_id_to_path[data["skill_id"]] = meta_path
        except (OSError, json.JSONDecodeError):
            continue

    counts = {"promoted": 0, "kept": 0, "updated": 0, "pruned": 0}

    for item in wrap_result.get("promoted", []):
        sid = item.get("original_id")
        if sid not in new_id_to_path:
            continue
        meta_path = new_id_to_path[sid]
        data = json.loads(meta_path.read_text())
        data["level"] = "general"
        data["status"] = "archived"
        data.pop("protected", None)
        data["description"] = item.get("new_description", data["description"])
        if item.get("new_guidance"):
            data["guidance"] = item["new_guidance"]
        meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        counts["promoted"] += 1

    for item in wrap_result.get("kept", []):
        sid = item.get("original_id")
        if sid not in new_id_to_path:
            continue
        meta_path = new_id_to_path[sid]
        data = json.loads(meta_path.read_text())
        data["level"] = "template"
        data["status"] = "archived"
        data.pop("protected", None)
        data["description"] = item.get("new_description", data["description"])
        if item.get("new_guidance"):
            data["guidance"] = item["new_guidance"]
        meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        counts["kept"] += 1

    for item in wrap_result.get("updated", []):
        existing_id = item.get("existing_id")
        new_id = item.get("original_id")
        if not existing_id or existing_id not in existing_id_to_path:
            continue
        # 更新 managed_skills 中的已有 skill
        meta_path = existing_id_to_path[existing_id]
        data = json.loads(meta_path.read_text())
        data["description"] = item.get("new_description", data["description"])
        if item.get("new_guidance"):
            data["guidance"] = item["new_guidance"]
        meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        # 从 archive 中删除新 skill（已合并到已有的）
        if new_id and new_id in new_id_to_path:
            shutil.rmtree(new_id_to_path[new_id].parent)
        counts["updated"] += 1

    for item in wrap_result.get("pruned", []):
        sid = item.get("original_id")
        if sid not in new_id_to_path:
            continue
        shutil.rmtree(new_id_to_path[sid].parent)
        counts["pruned"] += 1

    return counts
