"""Formatting helpers for compiled skill context."""

from __future__ import annotations


def format_skill_context(skill_context: dict) -> str:
    if not skill_context:
        return ""

    parts = ["## Strategy Guidance\n", "Compiled skill context based on current evolution state:\n"]

    positives = skill_context.get("positive", {})
    negatives = skill_context.get("negative", {})
    if positives:
        parts.append("### Positive Skills")
        for data in positives.values():
            parts.append(data.get("formatted", ""))
    if negatives:
        parts.append("\n### Anti-Skills")
        for data in negatives.values():
            parts.append(data.get("formatted", ""))
    return "\n".join(parts)
