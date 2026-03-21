"""Error-guided repair search strategy: inject failure information directly into the prompt."""

from __future__ import annotations

from typing import Any


async def run(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["obs"]
    diag = obs.get("diagnostic_packet", {})
    failure_breakdown = diag.get("failure_breakdown", {})
    recent_errors = obs.get("recent_errors", [])
    regressive = diag.get("regressive_edits", [])

    # Build error summary
    error_summary_parts = []
    if failure_breakdown:
        error_summary_parts.append("Failure breakdown: " + ", ".join(
            f"{k}={v}" for k, v in failure_breakdown.items()
        ))
    for err in recent_errors[-3:]:
        error_summary_parts.append(f"- {err[:150]}")
    for edit in regressive[-2:]:
        error_summary_parts.append(
            "- Regression: score {:.4f} → {:.4f}, diff: {}".format(
                edit.get("parent_score", 0),
                edit.get("score", 0),
                (edit.get("diff_summary") or "n/a")[:100],
            )
        )

    error_text = "\n".join(error_summary_parts) if error_summary_parts else "No specific error information"

    return {
        "guidance": {
            "principle": "Error-guided repair: analyze failure causes and apply targeted fixes",
            "mode": "error_guided_repair",
            "actions": [
                "Select parent from elites (ensure high-quality starting point)",
                "Low temperature to avoid introducing new errors",
                "Inject failure information into LLM prompt for targeted repair",
            ],
            "error_summary": error_text,
        },
        "search_policy": {
            "name": "error_guided_repair",
            "parent_selection": "elite",
            "context_mode": "top_only",
            "temperature": 0.5,
            "mutation_strength": 0.35,
            "max_retries": 3,
            "prompt_supplement": (
                "Current strategy: error-guided repair. Many recent candidates have failed.\n"
                "Below are the failure details — analyze them carefully and avoid repeating the same mistakes:\n"
                f"{error_text}\n"
                "Ensure your code: 1) has correct syntax 2) does not time out 3) satisfies all hard constraints"
            ),
        },
    }
