"""UCB exploration search strategy: balance exploitation and exploration."""

from __future__ import annotations

from typing import Any


async def run(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["obs"]
    diag = obs.get("diagnostic_packet", {})
    pop = diag.get("population", {})
    stagnation = obs.get("stagnation", 0)

    # Dynamically adjust exploration intensity based on stagnation level
    c_value = 1.4 + 0.3 * min(stagnation, 5)
    temperature = min(1.2, 0.7 + 0.1 * stagnation)

    return {
        "guidance": {
            "principle": "UCB balances exploration and exploitation, prioritizing individuals selected fewer times",
            "mode": "ucb_exploration",
            "actions": [
                "Use UCB1 to select parent (exploration coefficient c={:.1f})".format(c_value),
                "Select diverse top-k for context (maximize code distance)",
                "Allow bolder mutation directions",
            ],
        },
        "search_policy": {
            "name": "ucb_exploration",
            "parent_selection": "ucb",
            "context_mode": "diverse",
            "temperature": temperature,
            "mutation_strength": min(0.8, 0.5 + 0.05 * stagnation),
            "prompt_supplement": (
                "Current strategy: UCB exploration mode. You should try algorithmic ideas "
                "different from previous attempts — do not just fine-tune. Explore entirely new solution directions."
            ),
        },
    }
