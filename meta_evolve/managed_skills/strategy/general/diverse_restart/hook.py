"""Diverse restart search strategy: escape local optima."""

from __future__ import annotations

from typing import Any


async def run(context: dict[str, Any]) -> dict[str, Any]:
    obs = context["obs"]
    stagnation = obs.get("stagnation", 0)

    return {
        "guidance": {
            "principle": "Diverse restart: force departure from the current search direction and explore entirely new algorithm space",
            "mode": "diverse_restart",
            "actions": [
                "Tournament selection for parent (not necessarily the best)",
                "Select maximally diverse programs for context",
                "High temperature + high mutation to encourage novel algorithms",
                "Allow crossover to combine strengths from different directions",
            ],
        },
        "search_policy": {
            "name": "diverse_restart",
            "parent_selection": "tournament",
            "context_mode": "diverse",
            "temperature": 1.2,
            "mutation_strength": 0.85,
            "crossover_rate": 0.4,
            "exploration_rate": 0.6,
            "prompt_supplement": (
                "Current strategy: diverse restart. Stagnated for {} rounds. "
                "You must try completely different algorithms / data structures / mathematical methods. "
                "Do not fine-tune the existing code — design a brand-new solution from scratch. "
                "You may combine ideas from different programs in the context."
            ).format(stagnation),
        },
    }
