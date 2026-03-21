async def run(context):
    import math, json
    obs = context.get('obs', {})
    # Extract useful metrics, using defaults if missing
    stagnation = obs.get('stagnation', 0)
    best_score = obs.get('best_score', 0.0)
    error_rate = obs.get('error_rate', 0.0)
    phase = obs.get('phase', 'unknown')

    # Determine CMA‑ES parameters based on current state
    if stagnation >= 8 or error_rate > 0.1:
        # Population is struggling – increase diversity and shorten run
        pop_size = 20
        generations = 10
        restart = True
    else:
        pop_size = 30
        generations = 20
        restart = False

    # Choose penalty weight λ dynamically: larger when many infeasibilities observed
    recent_invalid = obs.get('recent_invalid', 0)
    lambda_penalty = 1e3 if recent_invalid < 5 else 2e3

    # Build a short dynamic guidance payload that the evolver can embed
    guidance = {
        "idea": "CMA‑ES global search with repair and periodic Nelder‑Mead polish",
        "parameters": {
            "population_size": pop_size,
            "generations": generations,
            "lambda_penalty": lambda_penalty,
            "restart_on_stagnation": restart,
            "seed": 42
        },
        "actions": [
            "initialise_population_using_heuristics",
            "run_cma_es_loop",
            "apply_overlap_repair_each_generation",
            "polish_top_individuals_with_nelder_mead",
            "restart_diversity_if_stagnant"
        ]
    }

    # Optionally, if we are already very close to the benchmark, suggest early termination
    target_value = 2.635
    if best_score / target_value > 0.99:
        guidance["note"] = "Current solution already exceeds 99% of benchmark; consider returning early."

    return {"guidance": guidance}
