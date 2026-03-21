async def run(context):
    import math, numpy as np
    obs = context.get('obs', {})
    # Extract dynamic signals that influence guidance
    stagnation = obs.get('stagnation', 0.0)          # proportion of iterations with no improvement
    error_rate = obs.get('error_rate', 0.0)          # fraction of runtimes that crashed
    best_score = obs.get('best_score', 0.0)
    population = context.get('population', [])
    # Basic thresholds (tuned empirically for this task)
    high_stagnation = stagnation > 0.8
    high_error = error_rate > 0.2
    # Build adaptive actions list
    actions = []
    # Always initialise a diverse population if none exists
    if not population:
        actions.append("initialise_population")
    # If stagnation is high, increase mutation magnitude and inject fresh seed
    if high_stagnation:
        actions.append("increase_mutation_scale")
        actions.append("inject_hex_lattice_seed")
    # If error rate is high, reduce crossover complexity
    if high_error:
        actions.append("disable_crossover_use_mutation_only")
    # When best_score is already close to target (>=0.995), switch to fine‑tuning mode
    if best_score >= 0.995:
        actions.append("fine_tune_with_small_mutations_and_more_relaxation_steps")
    # Default evolutionary loop actions
    if not actions:
        actions.extend([
            "run_evolutionary_generation",
            "apply_crossover",
            "apply_mutation",
            "run_force_directed_relaxation",
            "select_top_individuals"
        ])
    # Package guidance for the evolver
    guidance = {
        "idea": "population‑based evolutionary search with deterministic refinement",
        "actions": actions,
        "notes": {
            "stagnation": stagnation,
            "error_rate": error_rate,
            "best_score": best_score
        }
    }
    return {"guidance": guidance}
