async def run(context):
    import numpy as np, math, traceback
    obs = context.get('obs', {})
    # Extract useful metrics; provide different guidance depending on stagnation or error rate
    stagnation = obs.get('stagnation', 0)
    error_rate = obs.get('error_rate', 0.0)
    best_score = obs.get('best_score', 0.0)
    # Determine whether we are stuck (high stagnation, low recent improvement)
    stuck = stagnation > 15 or (best_score < 0.998 and stagnation > 5)
    # If many runtime errors, suggest adding robust try/except
    need_error_handling = error_rate > 0.1
    # Build dynamic guidance
    idea = "Add a deterministic feasibility‑preserving simulated annealing phase before the final SLSQP polish."
    actions = []
    if stuck:
        actions.append("Insert a deterministic RandomState(seed=42) and an annealing loop (≈2000 iterations) that proposes small translations for randomly chosen circles.")
        actions.append("Use the existing _optimal_radii LP to recompute radii after each feasible move; accept moves that increase total radius or with Metropolis probability exp((Δsum)/T).")
        actions.append("After annealing, run a short SLSQP refinement (maxiter=2000) on the best feasible configuration.")
    else:
        actions.append("Keep the current deterministic lattice + SLSQP pipeline, but optionally add a lightweight annealing pass (e.g., 500 iterations) to explore new basins.")
    if need_error_handling:
        actions.append("Wrap the entire packing routine in try/except and fall back to the last feasible solution on any exception.")
    # Return guidance dict for the evolver prompt
    return {
        "guidance": {
            "idea": idea,
            "actions": actions,
            "cautions": "Maintain deterministic RNG, respect the 30 s timeout, and validate feasibility after each move."
        }
    }
