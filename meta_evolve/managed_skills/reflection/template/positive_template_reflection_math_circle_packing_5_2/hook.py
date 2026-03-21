async def run(context):
    import json, math, numpy as np
    obs = context.get('obs', {})
    # Extract key metrics
    stagnation = obs.get('stagnation', 0)
    error_rate = obs.get('error_rate', 0.0)
    recent_improvements = obs.get('recent_improvements', 0)
    best_score = obs.get('best_score', 0.0)
    # Determine whether we are stuck
    suggest_opt = False
    if stagnation >= 3 and recent_improvements == 0:
        suggest_opt = True
    # If the environment frequently raises runtime errors, be cautious about new dependencies
    caution_msg = ''
    if error_rate > 0.2:
        caution_msg = 'High error rate detected; ensure SciPy is available and wrap calls in try/except.'
    # Build dynamic guidance
    guidance = {
        'idea': 'Add a deterministic SLSQP optimisation to maximise a common radius for all circles.',
        'actions': [],
        'cautions': caution_msg
    }
    if suggest_opt:
        guidance['actions'] = [
            'Import scipy.optimize (SLSQP) at the top of the file.',
            'Create a helper function optimise_packing(initial_centers, initial_r) that builds the variable vector [x1, y1, ..., x26, y26, r].',
            'Define the objective as -r and add box and non‑overlap inequality constraints for every circle and every pair of circles.',
            'Call scipy.optimize.minimize with method="SLSQP", a modest maxiter (e.g., 200) and tight tolerances, seeded with the current deterministic layout.',
            'After optimisation, clip tiny violations, compute sum_radii and return the result; if optimisation fails, fall back to construct_packing().'
        ]
    else:
        guidance['actions'] = ['Continue using the deterministic hexagonal layout; consider optimisation only after further stagnation.']
    return {'guidance': guidance}
