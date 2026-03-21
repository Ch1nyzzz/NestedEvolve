# positive_template_reflection_math/circle_packing_5_2

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `executable`
- task_origin: `math/circle_packing`

## Summary
Introduce a continuous non‑linear optimization step (e.g., SciPy SLSQP) to maximize a common radius for all 26 circles while respecting box and non‑overlap constraints.

## Guidance
```json
{
  "idea": "Replace the static deterministic lattice construction with a small, deterministic SLSQP optimisation that treats the 2·26+1 variables (x_i, y_i, r) as continuous and directly maximises r under the feasibility constraints. Seed the optimiser with the current best hexagonal layout so that only a few iterations are needed, keeping runtime well below the evaluation budget.",
  "actions": [
    "Import `scipy.optimize` (SLSQP) and add it to the program’s imports.",
    "Create a new function `def optimise_packing(initial_centers, initial_r):` that builds the variable vector `z = [x1, y1, …, x26, y26, r]` from the seed layout.",
    "Define the objective as `-r` (so that minimisation maximises the radius).",
    "Add inequality constraints: for each circle `0 <= xi <= 1 - r`, `0 <= yi <= 1 - r`; for each pair ` (xi - xj)^2 + (yi - yj)^2 >= (2r)^2 ` expressed as ` (2r)^2 - ((xi - xj)^2 + (yi - yj)^2) <= 0 `.",
    "Supply the constraints to `scipy.optimize.minimize` with method='SLSQP', a modest `maxiter` (e.g., 200) and a tight tolerance.",
    "After optimisation, clip any tiny violations, recompute the sum of radii, and return the final `(centers, radii, sum_radii)` tuple.",
    "Wrap the call in a try/except block; if the optimiser fails or exceeds the time budget, fall back to the original deterministic `construct_packing()` layout."
  ],
  "cautions": "The SLSQP solver is non‑convex; ensure the initial point is feasible (use the current deterministic layout). Keep the number of iterations low (≤ 200) to stay within the 2 s per‑call budget. The constraint functions must be vectorised for speed; avoid Python loops inside the constraint callbacks. If SciPy is unavailable in the sandbox, the fallback path must return the deterministic solution to avoid runtime errors.",
  "approach_type": "continuous_optimization"
}
```

## Hook
- entrypoint: `run`
- mode: `always`
