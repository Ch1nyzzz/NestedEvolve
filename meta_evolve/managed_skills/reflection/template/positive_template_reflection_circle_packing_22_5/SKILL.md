# positive_template_reflection_circle_packing_22_5

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `prompt-only`
- task_origin: `circle_packing`

## Summary
Introduce a global convex optimisation (SOCP) formulation using CVXPY to jointly optimise all circle centres and radii in one shot.

## Guidance
```json
{
  "idea": "Formulate the packing problem as a convex second‑order cone programme: maximise the sum of radii subject to ‖c_i‑c_j‖₂ ≤ r_i+r_j (non‑overlap) and 0≤c_i±r_i≤1 (boundary). Solve it with CVXPY (ECOS/SCS). This removes the need for handcrafted row patterns and linear‑program sweeps, giving a provably optimal solution for the continuous relaxation and typically the true optimum for the discrete n=26 case.",
  "actions": [
    "Add `cvxpy` (and a suitable solver like ECOS) to the project dependencies.",
    "Replace the current `construct_packing` implementation with a new function `solve_packing_cvxpy()` that: (a) creates CVXPY variables `x`, `y`, `r` for 26 circles, (b) adds the convex constraints described above, (c) sets the objective `maximize sum(r)`, (d) solves the problem, (e) returns `centers = np.column_stack((x.value, y.value))` and `radii = r.value`.",
    "Expose the new function under the same name (`construct_packing`) so the evaluator can call it unchanged, and optionally keep the old pattern‑based code as a fallback if the solver fails."
  ],
  "cautions": "SOCP solvers may struggle with very tight tolerances; set `solver=ECOS` and `abstol=1e-9, reltol=1e-9`. If the solver hits numerical issues, fall back to the previous pattern‑based method. Ensure that the returned arrays contain no NaNs and that the solver status is `optimal` before using the solution.",
  "approach_type": "convex optimisation / second‑order cone programming"
}
```
