# positive_template_reflection_math/circle_packing_28_8

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `prompt-only`
- task_origin: `math/circle_packing`

## Summary
Introduce a DE‑LP‑Hybrid global‑local optimisation to escape the current local optimum

## Guidance
```json
{
  "idea": "Combine a small‑population Differential Evolution (DE) that searches over the 26 circle centre coordinates with an exact linear‑program (LP) radius computation for each candidate, then locally polish the best individuals with a short SLSQP run. This provides true stochastic global exploration while keeping the final radii optimal for any centre set.",
  "actions": [
    "Implement a deterministic DE loop (fixed random seed) that evolves a population of centre matrices (shape (26,2)) using standard DE mutation‑crossover (e.g., \"rand/1/bin\").",
    "For every DE individual, call the existing LP routine (the same `radii_from_centers` used in the current code) to obtain the maximal feasible radii and the objective value (sum of radii). Use this value as the DE fitness, applying a heavy penalty if any centre lies outside the unit square.",
    "After each DE generation, select the top‑k individuals (e.g., k=3) and run a brief SLSQP optimisation (max 2000 iterations, loose tolerance) starting from those centres and their LP radii to further improve feasibility and objective.",
    "Keep track of the overall evaluation budget (≈224 calls). Allocate ~30 DE generations with a population of 8 (≈240 fitness evaluations) and reserve the remaining budget for the final SLSQP polish on the single best individual.",
    "Seed the initial DE population with the current best 0.9975 packing plus deterministic perturbations (small Gaussian noise with fixed seed) to guarantee reproducibility."
  ],
  "cautions": "Ensure the DE fitness evaluation never exceeds the evaluation budget; each LP solve counts as one evaluation. Clip mutated centres to the unit square before LP evaluation to avoid infeasible crashes. Use a fixed numpy random seed so the stochastic process is deterministic across runs. The final SLSQP should use a modest iteration limit to stay within the remaining time budget.",
  "approach_type": "global‑local evolutionary optimisation (Differential Evolution + linear‑program refinement + local SLSQP polish)"
}
```
