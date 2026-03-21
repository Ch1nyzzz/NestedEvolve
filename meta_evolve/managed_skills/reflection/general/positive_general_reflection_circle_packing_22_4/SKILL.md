# positive_general_reflection_circle_packing_22_4

- axis: `reflection`
- level: `general`
- polarity: `positive`
- mode: `prompt-only`
- task_origin: `circle_packing`

## Summary
Introduce a joint continuous optimizer (CMA‑ES) to refine both circle centers and radii beyond fixed row patterns

## Guidance
```json
{
  "idea": "Replace the pattern‑only linear‑program step with a global, gradient‑free optimisation (CMA‑ES) that treats the (x, y) coordinates and radii of all 26 circles as free variables. Seed the optimiser with the best pattern found so far, then maximise the sum of radii while penalising overlaps and boundary violations. This removes the structural ceiling imposed by pre‑defined truncated‑hexagonal rows and lets the algorithm discover truly optimal, possibly irregular, arrangements.",
  "actions": [
    "Add a new helper `cmaes_optimize(initial_centers, initial_radii, max_iter=2000)` that builds a 78‑dimensional vector (26 × ( x, y, r )) and runs `cma.CMAEvolutionStrategy` with a custom fitness: negative sum(r) + large penalty·(overlap + out‑of‑bounds). Return the best feasible solution.",
    "In `construct_packing`, after the exhaustive grid‑search over row‑patterns, collect the top‑k candidate layouts (e.g., k=5) and pass each to `cmaes_optimize`. Keep the refined layout with the highest feasible sum‑of‑radii.",
    "Install the lightweight `cma` package (pure‑Python) in the evaluation sandbox if not present, or fall back to a simple differential‑evolution from `scipy.optimize`.",
    "Add a guard to limit total runtime (e.g., stop after 15 s or after 2000 CMA‑ES generations) to respect the 20‑second evaluator timeout."
  ],
  "cautions": "CMA‑ES is stochastic and can exceed the time budget if not capped; ensure a hard iteration/timeout limit. Penalty weights must be large enough to keep infeasible solutions out of the final ranking, otherwise the optimiser may converge to high‑radius but overlapping configurations. Seed with the best known pattern to avoid wasting generations on clearly bad regions.",
  "transfer_rule": "Applicable to any geometric packing or layout optimisation task where the objective is to maximise a sum (area, radius, weight) under non‑overlap and boundary constraints."
}
```
