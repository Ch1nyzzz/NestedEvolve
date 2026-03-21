# positive_template_reflection_math/circle_packing_33_10

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `executable`
- task_origin: `math/circle_packing`

## Summary
Introduce a deterministic feasibility‑preserving simulated‑annealing refinement before the final gradient step

## Guidance
```json
{
  "idea": "After building the deterministic hexagonal seed, run a fixed‑schedule simulated‑annealing loop that only accepts moves which keep the packing feasible (no overlap, inside the unit square). Use a deterministic pseudo‑random generator (numpy RandomState with a constant seed) so the whole pipeline stays reproducible. When the annealing finishes, hand the resulting centres to the existing LP‑based radius computation and a short SLSQP polish. This adds a global‑exploration phase that can escape the local basins where pure SLSQP stalls, while staying within the 30 s budget and using only NumPy/SciPy.",
  "actions": [
    "Add a deterministic RandomState (seed = 42) and a simple annealing schedule (e.g., temperature = 0.1 → 1e‑4 over 2000 iterations).",
    "Implement a move operator that randomly selects a circle, proposes a small translation (Δx, Δy) drawn from a uniform box scaled by the current temperature, and optionally a tiny radius change (±Δr). Clip the new centre to [0,1] and recompute radii for the whole set via the LP helper; reject the move if any overlap or boundary violation appears.",
    "Accept moves with the Metropolis criterion: always accept if the total sum of radii increases; otherwise accept with probability exp((Δsum)/T). Keep the best feasible configuration seen.",
    "After the annealing loop, call the existing `radii_from_centers` (LP) and a short SLSQP run (maxiter ≈ 2000) to fine‑tune the final positions.",
    "Wrap the whole routine in a try/except block that falls back to the previous best configuration if any unexpected exception occurs, and log the reason for debugging."
  ],
  "cautions": "Ensure the move proposal never produces NaN or out‑of‑bounds values; the LP solver must be called only on feasible centre sets to avoid numerical failures. Keep the total number of annealing iterations low (≈2000) so the whole `run_packing` stays under the 30 s timeout. Use deterministic RNG to avoid stochastic regressions across runs.",
  "approach_type": "deterministic simulated‑annealing refinement (feasibility‑preserving local search)"
}
```

## Hook
- entrypoint: `run`
- mode: `always`
