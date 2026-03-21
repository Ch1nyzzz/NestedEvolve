# positive_template_reflection_math/circle_packing_30_6

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `executable`
- task_origin: `math/circle_packing`

## Summary
Introduce a small‑population evolutionary algorithm with crossover, mutation and force‑directed refinement to explore radically different circle packings.

## Guidance
```json
{
  "idea": "Replace the greedy centre‑selection with a population‑based search: initialise a diverse set of packings, evolve them through crossover and mutation, and after each generation run a short deterministic repulsive‑force relaxation to enforce feasibility and improve the sum of radii.",
  "actions": [
    "Create an initial population (size 8) using three seeding strategies: (a) the existing hex‑lattice/farthest‑point layout, (b) random sequential placement with wall‑distance checks, and (c) jittered uniform grid points.",
    "Define a fitness function that returns the sum of radii after a quick feasibility check (validate_packing) – infeasible individuals get fitness 0.",
    "Implement crossover: randomly pick two parents and exchange a random subset of their centre coordinates (keeping the partner’s radii unchanged). After crossover recompute radii for the offspring by running a few iterations of the existing _expand_radii routine.",
    "Implement mutation operators: • small Gaussian jitter of centre positions (clipped to the unit square), • optional radius jitter (add/subtract a tiny value), • a “push‑away” step that moves overlapping circles along the line connecting their centres.",
    "After each offspring is generated, run a deterministic force‑directed relaxation (10‑15 iterations) that applies repulsive forces proportional to overlap and wall‑penetration, then call _expand_radii to maximise individual radii while keeping centres fixed.",
    "Select the top‑k (e.g., 4) individuals by fitness to form the next generation; optionally inject a fresh seed (hex‑lattice) every few generations to preserve diversity.",
    "Terminate after a fixed budget (≤ 4 generations) or when stagnation exceeds a threshold, and output the best individual’s centres, radii and sum."
  ],
  "cautions": "Keep each generation cheap – use vectorised NumPy operations, limit relaxation to ≤15 iterations, and avoid excessive mutation magnitudes that create NaNs. Ensure every offspring respects the shape (26,2) and (26,) before validation. Clip centre updates to [0,1] after each move to stay inside the square.",
  "approach_type": "population‑based evolutionary optimisation with deterministic local refinement"
}
```

## Hook
- entrypoint: `run`
- mode: `always`
