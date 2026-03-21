# positive_template_reflection_math/circle_packing_18_5

- axis: `reflection`
- level: `template`
- polarity: `positive`
- mode: `prompt-only`
- task_origin: `math/circle_packing`

## Summary
Introduce a force‑directed relaxation loop to jointly optimise circle centres and radii

## Guidance
```json
{
  "idea": "Replace the static grid‑based construction with a physics‑inspired iterative simulation: start from a random (or low‑discrepancy) placement of 26 circles, apply repulsive forces between circles and from the walls while uniformly inflating radii, project back into the unit square and resolve any residual overlaps each iteration, and stop when the total radius stops increasing. This explores the full continuous configuration space and can break the current performance ceiling.",
  "actions": [
    "Remove the fixed‑grid `construct_packing` implementation and create a new `run_packing` that performs a force‑directed relaxation loop.",
    "Initialize 26 circle centres with a deterministic low‑discrepancy sequence (e.g., Halton) seeded for reproducibility; set all radii to a tiny value.",
    "Iterate: compute pairwise repulsive forces (e.g., proportional to 1/dist²) and wall repulsion; move each centre a small step along the net force, then increase all radii by a tiny inflation factor while clipping to keep circles inside the unit square.",
    "After each move, resolve any remaining overlaps by a simple pairwise correction (push overlapping circles apart along the line of centres) and re‑project centres that violate the square bounds.",
    "Terminate after a fixed number of iterations (e.g., 200) or when the increase in total radius falls below a tolerance, then return the final centres, radii, and their sum."
  ],
  "cautions": "The simulation must finish well within the 600 s evaluator timeout (preferably <2 s). Choose step sizes, inflation rate, and iteration count conservatively; too large steps cause instability, too many iterations waste time. Ensure deterministic behaviour by fixing the random seed, so the evaluator sees a reproducible layout.",
  "approach_type": "force_directed_relaxation"
}
```
