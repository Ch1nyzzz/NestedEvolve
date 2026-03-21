# Target Task Starter

This is the global starter skill for MetaEvolve.

Purpose:
- Run once for each new task before normal evolution starts
- Analyze the task objective, evaluator, constraints, and candidate method families
- Produce task-specific guidance that gets injected into the evolver prompt

Runtime behavior:
- Metadata lives in `skill.json`
- The executable hook lives in `hook.py`
- Per-task results are cached in the shared task artifact store

Expected output:
- A JSON payload containing general diagnosis guidance for early-stage search
- Candidate method families to try first
- Evaluator focus points and hard constraints
- A concrete early-phase search direction for evolver
