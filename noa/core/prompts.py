"""所有 LLM agent 的 prompt 模板 — 保持简洁，便于 L2 优化。"""

# --- Initiator ---

INITIATOR_SYSTEM = "You analyze AI pipeline source code."

INITIATOR_PROMPT = """\
## Source Code
{source_code}

{system_description}

Analyze the source code and identify:
1. Workflow summary (component order and data flow)
2. Component names in execution order

Output JSON:
{{"workflow_summary": "...", "component_names": [...]}}
"""

# --- Analyzer ---

ANALYZER_SYSTEM = "You are an expert error analyst for AI pipelines. You can read source code to diagnose issues."

ANALYZER_PROMPT = """\
## System
{system_context}

## Source Code
{source_code}

## Failed Trajectories ({n_failures} / {n_total})
{trajectories}

## Past Attempts
{past_attempts}

Analyze the failures by reading the source code. Avoid repeating past failed approaches. For each pattern:
1. Pattern description
2. Root cause (which component, which code, why)
3. Severity (high/medium/low)
4. Affected file (which source file contains the problematic code)
5. Suggested fix (brief description of what code change would help)

Output JSON: [{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "...", "affected_file": "...", "suggested_fix": "..."}}]
"""

# --- Single-Trajectory Analyzer (incremental pool-based) ---

SINGLE_ANALYZER_SYSTEM = "You are an expert error analyst for AI pipelines. You diagnose one failure at a time and categorize it."

SINGLE_ANALYZER_PROMPT = """\
## System
{system_context}

## Source Code
{source_code}

## Current Failure (1 sample)
{trajectory}

## Existing Failure Pool
{pool_context}

## Past Attempts
{past_attempts}

Analyze this single failure by reading the source code.

If the failure matches an EXISTING pattern in the pool, output `merge_to` with the pool index.
If it is a NEW pattern, omit `merge_to`.

Output JSON (a list with one or a few pattern objects):
[{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "high|medium|low", "affected_file": "...", "suggested_fix": "...", "merge_to": <int or null>}}]
"""

# --- Optimizer ---

OPTIMIZER_SYSTEM = "You are a precise code optimizer. You output SEARCH/REPLACE diffs."

OPTIMIZER_PROMPT = """\
## System
{system_context}

## Source Code
{source_code}

## Diagnosis
{diagnosis}

## Past Attempts
{past_attempts}

Generate a minimal code patch for the following pattern. Do NOT repeat past rejected patches.

Rules:
- SEARCH block must exactly match existing code (verbatim, including whitespace)
- Minimize changes — one patch should fix one specific issue
- Prefer modifying prompts and configuration values before changing logic

Output format — for each file you modify:

## File: <filename>
<<<<<<< SEARCH
exact original code
=======
replacement code
>>>>>>> REPLACE

After all diffs, output metadata as JSON:
```json
{{"rationale": "...", "target_patterns": ["..."]}}
```
"""

# --- Auto Adapter ---

AUTO_ADAPTER_SYSTEM = "You generate Python adapter code for AI pipelines."

AUTO_ADAPTER_PROMPT = """\
## Target System Source Code
```python
{source_code}
```

{entry_hint}

## Task
Generate a Python adapter that wraps this pipeline.

## Required Interface
The adapter class must be named `AutoAdapter` and implement:

1. `__init__(self)` — instantiate the target pipeline
2. `__call__(self, question: str)` — run the pipeline, return an object with `.answer` (str) and `.intermediate` (dict)

## Reference
```python
from dataclasses import dataclass, field

@dataclass
class AdapterResult:
    answer: str
    intermediate: dict = field(default_factory=dict)

class AutoAdapter:
    def __init__(self):
        from some_module import Pipeline
        self.pipeline = Pipeline()

    def __call__(self, question: str) -> AdapterResult:
        result = self.pipeline.run(question)
        return AdapterResult(answer=result.output, intermediate={{}})
```

## Rules
- Output ONLY valid Python code, no markdown fences, no explanation
- The code must be self-contained (include all necessary imports)
- Use `sys.path.insert` if needed to import from the target directory
- Use `AdapterResult` dataclass for the return type of `__call__`
- Do NOT modify the target source code
"""

# --- L2 Meta-Analyzer ---

L2_META_ANALYZER_SYSTEM = "You are a meta-optimizer that analyzes and improves AI optimization systems by studying their historical performance."

L2_META_ANALYZER_PROMPT = """\
## NOA Optimizer Source Code
{source_code}

## L1 Optimizer Run History
{l1_history}

## Past L2 Attempts
{past_attempts}

You are a meta-optimizer analyzing the NOA optimization framework (the code above) based on its L1 run history.

Study the L1 history and source code to identify improvements. Focus on:
1. Did the Analyzer miss recurring failure patterns? Are its prompts or parsing insufficient?
2. Did the Optimizer generate ineffective patches? Are its diff strategies or prompts flawed?
3. Are algorithm parameters (n_samples, thresholds, temperature) suboptimal?
4. Are there systematic issues in the I-O-A-O-E loop logic?

For each issue found, provide:
1. Pattern description
2. Root cause (which component, which code, why)
3. Severity (high/medium/low)
4. Affected file (which source file contains the problematic code)
5. Suggested fix (brief description of what code change would help)

Avoid repeating past L2 attempts that were rejected.

Output JSON: [{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "...", "affected_file": "...", "suggested_fix": "..."}}]
"""
