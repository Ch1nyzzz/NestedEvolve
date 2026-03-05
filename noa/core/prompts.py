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
{layer_context}"""

# --- Analyzer ---

ANALYZER_SYSTEM = "You are an expert system analyst with full visibility into source code, configuration, and runtime behavior."

ANALYZER_PROMPT = """\
## Source Code & Configuration
{source_code}

## System Overview
{system_context}

## Execution Trajectories ({n_failures} failures / {n_total} total)
Each trajectory includes component outputs, _artifacts (model config, token usage, finish_reason), and ERROR tracebacks when execution failed.
{trajectories}

## Trajectory Data Quality
{trajectory_quality}

## Past Attempts
{past_attempts}

You have full visibility into the system: source code, configuration files, runtime behavior, and LLM call metadata (_artifacts).
Identify the root causes of failures. Do not repeat past failed approaches. For each pattern found:
1. Pattern description
2. Root cause (which component, which code or config, why)
3. Severity (high/medium/low)
4. Affected file
5. Suggested fix

If you are a meta-optimizer (layer_context is present), the `affected_file` MUST be a file within your writable scope (the optimizer framework). Do NOT reference target system files.

Output JSON: [{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "...", "affected_file": "...", "suggested_fix": "..."}}]
{layer_context}"""

# --- Single-Trajectory Analyzer (incremental pool-based) ---

SINGLE_ANALYZER_SYSTEM = "You are an expert system analyst. You diagnose one failure at a time using all available context."

SINGLE_ANALYZER_PROMPT = """\
## Source Code & Configuration
{source_code}

## System Overview
{system_context}

## Current Failure (1 sample)
The trajectory below includes component outputs, _artifacts (model config, token usage, finish_reason), and ERROR tracebacks when execution failed.
{trajectory}

## Trajectory Data Quality
{trajectory_quality}

## Existing Failure Pool
{pool_context}

## Past Attempts
{past_attempts}

You have full visibility into the system. Diagnose this failure using source code, configuration, and runtime metadata.

If the failure matches an EXISTING pattern in the pool, output `merge_to` with the pool index.
If it is a NEW pattern, omit `merge_to`.

If you are a meta-optimizer (layer_context is present), the `affected_file` MUST be a file within your writable scope (the optimizer framework). Do NOT reference target system files.

Output JSON (a list with one or a few pattern objects):
[{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "high|medium|low", "affected_file": "...", "suggested_fix": "...", "merge_to": <int or null>}}]
{layer_context}"""

# --- Agentic Analyzer (with tool calling) ---

AGENTIC_ANALYZER_SYSTEM = "You are an expert system analyst with experimental capabilities. You can run individual components to verify hypotheses."

AGENTIC_ANALYZER_PROMPT = """\
## Source Code & Configuration
{source_code}

## System Overview
{system_context}

## Current Failure (1 sample)
{trajectory}

## Trajectory Data Quality
{trajectory_quality}

## Existing Failure Pool
{pool_context}

## Past Attempts
{past_attempts}

You have tools to run individual components of the target system for micro-experiments.

## Workflow
1. Review the failure trajectory and source code
2. Form a root-cause hypothesis
3. Use tools to run micro-experiments verifying your hypothesis (e.g., re-run a component with cleaned inputs)
4. Confirm or reject the hypothesis based on experimental results
5. Output only experimentally verified failure patterns

## Rules
- Experiment before concluding — do not guess
- Compare successful vs failed samples, isolate variables
- Focus on component boundaries (where does output from one component break the next?)
- Each experiment should test ONE specific hypothesis
- If trajectory evidence is incomplete (missing `intermediate`), diagnose the observability gap first

If you are a meta-optimizer (layer_context is present), the `affected_file` MUST be a file within your writable scope (the optimizer framework). Do NOT reference target system files.

When done, output your diagnosis as JSON:
[{{"pattern": "...", "root_cause": "...", "affected_component": "...", "severity": "high|medium|low", "affected_file": "...", "suggested_fix": "...", "merge_to": <int or null>}}]
{layer_context}"""

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

## Representative Failure Samples
{trajectory_samples}

Generate a minimal code patch for the following pattern. Do NOT repeat past rejected patches.
Past attempts include evaluation results (scores, errors, low-score sample details) — use this feedback to avoid repeating ineffective approaches.
Study the failure samples above — they show real component inputs/outputs. Trace the data flow across component boundaries to find the actual root cause.

Rules:
- SEARCH block must exactly match existing code (verbatim, including whitespace)
- Minimize changes — one patch should fix one specific issue
- Prefer modifying prompts and configuration values before changing logic
- You can ONLY modify files listed in the Source Code section above. If the diagnosis mentions files not in your source code, translate the fix to the corresponding optimizer framework file.

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
{layer_context}"""

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

    def get_components(self) -> dict:
        \"\"\"Return {{component_name: component_object}} dict.
        Each component must have a forward(**kwargs) -> dict method.\"\"\"
        return self.pipeline.get_components()
```

## Rules
- Output ONLY valid Python code, no markdown fences, no explanation
- The code must be self-contained (include all necessary imports)
- Use `sys.path.insert` if needed to import from the target directory
- Use `AdapterResult` dataclass for the return type of `__call__`
- Implement `get_components()` returning a dict of {{name: component}} where each component has a `forward(**kwargs)` method
- Do NOT modify the target source code
"""
