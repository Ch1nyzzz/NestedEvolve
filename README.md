# NestedEvolve (NOA — Nested Optimization Agents)

## What are you trying to achieve?

We are building a framework that enables AI systems to autonomously improve themselves through code-level modifications, not just prompt tuning or hyperparameter adjustments. Most existing optimization approaches treat AI systems as black boxes and only tweak surface-level configurations. NestedEvolve goes deeper: it reads the actual source code of a target AI system, runs it on real data, diagnoses failure patterns from execution traces, and generates precise code patches (SEARCH/REPLACE blocks) to fix the root causes. Each patch is evaluated in a sandboxed environment and only accepted if it measurably improves performance.

What makes this project unique is its nested, recursive architecture. The system operates in two layers: L1 (the Optimizer) analyzes and patches the target system, while L2 (the Meta-Optimizer) analyzes L1's own optimization history and patches L1's code to make it a better optimizer. This creates a self-improving loop where the optimization framework itself evolves over time. In our experiments on a HotpotQA retrieval-augmented generation pipeline, this approach has achieved up to 49.2% improvement in answer F1 score, with L2 meta-optimization contributing meaningfully by fixing framework bugs and enhancing diagnostic prompts.

Ultimately, we aim to demonstrate that LLM-driven, code-level, nested optimization is a viable and powerful paradigm for making AI systems that can diagnose their own weaknesses and fix them autonomously.
