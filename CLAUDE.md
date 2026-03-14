# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: NOA (Nested Optimization Agents)

A multi-layer nested agent optimization framework. Higher-order agents autonomously observe, diagnose, and incrementally patch lower-order agents through a standardized closed-loop.

**Core Thesis:** _The quality of optimization is bottlenecked by the quality of error analysis. Better diagnosis always leads to better patches._

## Architecture

### Unified Agentic Loop

Each optimization layer is driven by a single persistent LLM agent (`UnifiedOptimizerAgent`) with full tool access. The agent autonomously decides its workflow:

**DISCOVER → LOCALIZE → EXPERIMENT → VERIFY → COMMIT → REFLECT**

The agent can loop back from any step. It has access to a complete tool matrix (observe, analyze, read/search source, apply patch, checkpoint/eval/accept candidates, spawn sublayer, etc.) and drives the entire optimization process through tool calls.

- **No fixed stage pipeline** — the agent decides when to observe, analyze, patch, or evaluate
- **Persistent context** — the agent maintains state across all iterations within a single agentic loop
- **Diagnosis is still the core focus** — the agent must spend most compute on understanding WHY things fail before patching

### Nesting Hierarchy

```
L0: Target Agent System (the workflow being optimized)
L1: Optimizer (observes & patches L0)
L2: Meta-Optimizer (observes & patches L1)
```

Each layer treats the layer below as a "target system" via the **Unified System Description Protocol** — a standardized metadata structure covering: system topology, mutable configuration, historical trajectories, scores/metrics, and safety constraints.

### Key Constraints

- All modifications are **Delta Patches** — never full rewrites
- Every patch must pass **sandbox validation** before committing
- Rollback is always available if a patch degrades performance

## Success Metrics (PoC)

- **Task Performance Lift** — L1+L2 nested optimization must statistically outperform static baselines and single-layer optimization
- **Patch Acceptance Rate** — proportion of patches that pass sandbox and merge into live config
- **Context Convergence Stability** — config size must stay flat or sub-linear over iterations

## Conventions

- Always respond in Chinese (用中文回答)
- Keep code concise, correct, and efficient
- Protocol format: standardized JSON metadata
- Budget-bounded iteration loops to prevent runaway compute
- **Occam's Razor** — do not introduce unnecessary concepts at the design level
- **Agent prompts stay concise** — easier for upper layers to optimize and diff
- **Unified model** — all LLM agents default to `together_ai/moonshotai/Kimi-K2.5`

## Memory Bank

- `memory_bank/PRD.md` — the source-of-truth product requirements document
- `memory_bank/progress.md` — development progress log
- **Rule: Every time framework code is modified, record what changed and why in `memory_bank/progress.md`.**

## metaswarm

This project uses [metaswarm](https://github.com/dsifry/metaswarm) for multi-agent orchestration.

**Setup:** Run `/metaswarm-setup` to detect your project and configure metaswarm.

**Update:** Run `/metaswarm-update-version` to update metaswarm.
