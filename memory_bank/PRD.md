# NOA (Nested Optimization Agents) — Multi-Layer Nested Evolution Framework

**Document Status:** Proof of Concept (PoC)
**Core Positioning:** A scalable agent optimization engine with self-correction, incremental evolution, and meta-cognitive capabilities.
**Core Thesis:** _The quality of optimization is bottlenecked by the quality of error analysis. Better diagnosis always leads to better patches._

---

## 1. Project Overview

NOA is a standardized multi-layer nested agent optimization framework. It addresses the scalability bottleneck of complex LLM workflows that currently rely on manual prompt engineering and hand-tuned configurations. By establishing a **Unified System Description Protocol** and a **Standardized Closed-Loop Optimization Paradigm**, higher-order agents can autonomously observe, diagnose, and incrementally modify the runtime configurations of lower-order agents. The ultimate goal is to achieve **cross-layer, long-horizon self-evolution** of target workflows — without continuous human intervention and with controlled compute cost.

---

## 2. Background & Problem Statement

Building and maintaining complex LLM-based systems incurs significant friction costs. Current self-improvement approaches expose three critical challenges when scaling:

1. **Prompt Fragility** — Manual prompt engineering does not generalize; small changes can cascade unpredictably.
2. **Knowledge Forgetting** — Full rewrites of agent configurations destroy accumulated knowledge and introduce destructive updates.
3. **Lack of Meta-Cognition** — Single-layer systems cannot reason about *why* they fail, limiting their ability to self-correct.

NOA solves these by introducing a nested architecture where each optimization layer operates on a well-defined abstraction of the layer below it.

> **Central Insight:** Every agent makes mistakes. The fundamental challenge is not *how to fix* an agent, but *how to understand what went wrong and why*. A shallow diagnosis (e.g., "the output was wrong") leads to superficial patches (e.g., "add a reminder to be more careful"), which rarely generalize. A deep diagnosis (e.g., "the agent conflated two user intents because the routing logic lacks a disambiguation step") leads to surgical, high-impact patches. **Error analysis is the highest-leverage activity in the entire optimization loop.**

---

## 3. Core Design Principles

### 3.1 Unified System Description Protocol

**Intent:** Lay the physical foundation for infinite nesting.

Any controlled agent system's state is abstracted into a **standardized metadata structure** covering:

| Component | Description |
|---|---|
| **System Topology** | The graph of agents, their roles, and interconnections |
| **Mutable Configuration** | Prompts, parameters, tool bindings, and other tunable knobs |
| **Historical Trajectories** | Execution traces, input/output logs, and intermediate reasoning |
| **Scores & Metrics** | Task success rates, latency, cost, and custom KPIs |
| **Safety Constraints** | Guardrails, invariants, and boundaries that must never be violated |

As long as a target system implements this protocol, any higher-layer optimizer can perform **state sensing** and **policy dispatch** through a consistent interface.

### 3.2 Unified Agentic Loop

**Intent:** Guarantee monotonically non-decreasing system performance; eliminate knowledge forgetting and destructive updates caused by global rewrites.

Each optimization layer is driven by a single persistent LLM agent (`UnifiedOptimizerAgent`) with full tool access. The agent autonomously decides its workflow through tool calls:

```
┌──────────────────────────────────────────────────────────┐
│              Unified Agentic Loop                        │
│                                                          │
│   DISCOVER → LOCALIZE → EXPERIMENT → VERIFY → COMMIT    │
│       ↑                                       │          │
│       └──────────── REFLECT ←─────────────────┘          │
│                                                          │
│   The agent can loop back from any step.                 │
│   No fixed stage pipeline — the LLM decides the flow.   │
│   ★ Diagnosis remains the highest-leverage activity.     │
└──────────────────────────────────────────────────────────┘

★ The agent must spend the majority of compute on understanding
  WHY things fail before generating patches.
```

**Diagnosis (Error Analysis) is still the project's core focus.** The ceiling of any optimization system is set by the depth of its error analysis. A correct root-cause diagnosis constrains the solution space to a small set of high-quality patches; a vague diagnosis explodes it, forcing the optimizer to guess blindly. Therefore:

- The agent MUST allocate the majority of its compute budget to analysis, not patch generation.
- Analysis must go beyond "what went wrong" to answer "where did it originate" and "why did it happen."
- Error traces from the System Description Protocol (§3.1) must be rich enough to support deep fault localization.

| Shallow Analysis | Deep Analysis |
|---|---|
| "Output was wrong" | "Agent used web_search instead of code_interpreter because routing prompt lacks a 'computation' intent category" |
| Patch: "Be more careful" | Patch: "Add 'computation/math' as explicit intent category with examples" |
| Fixes one instance | Fixes an entire failure class |

**Key constraints on the optimizer's action space:**

- All modifications MUST be issued as **local incremental patches (Delta Patches)** — never full rewrites.
- Every patch MUST pass a **sandbox trial run** before being committed to the live configuration.
- Rollback is always available if a patch degrades performance.

### 3.3 Cross-Layer Meta-Optimization

**Intent:** Empirically validate whether nested architecture can deliver compounding optimization gains.

The nesting hierarchy works as follows:

```
L0: Target Agent System (the workflow being optimized)
L1: First-Order Optimizer (observes & patches L0)
L2: Second-Order Meta-Optimizer (observes & patches L1)
...
Ln: Nth-Order Optimizer (observes & patches L(n-1))
```

Each layer treats the layer below it as a "target system" and runs its own `UnifiedOptimizerAgent` through the Unified System Description Protocol. This **homogeneous, recursive** structure means:

- The same agent + tool matrix applies at every layer.
- Adding a new layer requires no new architecture — just another agent instance with appropriate `LayerContext`.
- Higher layers can tune the *optimization strategy itself*, not just the target task.
- Critically, higher layers can improve the **analysis quality** of lower layers — better analysis → better patches → better performance → richer error signals → even better analysis. This compounding effect is the core value proposition of nesting.

---

## 4. Success Metrics (PoC Phase)

### 4.1 Task Performance Lift

On selected benchmarks (e.g., complex reasoning, tool-use tasks), a system with **L1 + L2 nested optimization** must demonstrate **statistically significant improvement** in final task success rate and convergence speed over:
- Static baselines (no optimization)
- Single-layer optimization (L1 only)

### 4.2 Patch Acceptance Rate

The proportion of optimizer-generated modification proposals that:
1. Pass sandbox validation
2. Are ultimately merged into the live configuration

This metric reflects the precision of the system's **diagnosis and attribution logic**.

### 4.3 Context Convergence Stability (Context Bloat Control)

Track the length/complexity curve of system configurations (e.g., prompt libraries) over dozens of iteration rounds. The system must demonstrate:
- Ability to incorporate new rules while maintaining **flat or sub-linear growth** in configuration size.
- Evidence that the **anti-collapse mechanism** is functioning — preventing unbounded prompt/config bloat.

---

## 5. Architecture Summary

```
┌───────────────────────────────────────────────────────┐
│  Ln: Meta-Meta-Optimizer                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │  L2: Meta-Optimizer (UnifiedOptimizerAgent)     │  │
│  │  ┌───────────────────────────────────────────┐  │  │
│  │  │  L1: Optimizer (UnifiedOptimizerAgent)    │  │  │
│  │  │  ┌─────────────────────────────────────┐  │  │  │
│  │  │  │  L0: Target Agent System            │  │  │  │
│  │  │  │  (prompts, tools, configs, logic)   │  │  │  │
│  │  │  └─────────────────────────────────────┘  │  │  │
│  │  │  Agentic loop on L0                       │  │  │
│  │  └───────────────────────────────────────────┘  │  │
│  │  Agentic loop on L1                             │  │
│  └─────────────────────────────────────────────────┘  │
│  Agentic loop on L2                                   │
└───────────────────────────────────────────────────────┘

Each layer communicates via the Unified System Description Protocol.
All modifications are Structured Patches validated in Sandboxes.
```

---

## 6. Key Design Decisions for PoC

| Decision | Choice | Rationale |
|---|---|---|
| Optimization granularity | Delta Patch (not full rewrite) | Prevents knowledge loss; enables rollback |
| Validation strategy | Sandbox trial before commit | Ensures monotonic improvement |
| Nesting depth for PoC | L0 + L1 + L2 (3 layers) | Sufficient to validate cross-layer gains |
| Protocol format | Standardized JSON metadata | Machine-readable, extensible, language-agnostic |
| Iteration control | Budget-bounded loops | Prevents runaway compute cost |

---

## 7. Open Questions

1. **Diminishing Returns** — At what nesting depth do additional layers stop providing meaningful improvement?
2. **Compute Budget Allocation** — How should compute be distributed across layers for optimal ROI?
3. **Conflict Resolution** — When patches from different optimization rounds conflict, what merge strategy should be used?
4. **Convergence Criteria** — How do we define "good enough" to stop iterating at each layer?
