"""NOA (Nested Optimization Agents) — 多层嵌套优化框架。"""

from .auto_adapter import auto_adapt
from .engine import NOptimizer
from .orchestrator import Orchestrator
from .core.protocol import (
    SystemDescription,
    SourceFile,
    Trajectory,
    Diagnosis,
    DiffBlock,
    DeltaPatch,
    EvalResult,
    OptimizationBudget,
)

__all__ = [
    "auto_adapt",
    "NOptimizer",
    "Orchestrator",
    "SystemDescription",
    "SourceFile",
    "Trajectory",
    "Diagnosis",
    "DiffBlock",
    "DeltaPatch",
    "EvalResult",
    "OptimizationBudget",
]
