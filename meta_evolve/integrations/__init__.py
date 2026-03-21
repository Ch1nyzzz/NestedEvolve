"""External integrations and target-system adapters."""

from .llm import LLMClient
from .targets import AdaEvolveAdapter, NativeAdapter, OpenEvolveAdapter, get_adapter

__all__ = [
    "AdaEvolveAdapter",
    "LLMClient",
    "NativeAdapter",
    "OpenEvolveAdapter",
    "get_adapter",
]
