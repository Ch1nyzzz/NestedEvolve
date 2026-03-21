"""MetaEvolve: Skill-orchestrated meta-learning for evolutionary search."""

from .cli import main
from .config import load_config

__all__ = ["load_config", "main"]
