"""Optimizer package.

Concrete optimizer classes are eagerly imported here so the Optimizer ABC's
__init_subclass__ hook fires and registers them, regardless of whether
the caller imports them directly.
"""

from ai.optimizers.base import Optimizer
from ai.optimizers.nsga2 import NSGAIIOptimizer  # noqa: F401 — import for registration
# ai.optimizers.ccmo is imported in a later commit; until then, only nsga2 registers.

__all__ = ["Optimizer", "NSGAIIOptimizer"]
