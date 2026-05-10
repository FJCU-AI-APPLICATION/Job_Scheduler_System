"""Abstract base class for evolutionary optimizers with auto-registration.

Concrete subclasses are auto-registered when their class body executes,
via __init_subclass__. They must declare a 'name' class attribute.

Adding a new optimizer:
  1. Subclass Optimizer
  2. Set `name`, `config_class`, `result_class` class attributes
  3. Implement run()
  4. Import the module from src/ai/optimizers/__init__.py so the class is loaded
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from ai.domain.problem import SchedulingProblem
from ai.optimizers.result import OptimizerConfig, OptimizerResult


class Optimizer(ABC):
    """Abstract base for all evolutionary optimizers."""

    name: ClassVar[str] = ""
    config_class: ClassVar[type[OptimizerConfig]]
    result_class: ClassVar[type[OptimizerResult]]

    _registry: ClassVar[dict[str, type["Optimizer"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return  # abstract intermediate class, skip
        if cls.name in Optimizer._registry:
            raise ValueError(
                f"Optimizer '{cls.name}' is already registered by "
                f"{Optimizer._registry[cls.name].__name__}"
            )
        Optimizer._registry[cls.name] = cls

    def __init__(self, scheduling_problem: SchedulingProblem):
        self._sp = scheduling_problem

    @abstractmethod
    def run(
        self,
        config: OptimizerConfig | None = None,
        verbose: bool = False,
    ) -> OptimizerResult:
        ...

    @classmethod
    def create(cls, name: str, problem: SchedulingProblem) -> "Optimizer":
        if name not in cls._registry:
            raise ValueError(
                f"Unknown optimizer '{name}'. Available: {sorted(cls._registry)}"
            )
        return cls._registry[name](problem)

    @classmethod
    def list_available(cls) -> list[str]:
        return sorted(cls._registry)
