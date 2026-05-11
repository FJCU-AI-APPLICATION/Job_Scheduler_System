"""Tests for the Optimizer ABC and __init_subclass__ auto-registration."""

import pytest

from ai.optimizers.base import Optimizer


def test_init_subclass_registers(tiny_problem):
    """Concrete subclasses with a 'name' attribute are auto-registered."""
    available = Optimizer.list_available()
    assert "nsga2" in available
    assert "ccmo" in available
    assert "cpsat" in available
    assert "matheuristic" in available
    assert "last_rl" in available


def test_create_returns_concrete_optimizer(tiny_problem):
    """Optimizer.create() returns an instance of the registered class."""
    from ai.optimizers.nsga2 import NSGAIIOptimizer

    optimizer = Optimizer.create("nsga2", tiny_problem)
    assert isinstance(optimizer, NSGAIIOptimizer)


def test_create_unknown_raises(tiny_problem):
    """Asking for an unknown algorithm raises ValueError listing valid choices."""
    with pytest.raises(ValueError) as exc:
        Optimizer.create("does-not-exist", tiny_problem)
    assert "Unknown optimizer" in str(exc.value)
    assert "nsga2" in str(exc.value)
    assert "ccmo" in str(exc.value)
    assert "cpsat" in str(exc.value)
    assert "matheuristic" in str(exc.value)
    assert "last_rl" in str(exc.value)


def test_list_available_returns_sorted_names():
    """list_available() returns names in sorted order for stable display."""
    names = Optimizer.list_available()
    assert names == sorted(names)


def test_duplicate_name_raises():
    """Defining a second class with an existing name raises ValueError."""

    with pytest.raises(ValueError):
        class DuplicateNSGA2(Optimizer):
            name = "nsga2"  # collides with the existing NSGAIIOptimizer

            def run(self, config=None, verbose=False):
                raise NotImplementedError


def test_abstract_intermediate_class_skips_registration():
    """A subclass with name='' (abstract intermediate) does NOT register."""
    pre_count = len(Optimizer.list_available())

    class AbstractMid(Optimizer):
        # No 'name' override; inherits the empty default. Should NOT register.
        def run(self, config=None, verbose=False):
            raise NotImplementedError

    post_count = len(Optimizer.list_available())
    assert post_count == pre_count
