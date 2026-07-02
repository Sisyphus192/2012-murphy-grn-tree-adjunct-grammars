"""Tests for fitness evaluation."""

import pytest

from grntage.simulation.fitness import FitnessEvaluator, compute_fitness


class TestComputeFitness:
    """Tests for compute_fitness function."""

    def test_perfect_fitness(self) -> None:
        """Test fitness = 0 at 120000 steps."""
        fitness = compute_fitness(120000)
        assert fitness == pytest.approx(0.0)

    def test_zero_steps(self) -> None:
        """Test fitness = inf at 0 steps."""
        fitness = compute_fitness(0)
        assert fitness == float("inf")

    def test_one_step(self) -> None:
        """Test fitness at 1 step."""
        fitness = compute_fitness(1)
        assert fitness == pytest.approx(119999.0)

    def test_halfway(self) -> None:
        """Test fitness at 60000 steps."""
        fitness = compute_fitness(60000)
        assert fitness == pytest.approx(1.0)

    def test_lower_is_better(self) -> None:
        """Test that more steps = lower fitness."""
        fitness_100 = compute_fitness(100)
        fitness_1000 = compute_fitness(1000)
        fitness_10000 = compute_fitness(10000)
        assert fitness_100 > fitness_1000 > fitness_10000


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator class."""

    def test_compute(self) -> None:
        """Test compute method matches function."""
        evaluator = FitnessEvaluator()
        assert evaluator.compute(1000) == pytest.approx(compute_fitness(1000))

    def test_is_solution_perfect(self) -> None:
        """Test perfect solution is recognized."""
        evaluator = FitnessEvaluator()
        assert evaluator.is_solution(0.0)

    def test_is_solution_negative(self) -> None:
        """Test negative fitness (overshoot) is solution."""
        evaluator = FitnessEvaluator()
        assert evaluator.is_solution(-0.5)

    def test_is_not_solution(self) -> None:
        """Test non-solution fitness."""
        evaluator = FitnessEvaluator()
        assert not evaluator.is_solution(1.0)

    def test_steps_for_fitness_zero(self) -> None:
        """Test steps calculation for fitness=0."""
        evaluator = FitnessEvaluator()
        steps = evaluator.steps_for_fitness(0.0)
        assert steps == 120000

    def test_steps_for_fitness_one(self) -> None:
        """Test steps calculation for fitness=1."""
        evaluator = FitnessEvaluator()
        steps = evaluator.steps_for_fitness(1.0)
        assert steps == 60000

    def test_steps_inverse(self) -> None:
        """Test steps_for_fitness is inverse of compute."""
        evaluator = FitnessEvaluator()
        for target_steps in [100, 1000, 10000, 60000, 120000]:
            fitness = evaluator.compute(target_steps)
            if fitness >= 0:
                recovered_steps = evaluator.steps_for_fitness(fitness)
                # Allow small rounding error
                assert recovered_steps == pytest.approx(target_steps, rel=0.01)
