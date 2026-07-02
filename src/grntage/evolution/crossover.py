"""Crossover operators for evolutionary algorithm."""

import random

from grntage.evolution.individual import Individual


class OnePointCrossover:
    """One-point crossover operator for Grammatical Evolution.

    Recombines two parent genomes by selecting a random crossover point
    and swapping the high-order bits between the parents to produce two
    children.  Applied with probability ``crossover_rate``; otherwise the
    two copies are returned unchanged (clonal reproduction).

    Attributes:
        crossover_rate: Probability of performing crossover (default 0.9)
    """

    def __init__(self, crossover_rate: float = 0.9) -> None:
        """Initialize the crossover operator.

        Args:
            crossover_rate: Per-pair one-point crossover probability (default 0.9,
                standard GE).
        """
        self.crossover_rate = crossover_rate

    def cross(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual]:
        """Produce two children from two parents via one-point crossover.

        Args:
            parent1: First parent individual
            parent2: Second parent individual

        Returns:
            Pair of child individuals (copies of parents if no crossover occurs)
        """
        assert parent1.genome.length == parent2.genome.length, (
            "Parents must have the same genome length for crossover; "
            f"got {parent1.genome.length} and {parent2.genome.length}"
        )

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Skip recombination with probability (1 - crossover_rate)
        if random.random() >= self.crossover_rate:
            return child1, child2

        length = parent1.genome.length
        point = random.randint(1, length - 1)

        # Build masks: low_mask covers bits [0, point), high_mask covers [point, length)
        low_mask = (1 << point) - 1
        full_mask = (1 << length) - 1
        high_mask = full_mask ^ low_mask

        child1.genome.bits = (parent1.genome.bits & low_mask) | (
            parent2.genome.bits & high_mask
        )
        child2.genome.bits = (parent2.genome.bits & low_mask) | (
            parent1.genome.bits & high_mask
        )

        # Invalidate caches and mark as unevaluated
        child1.fitness = float("inf")
        child1.invalidate_grn()
        child2.fitness = float("inf")
        child2.invalidate_grn()

        return child1, child2
