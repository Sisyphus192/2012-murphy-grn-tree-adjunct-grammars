"""Population management for evolutionary algorithm."""

from typing import Iterator

from grntage.evolution.individual import Individual


class Population:
    """A population of individuals.

    Provides methods for managing a collection of individuals
    including sorting, selection, and replacement.

    Attributes:
        individuals: List of individuals
        generation: Current generation number
    """

    def __init__(
        self,
        individuals: list[Individual] | None = None,
        generation: int = 0,
    ) -> None:
        """Initialize population.

        Args:
            individuals: Initial population (default empty)
            generation: Starting generation number
        """
        self.individuals = individuals or []
        self.generation = generation

    @classmethod
    def random(
        cls,
        size: int = 250,
        genome_bits: int = 4096,
    ) -> "Population":
        """Create a population with random individuals.

        Args:
            size: Population size (default 250)
            genome_bits: Bits per genome (default 4096)

        Returns:
            New population with random individuals
        """
        individuals = [
            Individual.random(num_bits=genome_bits, generation=0) for _ in range(size)
        ]
        return cls(individuals=individuals, generation=0)

    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals."""
        return iter(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]

    def add(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)

    def sort_by_fitness(self) -> None:
        """Sort individuals by fitness (best first, lower is better)."""
        self.individuals.sort(key=lambda x: x.fitness)

    def get_best(self, n: int = 1) -> list[Individual]:
        """Get the n best individuals.

        Args:
            n: Number of individuals to return

        Returns:
            List of n best individuals (by fitness)
        """
        self.sort_by_fitness()
        return self.individuals[:n]

    def get_worst(self, n: int = 1) -> list[Individual]:
        """Get the n worst individuals.

        Args:
            n: Number of individuals to return

        Returns:
            List of n worst individuals (by fitness)
        """
        self.sort_by_fitness()
        return self.individuals[-n:]

    def best_fitness(self) -> float:
        """Get the best fitness in the population."""
        if not self.individuals:
            return float("inf")
        return min(ind.fitness for ind in self.individuals)

    def worst_fitness(self) -> float:
        """Get the worst fitness in the population."""
        if not self.individuals:
            return float("inf")
        return max(ind.fitness for ind in self.individuals)

    def mean_fitness(self) -> float:
        """Get the mean fitness of the population."""
        if not self.individuals:
            return float("inf")
        total = sum(
            ind.fitness for ind in self.individuals if ind.fitness != float("inf")
        )
        count = sum(1 for ind in self.individuals if ind.fitness != float("inf"))
        return total / count if count > 0 else float("inf")

    def replace(self, new_individuals: list[Individual]) -> None:
        """Replace population with new individuals.

        Args:
            new_individuals: New population
        """
        self.individuals = new_individuals
        self.generation += 1

    def reset_grns(self) -> None:
        """Reset all GRNs in the population."""
        for individual in self.individuals:
            individual.reset_grn()
