"""Selection operators for evolutionary algorithm."""

import random

from grntage.evolution.individual import Individual
from grntage.evolution.population import Population


class TournamentSelector:
    """Tournament selection operator.

    Selects individuals by running tournaments among random subsets.
    The best individual in each tournament is selected.

    Attributes:
        tournament_size: Number of individuals in each tournament
    """

    def __init__(self, tournament_size: int = 3) -> None:
        """Initialize the selector.

        Args:
            tournament_size: Size of each tournament (default 3)
        """
        self.tournament_size = tournament_size

    def select_one(self, population: Population) -> Individual:
        """Select one individual via tournament.

        Args:
            population: Population to select from

        Returns:
            Selected individual (winner of tournament)
        """
        # Select random contestants
        contestants = random.sample(population.individuals, self.tournament_size)

        # Return the best (lowest fitness)
        return min(contestants, key=lambda x: x.fitness)

    def select_many(self, population: Population, n: int) -> list[Individual]:
        """Select n individuals via tournaments.

        Args:
            population: Population to select from
            n: Number of individuals to select

        Returns:
            List of selected individuals
        """
        return [self.select_one(population) for _ in range(n)]

    def select_parents(
        self, population: Population, num_offspring: int
    ) -> list[Individual]:
        """Select parents for creating offspring.

        Each parent produces one offspring (mutation-only, no crossover).

        Args:
            population: Population to select from
            num_offspring: Number of offspring to create

        Returns:
            List of parent individuals
        """
        return self.select_many(population, num_offspring)
