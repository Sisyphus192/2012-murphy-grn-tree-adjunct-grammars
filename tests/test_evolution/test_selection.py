"""Tests for selection operators."""

import random


from grntage.evolution.individual import Individual
from grntage.evolution.population import Population
from grntage.evolution.selection import TournamentSelector


class TestTournamentSelector:
    """Tests for TournamentSelector class."""

    def test_default_tournament_size(self) -> None:
        """Test default tournament size."""
        selector = TournamentSelector()
        assert selector.tournament_size == 3

    def test_custom_tournament_size(self) -> None:
        """Test custom tournament size."""
        selector = TournamentSelector(tournament_size=5)
        assert selector.tournament_size == 5

    def test_select_one(self) -> None:
        """Test selecting one individual."""
        random.seed(42)
        pop = Population.random(size=10)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)

        selector = TournamentSelector(tournament_size=3)
        selected = selector.select_one(pop)
        assert isinstance(selected, Individual)
        # Selected should be one of the best 3 (tournament picks best)
        assert selected.fitness < 10.0

    def test_select_many(self) -> None:
        """Test selecting multiple individuals."""
        random.seed(42)
        pop = Population.random(size=20)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)

        selector = TournamentSelector(tournament_size=3)
        selected = selector.select_many(pop, 5)
        assert len(selected) == 5
        for s in selected:
            assert isinstance(s, Individual)

    def test_selection_bias_toward_fitter(self) -> None:
        """Test that selection is biased toward fitter individuals."""
        random.seed(42)
        pop = Population.random(size=100)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)  # 0 is best, 99 is worst

        selector = TournamentSelector(tournament_size=5)
        selected = selector.select_many(pop, 100)

        # Average fitness of selected should be lower than population average
        pop_avg = sum(ind.fitness for ind in pop) / len(pop)
        sel_avg = sum(ind.fitness for ind in selected) / len(selected)
        assert sel_avg < pop_avg

    def test_select_parents(self) -> None:
        """Test selecting parents for reproduction."""
        random.seed(42)
        pop = Population.random(size=50)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)

        selector = TournamentSelector(tournament_size=3)
        parents = selector.select_parents(pop, num_offspring=25)
        assert len(parents) == 25
