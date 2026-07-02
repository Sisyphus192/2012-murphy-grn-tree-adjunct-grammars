"""Tests for Population class."""

from grntage.evolution.individual import Individual
from grntage.evolution.population import Population


class TestPopulation:
    """Tests for Population class."""

    def test_create_empty(self) -> None:
        """Test creating an empty population."""
        pop = Population()
        assert len(pop) == 0
        assert pop.generation == 0

    def test_create_with_individuals(self) -> None:
        """Test creating a population with individuals."""
        inds = [Individual.random() for _ in range(10)]
        pop = Population(individuals=inds, generation=5)
        assert len(pop) == 10
        assert pop.generation == 5

    def test_random_population(self) -> None:
        """Test creating a random population."""
        pop = Population.random(size=250, genome_bits=4096)
        assert len(pop) == 250
        for ind in pop:
            assert ind.genome.length == 4096

    def test_iteration(self) -> None:
        """Test iterating over population."""
        pop = Population.random(size=10)
        count = sum(1 for _ in pop)
        assert count == 10

    def test_indexing(self) -> None:
        """Test indexing into population."""
        pop = Population.random(size=10)
        ind = pop[0]
        assert isinstance(ind, Individual)

    def test_add(self) -> None:
        """Test adding an individual."""
        pop = Population()
        ind = Individual.random()
        pop.add(ind)
        assert len(pop) == 1
        assert pop[0] is ind

    def test_sort_by_fitness(self) -> None:
        """Test sorting by fitness."""
        pop = Population.random(size=10)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        pop.individuals.reverse()  # Worst first
        pop.sort_by_fitness()
        for i, ind in enumerate(pop):
            assert ind.fitness == float(i)

    def test_get_best(self) -> None:
        """Test getting best individuals."""
        pop = Population.random(size=10)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        best = pop.get_best(3)
        assert len(best) == 3
        assert best[0].fitness == 0.0
        assert best[1].fitness == 1.0
        assert best[2].fitness == 2.0

    def test_get_worst(self) -> None:
        """Test getting worst individuals."""
        pop = Population.random(size=10)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        worst = pop.get_worst(3)
        assert len(worst) == 3
        assert worst[0].fitness == 7.0
        assert worst[1].fitness == 8.0
        assert worst[2].fitness == 9.0

    def test_best_fitness(self) -> None:
        """Test getting best fitness."""
        pop = Population.random(size=5)
        for i, ind in enumerate(pop):
            ind.fitness = float(i + 1)
        assert pop.best_fitness() == 1.0

    def test_mean_fitness(self) -> None:
        """Test getting mean fitness."""
        pop = Population.random(size=5)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)  # 0, 1, 2, 3, 4
        assert pop.mean_fitness() == 2.0

    def test_replace(self) -> None:
        """Test replacing population."""
        pop = Population(generation=5)
        new_inds = [Individual.random() for _ in range(10)]
        pop.replace(new_inds)
        assert len(pop) == 10
        assert pop.generation == 6
