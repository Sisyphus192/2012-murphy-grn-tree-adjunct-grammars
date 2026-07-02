"""Tests for Individual class."""

from grntage.evolution.individual import Individual
from grntage.grn.genome import Genome


class TestIndividual:
    """Tests for Individual class."""

    def test_create_with_genome(self) -> None:
        """Test creating an individual with a genome."""
        genome = Genome.random(4096)
        ind = Individual(genome=genome)
        assert ind.genome is genome
        assert ind.fitness == float("inf")
        assert ind.generation == 0

    def test_create_random(self) -> None:
        """Test creating a random individual."""
        ind = Individual.random(num_bits=4096, generation=5)
        assert ind.genome is not None
        assert ind.genome.length == 4096
        assert ind.fitness == float("inf")
        assert ind.generation == 5

    def test_grn_property(self) -> None:
        """Test that GRN is lazily created."""
        ind = Individual.random()
        assert ind._grn is None
        grn = ind.grn  # Access triggers creation
        assert grn is not None
        assert ind._grn is grn

    def test_reset_grn(self) -> None:
        """Test resetting the GRN."""
        ind = Individual.random()
        _ = ind.grn  # Create GRN
        ind.reset_grn()
        # After reset, GRN should still exist
        assert ind._grn is not None

    def test_invalidate_grn(self) -> None:
        """Test invalidating the GRN."""
        ind = Individual.random()
        _ = ind.grn  # Create GRN
        assert ind._grn is not None
        ind.invalidate_grn()
        assert ind._grn is None

    def test_copy(self) -> None:
        """Test copying an individual."""
        ind = Individual.random()
        ind.fitness = 1.5
        ind.generation = 3

        copy = ind.copy()
        assert copy is not ind
        assert copy.genome is not ind.genome
        assert copy.genome.bits == ind.genome.bits
        assert copy.fitness == ind.fitness
        assert copy.generation == ind.generation

    def test_is_evaluated(self) -> None:
        """Test is_evaluated method."""
        ind = Individual.random()
        assert not ind.is_evaluated()
        ind.fitness = 0.5
        assert ind.is_evaluated()

    def test_comparison_by_fitness(self) -> None:
        """Test that individuals compare by fitness."""
        ind1 = Individual.random()
        ind2 = Individual.random()
        ind1.fitness = 1.0
        ind2.fitness = 2.0
        assert ind1 < ind2
        assert not ind2 < ind1

    def test_repr(self) -> None:
        """Test string representation."""
        ind = Individual.random()
        ind.fitness = 1.234
        ind.generation = 5
        s = repr(ind)
        assert "1.234" in s
        assert "5" in s
