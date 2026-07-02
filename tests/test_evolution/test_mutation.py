"""Tests for mutation operators."""

import random


from grntage.evolution.individual import Individual
from grntage.evolution.mutation import BitMutator


class TestBitMutator:
    """Tests for BitMutator class."""

    def test_default_mutation_rate(self) -> None:
        """Test default mutation rate."""
        mutator = BitMutator()
        assert mutator.mutation_rate == 0.005

    def test_custom_mutation_rate(self) -> None:
        """Test custom mutation rate."""
        mutator = BitMutator(mutation_rate=0.01)
        assert mutator.mutation_rate == 0.01

    def test_mutate_creates_copy(self) -> None:
        """Test that mutate creates a new individual."""
        random.seed(42)
        ind = Individual.random(num_bits=100)
        mutator = BitMutator(mutation_rate=0.1)

        mutant = mutator.mutate(ind)
        assert mutant is not ind
        assert mutant.genome is not ind.genome

    def test_mutate_changes_bits(self) -> None:
        """Test that mutate changes some bits with high rate."""
        random.seed(42)
        ind = Individual.random(num_bits=1000)
        original_bits = ind.genome.bits
        mutator = BitMutator(mutation_rate=0.5)  # High rate

        mutant = mutator.mutate(ind)
        # Should have many different bits
        diff = original_bits ^ mutant.genome.bits
        num_different = bin(diff).count("1")
        assert num_different > 100  # At least some mutations

    def test_mutate_resets_fitness(self) -> None:
        """Test that mutate resets fitness to inf."""
        random.seed(42)
        ind = Individual.random()
        ind.fitness = 1.0
        mutator = BitMutator()

        mutant = mutator.mutate(ind)
        assert mutant.fitness == float("inf")

    def test_mutate_invalidates_grn(self) -> None:
        """Test that mutate invalidates GRN cache."""
        random.seed(42)
        ind = Individual.random()
        _ = ind.grn  # Create GRN
        assert ind._grn is not None

        mutator = BitMutator()
        mutant = mutator.mutate(ind)
        assert mutant._grn is None

    def test_mutate_preserves_original(self) -> None:
        """Test that mutate doesn't modify original."""
        random.seed(42)
        ind = Individual.random()
        original_bits = ind.genome.bits
        original_fitness = 1.0
        ind.fitness = original_fitness

        mutator = BitMutator(mutation_rate=0.5)
        _ = mutator.mutate(ind)

        assert ind.genome.bits == original_bits
        assert ind.fitness == original_fitness

    def test_mutate_in_place(self) -> None:
        """Test in-place mutation."""
        random.seed(42)
        ind = Individual.random(num_bits=1000)
        original_bits = ind.genome.bits
        ind.fitness = 1.0

        mutator = BitMutator(mutation_rate=0.5)
        num_mutations = mutator.mutate_in_place(ind)

        assert num_mutations > 100  # Many mutations at 50%
        assert ind.genome.bits != original_bits
        assert ind.fitness == float("inf")

    def test_zero_mutation_rate(self) -> None:
        """Test that zero mutation rate doesn't change bits."""
        random.seed(42)
        ind = Individual.random()
        original_bits = ind.genome.bits

        mutator = BitMutator(mutation_rate=0.0)
        mutant = mutator.mutate(ind)

        assert mutant.genome.bits == original_bits

    def test_mutation_rate_approximation(self) -> None:
        """Test that actual mutation rate approximates expected."""
        random.seed(42)
        num_bits = 10000
        expected_rate = 0.01
        ind = Individual.random(num_bits=num_bits)
        original_bits = ind.genome.bits

        mutator = BitMutator(mutation_rate=expected_rate)
        mutant = mutator.mutate(ind)

        diff = original_bits ^ mutant.genome.bits
        num_mutations = bin(diff).count("1")
        actual_rate = num_mutations / num_bits

        # Should be within 50% of expected rate
        assert 0.005 < actual_rate < 0.015
