"""Mutation operators for evolutionary algorithm."""

import random

from grntage.evolution.individual import Individual


class BitMutator:
    """Bit mutation operator.

    Flips each bit in the genome with a given probability.

    Attributes:
        mutation_rate: Probability of flipping each bit
    """

    def __init__(self, mutation_rate: float = 0.005) -> None:
        """Initialize the mutator.

        Args:
            mutation_rate: Per-bit mutation probability (default 0.005)
        """
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual) -> Individual:
        """Create a mutated copy of an individual.

        Args:
            individual: Individual to mutate

        Returns:
            New individual with mutated genome
        """
        # Copy the individual
        mutant = individual.copy()

        # Mutate each bit with probability mutation_rate
        # genome.bits is an integer, not bytes
        bits = mutant.genome.bits
        num_bits = mutant.genome.length

        for bit_idx in range(num_bits):
            if random.random() < self.mutation_rate:
                # Flip the bit using XOR
                bits ^= 1 << bit_idx

        mutant.genome.bits = bits

        # Invalidate cached GRN (genome changed)
        mutant.invalidate_grn()

        # Reset fitness (needs re-evaluation)
        mutant.fitness = float("inf")

        return mutant

    def mutate_in_place(self, individual: Individual) -> int:
        """Mutate an individual in place.

        Args:
            individual: Individual to mutate

        Returns:
            Number of bits mutated
        """
        bits = individual.genome.bits
        num_bits = individual.genome.length
        num_mutations = 0

        for bit_idx in range(num_bits):
            if random.random() < self.mutation_rate:
                bits ^= 1 << bit_idx
                num_mutations += 1

        individual.genome.bits = bits
        individual.invalidate_grn()
        individual.fitness = float("inf")

        return num_mutations
