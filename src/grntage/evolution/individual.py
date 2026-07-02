"""Individual representation for evolutionary algorithm.

Each individual has a genome (bit string) and associated fitness.
"""

from grntage.grn.genome import Genome
from grntage.grn.network import GRN


class Individual:
    """An individual in the evolutionary population.

    Attributes:
        genome: The genome (bit string representation)
        fitness: Fitness value (lower is better)
        generation: Generation this individual was created
    """

    def __init__(
        self,
        genome: Genome,
        fitness: float = float("inf"),
        generation: int = 0,
    ) -> None:
        """Initialize an individual.

        Args:
            genome: The genome
            fitness: Initial fitness (default inf = unevaluated)
            generation: Generation created (default 0)
        """
        self.genome = genome
        self.fitness = fitness
        self.generation = generation
        self._grn: GRN | None = None

    @classmethod
    def random(cls, num_bits: int = 4096, generation: int = 0) -> "Individual":
        """Create an individual with a random genome.

        Args:
            num_bits: Size of genome in bits (default 4096)
            generation: Generation created

        Returns:
            New individual with random genome
        """
        genome = Genome.random(num_bits)
        return cls(genome=genome, generation=generation)

    @property
    def grn(self) -> GRN:
        """Get the GRN for this individual, creating if needed."""
        if self._grn is None:
            self._grn = GRN(self.genome)
        return self._grn

    def reset_grn(self) -> None:
        """Reset the GRN to initial state."""
        if self._grn is not None:
            self._grn.reset()

    def invalidate_grn(self) -> None:
        """Invalidate cached GRN (call after mutation)."""
        self._grn = None

    def copy(self) -> "Individual":
        """Create a copy of this individual.

        Returns:
            New individual with copied genome
        """
        new_genome = self.genome.copy()
        return Individual(
            genome=new_genome,
            fitness=self.fitness,
            generation=self.generation,
        )

    def is_evaluated(self) -> bool:
        """Check if this individual has been evaluated.

        Returns:
            True if fitness is not infinity
        """
        return self.fitness != float("inf")

    def __lt__(self, other: "Individual") -> bool:
        """Compare by fitness (lower is better)."""
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        """String representation."""
        return f"Individual(fitness={self.fitness:.4f}, gen={self.generation})"
