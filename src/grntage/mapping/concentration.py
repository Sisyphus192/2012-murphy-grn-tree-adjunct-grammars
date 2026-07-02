"""Concentration Value output mapping method.

Binary method: Uses a single P-protein's concentration.
If concentration > 0.5, produces codon 1, else codon 0.
"""

from grntage.grn.protein import Protein
from grntage.mapping.base import OutputMapper, OutputMethod


class ConcentrationValueMapper(OutputMapper):
    """Maps P-protein concentration to binary codon.

    Uses the first P-protein's concentration with a threshold of 0.5.
    Produces codon 1 if concentration > 0.5, else codon 0.

    Attributes:
        threshold: Concentration threshold (default 0.5)
        protein_index: Index of P-protein to use (default 0)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        protein_index: int = 0,
    ) -> None:
        """Initialize the mapper.

        Args:
            threshold: Concentration threshold for binary decision
            protein_index: Which P-protein to use (by index)
        """
        self.threshold = threshold
        self.protein_index = protein_index

    @property
    def method(self) -> OutputMethod:
        """Get the output method type."""
        return OutputMethod.CONCENTRATION_VALUE

    def map_to_codons(
        self,
        proteins: list[Protein],
        previous_concentrations: dict[int, float] | None = None,
    ) -> list[int]:
        """Map P-protein concentration to binary codon.

        Args:
            proteins: List of P-proteins
            previous_concentrations: Not used for this method

        Returns:
            List containing single codon (0 or 1)
        """
        if not proteins:
            return [0]

        # Get the target protein
        idx = min(self.protein_index, len(proteins) - 1)
        protein = proteins[idx]

        # Binary decision based on threshold
        codon = 1 if protein.concentration > self.threshold else 0
        return [codon]

    def is_binary(self) -> bool:
        """Check if this mapper produces binary output.

        Returns:
            True (this is a binary method)
        """
        return True
