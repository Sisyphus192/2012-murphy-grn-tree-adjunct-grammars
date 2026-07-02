"""Concentration Tendency output mapping method.

Binary method: Tracks the direction of concentration change.
Per the paper (sec. 3.1): a positive change -> codon 0, a negative change ->
codon 1. Reuses the previous codon if the change is below threshold.
"""

from grntage.grn.protein import Protein
from grntage.mapping.base import OutputMapper, OutputMethod


class ConcentrationTendencyMapper(OutputMapper):
    """Maps P-protein concentration change direction to binary codon.

    Tracks whether concentration is increasing or decreasing.
    Produces codon 0 if concentration increased (positive change), codon 1 if it
    decreased (negative change). If the change is below threshold, reuses the
    previous codon.

    Attributes:
        threshold: Minimum change magnitude to trigger update (default 1e-10)
        protein_index: Index of P-protein to use (default 0)
        previous_codon: Last codon produced (for reuse when change is small)
    """

    def __init__(
        self,
        threshold: float = 1e-10,
        protein_index: int = 0,
    ) -> None:
        """Initialize the mapper.

        Args:
            threshold: Minimum change magnitude to trigger codon update
            protein_index: Which P-protein to use (by index)
        """
        self.threshold = threshold
        self.protein_index = protein_index
        self.previous_codon = 0  # Default to 0 initially

    @property
    def method(self) -> OutputMethod:
        """Get the output method type."""
        return OutputMethod.CONCENTRATION_TENDENCY

    def map_to_codons(
        self,
        proteins: list[Protein],
        previous_concentrations: dict[int, float] | None = None,
    ) -> list[int]:
        """Map P-protein concentration change to binary codon.

        Args:
            proteins: List of P-proteins with current concentrations
            previous_concentrations: Dict mapping signature to previous concentration

        Returns:
            List containing single codon (0 or 1)
        """
        if not proteins:
            return [self.previous_codon]

        # Get the target protein
        idx = min(self.protein_index, len(proteins) - 1)
        protein = proteins[idx]

        # If no previous concentrations, return previous codon
        if previous_concentrations is None:
            return [self.previous_codon]

        # Get previous concentration for this protein
        prev_conc = previous_concentrations.get(
            protein.signature, protein.concentration
        )
        change = protein.concentration - prev_conc

        # Check if change is significant
        if abs(change) < self.threshold:
            return [self.previous_codon]

        # Determine codon based on change direction (paper sec. 3.1: a positive
        # change -> codon 0, a negative change -> codon 1).
        codon = 0 if change > 0 else 1
        self.previous_codon = codon
        return [codon]

    def is_binary(self) -> bool:
        """Check if this mapper produces binary output.

        Returns:
            True (this is a binary method)
        """
        return True

    def reset(self) -> None:
        """Reset the mapper state."""
        self.previous_codon = 0
