"""Sort by Concentration output mapping method.

Multi-codon method: Sorts P-proteins by concentration (descending).
Converts 32-bit protein signatures to integer codons.
"""

from grntage.grn.protein import Protein
from grntage.mapping.base import OutputMapper, OutputMethod


class SortByConcentrationMapper(OutputMapper):
    """Maps P-proteins to codons sorted by concentration.

    Sorts all P-proteins by concentration in descending order.
    Returns their 32-bit signatures as integer codons.

    Attributes:
        descending: Sort order (True for highest first)
    """

    def __init__(self, descending: bool = True) -> None:
        """Initialize the mapper.

        Args:
            descending: If True, highest concentration first
        """
        self.descending = descending

    @property
    def method(self) -> OutputMethod:
        """Get the output method type."""
        return OutputMethod.SORT_BY_CONCENTRATION

    def map_to_codons(
        self,
        proteins: list[Protein],
        previous_concentrations: dict[int, float] | None = None,
    ) -> list[int]:
        """Map P-proteins to codons sorted by concentration.

        Args:
            proteins: List of P-proteins
            previous_concentrations: Not used for this method

        Returns:
            List of integer codons (protein signatures) sorted by concentration
        """
        if not proteins:
            return []

        # Sort proteins by concentration
        sorted_proteins = sorted(
            proteins,
            key=lambda p: p.concentration,
            reverse=self.descending,
        )

        # Convert signatures to codons
        # Signatures are 32-bit integers
        return [p.signature for p in sorted_proteins]

    def is_binary(self) -> bool:
        """Check if this mapper produces binary output.

        Returns:
            False (this is a multi-codon method)
        """
        return False
