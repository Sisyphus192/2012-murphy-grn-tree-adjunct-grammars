"""Sort by Tendency output mapping method.

Multi-codon method: Sorts P-proteins by signed magnitude of concentration change.
Largest positive change first, largest negative change last.
"""

from grntage.grn.protein import Protein
from grntage.mapping.base import OutputMapper, OutputMethod


class SortByTendencyMapper(OutputMapper):
    """Maps P-proteins to codons sorted by concentration tendency.

    Sorts all P-proteins by the magnitude of their concentration change.
    Returns their 32-bit signatures as integer codons.
    Largest positive change first, largest negative change last.

    Attributes:
        threshold: Minimum change to consider (changes below are treated as 0)
    """

    def __init__(self, threshold: float = 1e-10, by_absolute: bool = False) -> None:
        """Initialize the mapper.

        Args:
            threshold: Minimum change magnitude to consider
            by_absolute: Sort key for the tendency. The SOURCES CONTRADICT here:
                the 2012 PPSN paper says "signed magnitude" and its worked example
                puts the most-negative-change protein LAST (signed descending), but
                the Murphy thesis (Ch. 9) says "absolute magnitude of the change ...
                changes the most first ... changed the least last" (absolute
                descending). Default False = signed (paper's worked example). Set
                True to test the thesis interpretation; the two produce a different
                SymReg force in ~80% of states (scratch/probe_signed_vs_absolute.py).
        """
        self.threshold = threshold
        self.by_absolute = by_absolute
        self._previous_concentrations: dict[int, float] = {}

    @property
    def method(self) -> OutputMethod:
        """Get the output method type."""
        return OutputMethod.SORT_BY_TENDENCY

    def map_to_codons(
        self,
        proteins: list[Protein],
        previous_concentrations: dict[int, float] | None = None,
    ) -> list[int]:
        """Map P-proteins to codons sorted by concentration tendency.

        Args:
            proteins: List of P-proteins with current concentrations
            previous_concentrations: Dict mapping signature to previous concentration

        Returns:
            List of integer codons (protein signatures) sorted by tendency
        """
        if not proteins:
            return []

        # Use provided previous concentrations or internal state
        prev_conc = previous_concentrations or self._previous_concentrations

        # Calculate change for each protein
        protein_changes: list[tuple[Protein, float]] = []
        for protein in proteins:
            prev = prev_conc.get(protein.signature, protein.concentration)
            change = protein.concentration - prev

            # Apply threshold
            if abs(change) < self.threshold:
                change = 0.0

            protein_changes.append((protein, change))

        # Sort by change descending. Signed (default, paper's worked example: largest
        # positive first, most negative last) or absolute magnitude (thesis Ch. 9:
        # largest |change| first, smallest last).
        sorted_proteins = sorted(
            protein_changes,
            key=(lambda pc: abs(pc[1])) if self.by_absolute else (lambda pc: pc[1]),
            reverse=True,
        )

        # Update internal state
        self._previous_concentrations = {p.signature: p.concentration for p in proteins}

        # Return signatures as codons
        return [p.signature for p, _ in sorted_proteins]

    def is_binary(self) -> bool:
        """Check if this mapper produces binary output.

        Returns:
            False (this is a multi-codon method)
        """
        return False

    def reset(self) -> None:
        """Reset the mapper state."""
        self._previous_concentrations = {}
