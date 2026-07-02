"""Protein representation for the GRN model.

Proteins are produced by genes and have a 32-bit signature.
Two types exist:
- TF-proteins: Can bind to regulatory sites and affect gene expression
- P-proteins: Used for output extraction only, cannot bind
"""

from enum import Enum, auto


class ProteinType(Enum):
    """Type of protein in the GRN model."""

    TF = auto()  # Transcription Factor - can bind and regulate
    P = auto()  # Product - used for output only


class Protein:
    """A protein in the Gene Regulatory Network.

    Attributes:
        signature: 32-bit signature determining binding affinity
        protein_type: Whether this is a TF or P protein
        concentration: Current concentration level (0.0 to 1.0)
        gene_index: Index of the gene that produces this protein
    """

    __slots__ = ("signature", "protein_type", "concentration", "gene_index")

    def __init__(
        self,
        signature: int,
        protein_type: ProteinType,
        concentration: float = 0.0,
        gene_index: int = -1,
    ) -> None:
        """Initialize a protein.

        Args:
            signature: 32-bit signature (0 to 2^32 - 1)
            protein_type: TF or P protein type
            concentration: Initial concentration (default 0.0)
            gene_index: Index of producing gene (-1 for free/input proteins)
        """
        if not 0 <= signature <= 0xFFFFFFFF:
            raise ValueError(f"Signature must be 32-bit unsigned: {signature}")
        if not 0.0 <= concentration <= 1.0:
            raise ValueError(f"Concentration must be in [0, 1]: {concentration}")

        self.signature = signature
        self.protein_type = protein_type
        self.concentration = concentration
        self.gene_index = gene_index

    def count_complementary_bits(self, other_signature: int) -> int:
        """Count complementary bits between this protein and another signature.

        Complementary bits are the bits that DIFFER between the two signatures
        (the set bits of the XOR). Per the paper (Eq. 1), binding affinity grows
        with the number of complementary bits: ``u_j = popcount(XOR(protein, site))``.
        More complementary bits => stronger binding.

        Args:
            other_signature: 32-bit signature to compare against

        Returns:
            Number of complementary (differing) bits (0 to 32)
        """
        # XOR sets a bit wherever the two signatures differ (are complementary).
        xor_result = self.signature ^ other_signature
        return bin(xor_result).count("1")

    def __repr__(self) -> str:
        return (
            f"Protein(sig=0x{self.signature:08X}, "
            f"type={self.protein_type.name}, "
            f"conc={self.concentration:.4f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protein):
            return NotImplemented
        return (
            self.signature == other.signature
            and self.protein_type == other.protein_type
        )

    def __hash__(self) -> int:
        return hash((self.signature, self.protein_type))
