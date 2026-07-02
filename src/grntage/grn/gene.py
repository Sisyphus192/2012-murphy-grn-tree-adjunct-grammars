"""Gene representation for the GRN model.

A gene consists of:
- Enhancer site (32 bits): Positive regulation binding site
- Inhibitor site (32 bits): Negative regulation binding site
- Promoter site (32 bits): Identifies gene type (TF or P)
- Gene information (160 bits): Encodes the protein signature
"""

from grntage.grn.constants import (
    ENHANCER_BITS,
    GENE_INFO_BITS,
    INHIBITOR_BITS,
    P_GENE_SIGNATURE,
    PROMOTER_BITS,
    TF_GENE_SIGNATURE,
    TYPE_SIGNATURE_MASK,
    TYPE_SIGNATURE_SHIFT,
)
from grntage.grn.protein import Protein, ProteinType


def classify_promoter(promoter: int) -> ProteinType | None:
    """Classify a 32-bit promoter by its 8-bit type signature.

    Returns ``ProteinType.TF`` for a TF-gene signature (``TF_GENE_SIGNATURE``),
    ``ProteinType.P`` for a P-gene signature (``P_GENE_SIGNATURE``), or ``None``
    when the promoter carries no recognized type signature (i.e. it does not
    start a gene). Which 8 bits are examined is set by ``TYPE_SIGNATURE_SHIFT``
    (the D8 calibration knob).
    """
    type_bits = (promoter >> TYPE_SIGNATURE_SHIFT) & TYPE_SIGNATURE_MASK
    if type_bits == TF_GENE_SIGNATURE:
        return ProteinType.TF
    if type_bits == P_GENE_SIGNATURE:
        return ProteinType.P
    return None


class Gene:
    """A gene in the Gene Regulatory Network.

    Attributes:
        enhancer: 32-bit enhancer site signature
        inhibitor: 32-bit inhibitor site signature
        promoter: 32-bit promoter site (determines gene type)
        gene_info: 160-bit gene information region
        protein_signature: Computed 32-bit protein signature
        gene_type: TF or P based on promoter signature
    """

    __slots__ = (
        "enhancer",
        "inhibitor",
        "promoter",
        "gene_info",
        "protein_signature",
        "gene_type",
        "index",
    )

    def __init__(
        self,
        enhancer: int,
        inhibitor: int,
        promoter: int,
        gene_info: int,
        index: int = -1,
    ) -> None:
        """Initialize a gene from its component parts.

        Args:
            enhancer: 32-bit enhancer site
            inhibitor: 32-bit inhibitor site
            promoter: 32-bit promoter site
            gene_info: 160-bit gene information region
            index: Position index in genome
        """
        self.enhancer = enhancer
        self.inhibitor = inhibitor
        self.promoter = promoter
        self.gene_info = gene_info
        self.index = index

        # Determine gene type from the promoter's 8-bit type signature (paper
        # sec. 2.2). Genes extracted by the genome scanner always carry a
        # recognized signature; for a directly-constructed gene whose promoter is
        # unrecognized we default to TF so a gene always has a defined type.
        self.gene_type = classify_promoter(promoter) or ProteinType.TF

        # Compute protein signature using majority vote
        self.protein_signature = self._compute_protein_signature()

    def _compute_protein_signature(self) -> int:
        """Compute protein signature using majority vote across 5 sections.

        The 160-bit gene_info is split into 5 x 32-bit sections.
        For each bit position, the majority value across sections is used.

        Returns:
            32-bit protein signature
        """
        # Extract 5 x 32-bit sections from gene_info
        sections = []
        for i in range(5):
            shift = (4 - i) * 32  # Start from most significant
            section = (self.gene_info >> shift) & 0xFFFFFFFF
            sections.append(section)

        # Majority vote for each bit position
        result = 0
        for bit_pos in range(32):
            count = sum((section >> bit_pos) & 1 for section in sections)
            if count >= 3:  # Majority (3 or more out of 5)
                result |= 1 << bit_pos

        return result

    def create_protein(self, initial_concentration: float = 0.0) -> Protein:
        """Create a protein from this gene.

        Args:
            initial_concentration: Starting concentration level

        Returns:
            Protein with this gene's computed signature
        """
        return Protein(
            signature=self.protein_signature,
            protein_type=self.gene_type,
            concentration=initial_concentration,
            gene_index=self.index,
        )

    @classmethod
    def from_bits(cls, bits: int, index: int = -1) -> "Gene":
        """Create a gene from a 256-bit integer.

        Args:
            bits: 256-bit integer containing all gene data
            index: Position index in genome

        Returns:
            Gene instance
        """
        # Extract components (from MSB to LSB)
        gene_info = bits & ((1 << GENE_INFO_BITS) - 1)
        bits >>= GENE_INFO_BITS

        promoter = bits & ((1 << PROMOTER_BITS) - 1)
        bits >>= PROMOTER_BITS

        inhibitor = bits & ((1 << INHIBITOR_BITS) - 1)
        bits >>= INHIBITOR_BITS

        enhancer = bits & ((1 << ENHANCER_BITS) - 1)

        return cls(enhancer, inhibitor, promoter, gene_info, index)

    def __repr__(self) -> str:
        return (
            f"Gene(type={self.gene_type.name}, "
            f"protein_sig=0x{self.protein_signature:08X}, "
            f"idx={self.index})"
        )
