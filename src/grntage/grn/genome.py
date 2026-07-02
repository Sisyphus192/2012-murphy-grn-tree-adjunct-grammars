"""Genome representation for the GRN model.

The genome is a bit string that is scanned for promoter patterns
to extract genes. Each gene produces a protein.
"""

import random

from grntage.grn.constants import (
    ENHANCER_BITS,
    GENE_INFO_BITS,
    GENE_TOTAL_BITS,
    INHIBITOR_BITS,
    PROMOTER_BITS,
)
from grntage.grn.gene import Gene, classify_promoter


class Genome:
    """A genome containing genes for the GRN model.

    The genome is stored as a bit string and scanned for promoter
    patterns to extract genes.

    Attributes:
        bits: The raw bit string as an integer
        length: Number of bits in the genome
        genes: List of extracted genes
    """

    __slots__ = ("bits", "length", "_genes", "_genes_extracted")

    def __init__(self, bits: int, length: int) -> None:
        """Initialize a genome.

        Args:
            bits: Bit string as an integer
            length: Number of bits in the genome
        """
        self.bits = bits
        self.length = length
        self._genes: list[Gene] = []
        self._genes_extracted = False

    @classmethod
    def random(cls, length: int = 4096, rng: random.Random | None = None) -> "Genome":
        """Create a random genome.

        Args:
            length: Number of bits (default 4096 = 128 codons * 32 bits)
            rng: Random number generator (uses global random module if None)

        Returns:
            Randomly initialized genome
        """
        if rng is None:
            # Use global random module (respects random.seed() calls)
            bits = random.getrandbits(length)
        else:
            bits = rng.getrandbits(length)
        return cls(bits, length)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Genome":
        """Create a genome from bytes.

        Args:
            data: Byte string

        Returns:
            Genome instance
        """
        bits = int.from_bytes(data, byteorder="big")
        return cls(bits, len(data) * 8)

    def to_bytes(self) -> bytes:
        """Convert genome to bytes.

        Returns:
            Byte representation
        """
        byte_length = (self.length + 7) // 8
        return self.bits.to_bytes(byte_length, byteorder="big")

    def copy(self) -> "Genome":
        """Create a copy of this genome.

        Returns:
            New genome with same bits
        """
        new_genome = Genome(self.bits, self.length)
        return new_genome

    @property
    def genes(self) -> list[Gene]:
        """Get extracted genes, scanning if not already done."""
        if not self._genes_extracted:
            self._extract_genes()
        return self._genes

    def _read_bits(self, start: int, n: int) -> int:
        """Read ``n`` bits LSB-first starting at bit ``start`` (circular, knob D4)."""
        start %= self.length
        if start + n <= self.length:
            return (self.bits >> start) & ((1 << n) - 1)
        # Wrap across the genome boundary.
        low_n = self.length - start
        low = (self.bits >> start) & ((1 << low_n) - 1)
        high = self.bits & ((1 << (n - low_n)) - 1)
        return low | (high << low_n)

    def _extract_genes(self) -> None:
        """Scan the genome for promoter type signatures and extract genes.

        The genome is treated as circular (knob D4) and overlapping genes are
        allowed (knob D3): every bit position whose promoter window carries a
        recognized type signature (see ``classify_promoter``) starts a gene. The
        gene layout around a promoter found at ``pos`` is, from low to high bits,
        ``[gene_info 160][promoter 32][inhibitor 32][enhancer 32]`` -- the same
        order ``Gene.from_bits`` expects.
        """
        self._genes = []
        if self.length < GENE_TOTAL_BITS:
            self._genes_extracted = True
            return

        gene_index = 0
        for pos in range(self.length):
            promoter = self._read_bits(pos, PROMOTER_BITS)
            if classify_promoter(promoter) is None:
                continue
            gene_info = self._read_bits(pos - GENE_INFO_BITS, GENE_INFO_BITS)
            inhibitor = self._read_bits(pos + PROMOTER_BITS, INHIBITOR_BITS)
            enhancer = self._read_bits(
                pos + PROMOTER_BITS + INHIBITOR_BITS, ENHANCER_BITS
            )
            self._genes.append(
                Gene(enhancer, inhibitor, promoter, gene_info, gene_index)
            )
            gene_index += 1

        self._genes_extracted = True
