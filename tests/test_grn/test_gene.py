"""Tests for the Gene class."""

from grntage.grn.gene import Gene, classify_promoter
from grntage.grn.protein import ProteinType


class TestGene:
    """Tests for Gene class."""

    def test_create_tf_gene(self) -> None:
        """A promoter whose type signature (final/low 8 bits) is 0x00 is a TF gene."""
        promoter = 0x12345600  # low byte 0x00 -> TF
        gene = Gene(
            enhancer=0xAAAAAAAA,
            inhibitor=0x55555555,
            promoter=promoter,
            gene_info=0,
        )
        assert gene.gene_type == ProteinType.TF

    def test_create_p_gene(self) -> None:
        """A promoter whose type signature (final/low 8 bits) is 0xFF is a P gene."""
        promoter = 0x123456FF  # low byte 0xFF -> P
        gene = Gene(
            enhancer=0xAAAAAAAA,
            inhibitor=0x55555555,
            promoter=promoter,
            gene_info=0,
        )
        assert gene.gene_type == ProteinType.P

    def test_classify_promoter(self) -> None:
        """classify_promoter reads the low-byte type signature (D8 default)."""
        assert classify_promoter(0x12345600) == ProteinType.TF
        assert classify_promoter(0x123456FF) == ProteinType.P
        assert classify_promoter(0x12345655) is None  # not a type signature
        assert classify_promoter(0x00000000) == ProteinType.TF
        assert classify_promoter(0xFFFFFFFF) == ProteinType.P

    def test_majority_vote_all_ones(self) -> None:
        """Test majority vote when all sections have same bit set."""
        # Create gene_info where all 5 sections have bit 0 set
        section = 0x00000001
        gene_info = 0
        for i in range(5):
            gene_info |= section << (i * 32)

        gene = Gene(0, 0, 0, gene_info)
        assert gene.protein_signature & 1 == 1

    def test_majority_vote_mixed(self) -> None:
        """Test majority vote with mixed bits."""
        # 3 sections with bit 0 set, 2 without
        sections = [0x00000001, 0x00000001, 0x00000001, 0x00000000, 0x00000000]
        gene_info = 0
        for i, section in enumerate(sections):
            gene_info |= section << ((4 - i) * 32)

        gene = Gene(0, 0, 0, gene_info)
        assert gene.protein_signature & 1 == 1  # Majority wins

    def test_majority_vote_minority(self) -> None:
        """Test majority vote when minority has bit set."""
        # 2 sections with bit 0 set, 3 without
        sections = [0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000]
        gene_info = 0
        for i, section in enumerate(sections):
            gene_info |= section << ((4 - i) * 32)

        gene = Gene(0, 0, 0, gene_info)
        assert gene.protein_signature & 1 == 0  # Majority wins

    def test_create_protein(self) -> None:
        """Test creating a protein from a gene."""
        gene = Gene(0, 0, 0, 0, index=5)
        protein = gene.create_protein(0.25)

        assert protein.signature == gene.protein_signature
        assert protein.protein_type == gene.gene_type
        assert protein.concentration == 0.25
        assert protein.gene_index == 5

    def test_from_bits(self) -> None:
        """Test creating a gene from a 256-bit integer."""
        # Create a 256-bit value with known components
        enhancer = 0xAAAAAAAA
        inhibitor = 0x55555555
        promoter = 0x12345600  # TF gene
        gene_info = 0x0123456789ABCDEF0123456789ABCDEF01234567

        # Pack into 256 bits: [enhancer][inhibitor][promoter][gene_info]
        bits = enhancer
        bits = (bits << 32) | inhibitor
        bits = (bits << 32) | promoter
        bits = (bits << 160) | gene_info

        gene = Gene.from_bits(bits)

        assert gene.enhancer == enhancer
        assert gene.inhibitor == inhibitor
        assert gene.promoter == promoter
        assert gene.gene_info == gene_info

    def test_gene_repr(self) -> None:
        """Test gene string representation."""
        gene = Gene(0, 0, 0, 0, index=3)
        repr_str = repr(gene)
        assert "TF" in repr_str or "P" in repr_str
        assert "3" in repr_str
