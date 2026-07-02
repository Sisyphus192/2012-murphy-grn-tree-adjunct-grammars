"""Tests for the Genome class."""

import random


from grntage.grn.genome import Genome
from grntage.grn.protein import ProteinType


class TestGenome:
    """Tests for Genome class."""

    def test_random_genome_length(self) -> None:
        """Test that random genome has correct length."""
        genome = Genome.random(4096)
        assert genome.length == 4096

    def test_random_genome_reproducible(self) -> None:
        """Test that seeded random genomes are reproducible."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        genome1 = Genome.random(1024, rng1)
        genome2 = Genome.random(1024, rng2)

        assert genome1.bits == genome2.bits

    def test_from_bytes_roundtrip(self) -> None:
        """Test bytes conversion roundtrip."""
        original = Genome.random(256)
        data = original.to_bytes()
        restored = Genome.from_bytes(data)

        assert restored.bits == original.bits
        assert restored.length == original.length

    def test_genes_property_caches(self) -> None:
        """Test that genes property caches extraction."""
        genome = Genome.random(4096, random.Random(42))

        genes1 = genome.genes
        genes2 = genome.genes

        assert genes1 is genes2  # Same object

    def test_genome_without_signatures_has_no_genes(self) -> None:
        """An alternating-bit genome has no 0x00/0xFF type byte -> no genes."""
        genome = Genome(int("AA" * 32, 16), 256)  # never 8 equal bits in a row
        assert genome.genes == []

    def test_all_zero_genome_is_all_tf_genes(self) -> None:
        """All-zero genome: every position is a 0x00 (TF) promoter.

        A faithful consequence of identifying genes by an all-zero type
        signature with circular (D4), overlap-allowed (D3) scanning.
        """
        genes = Genome(0, 256).genes
        assert len(genes) == 256
        assert all(g.gene_type == ProteinType.TF for g in genes)

    def test_hand_built_tf_gene_extraction(self) -> None:
        """A planted TF promoter (final byte 0x00) extracts one gene, exact fields.

        Fillers use alternating 0x55/0xAA bytes so the only 8-equal-bit run in the
        genome is the planted type signature -> exactly one gene.
        """
        gene_info = int("AA" * 20, 16)  # 160 alternating bits (top bit 1)
        promoter = 0x55555500  # final byte 0x00 -> TF; isolated 8-zero run
        inhibitor = 0x55555555
        enhancer = 0xAAAAAAAA
        bits = gene_info | (promoter << 160) | (inhibitor << 192) | (enhancer << 224)

        genes = Genome(bits, 256).genes
        assert len(genes) == 1
        gene = genes[0]
        assert gene.gene_type == ProteinType.TF
        assert gene.promoter == promoter
        assert gene.gene_info == gene_info
        assert gene.inhibitor == inhibitor
        assert gene.enhancer == enhancer

    def test_hand_built_p_gene_extraction(self) -> None:
        """A planted P promoter (final byte 0xFF) extracts one P gene."""
        gene_info = int("55" * 20, 16)  # 160 alternating bits (top bit 0)
        promoter = 0xAAAAAAFF  # final byte 0xFF -> P; isolated 8-one run
        inhibitor = 0xAAAAAAAA
        enhancer = 0x55555555
        bits = gene_info | (promoter << 160) | (inhibitor << 192) | (enhancer << 224)

        genes = Genome(bits, 256).genes
        assert len(genes) == 1
        gene = genes[0]
        assert gene.gene_type == ProteinType.P
        assert gene.promoter == promoter
        assert gene.gene_info == gene_info
        assert gene.inhibitor == inhibitor
        assert gene.enhancer == enhancer

    def test_large_genome_extraction(self) -> None:
        """Gene extraction from a fixed-seed random genome is pinned exactly.

        Pins the scan/overlap/circular behavior so a regression in the scan range
        or overlap policy changes the count and is caught.
        """
        genome = Genome.random(4096, random.Random(12345))
        genes = genome.genes

        assert len(genes) == 22  # seed 12345, low-byte signature, circular+overlap
        assert sum(g.gene_type == ProteinType.TF for g in genes) == 15
        assert sum(g.gene_type == ProteinType.P for g in genes) == 7
        assert [g.index for g in genes] == list(range(22))  # contiguous indices
