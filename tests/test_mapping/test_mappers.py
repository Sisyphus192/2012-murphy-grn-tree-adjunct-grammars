"""Tests for output mapping methods."""

import pytest

from grntage.grn.protein import Protein, ProteinType
from grntage.mapping import (
    ConcentrationTendencyMapper,
    ConcentrationValueMapper,
    OutputMethod,
    SortByConcentrationMapper,
    SortByTendencyMapper,
)


@pytest.fixture
def sample_proteins() -> list[Protein]:
    """Create sample P-proteins for testing."""
    proteins = []
    for i, (sig, conc) in enumerate(
        [
            (0x12345678, 0.3),
            (0x87654321, 0.7),
            (0xABCDEF00, 0.5),
        ]
    ):
        p = Protein(signature=sig, protein_type=ProteinType.P)
        p.concentration = conc
        proteins.append(p)
    return proteins


class TestConcentrationValueMapper:
    """Tests for ConcentrationValueMapper."""

    def test_method_type(self) -> None:
        """Test method type is correct."""
        mapper = ConcentrationValueMapper()
        assert mapper.method == OutputMethod.CONCENTRATION_VALUE

    def test_is_binary(self) -> None:
        """Test is_binary returns True."""
        mapper = ConcentrationValueMapper()
        assert mapper.is_binary() is True

    def test_above_threshold(self, sample_proteins: list[Protein]) -> None:
        """Test codon 1 when concentration > threshold."""
        mapper = ConcentrationValueMapper(threshold=0.5, protein_index=1)
        codons = mapper.map_to_codons(sample_proteins)
        assert codons == [1]  # 0.7 > 0.5

    def test_below_threshold(self, sample_proteins: list[Protein]) -> None:
        """Test codon 0 when concentration <= threshold."""
        mapper = ConcentrationValueMapper(threshold=0.5, protein_index=0)
        codons = mapper.map_to_codons(sample_proteins)
        assert codons == [0]  # 0.3 <= 0.5

    def test_empty_proteins(self) -> None:
        """Test empty protein list returns [0]."""
        mapper = ConcentrationValueMapper()
        codons = mapper.map_to_codons([])
        assert codons == [0]


class TestConcentrationTendencyMapper:
    """Tests for ConcentrationTendencyMapper."""

    def test_method_type(self) -> None:
        """Test method type is correct."""
        mapper = ConcentrationTendencyMapper()
        assert mapper.method == OutputMethod.CONCENTRATION_TENDENCY

    def test_is_binary(self) -> None:
        """Test is_binary returns True."""
        mapper = ConcentrationTendencyMapper()
        assert mapper.is_binary() is True

    def test_increasing_concentration(self, sample_proteins: list[Protein]) -> None:
        """Paper sec. 3.1: a positive (increasing) change -> codon 0."""
        mapper = ConcentrationTendencyMapper(protein_index=0)
        prev = {sample_proteins[0].signature: 0.1}
        codons = mapper.map_to_codons(sample_proteins, prev)
        assert codons == [0]  # 0.3 > 0.1, increasing -> codon 0

    def test_decreasing_concentration(self, sample_proteins: list[Protein]) -> None:
        """Paper sec. 3.1: a negative (decreasing) change -> codon 1."""
        mapper = ConcentrationTendencyMapper(protein_index=0)
        prev = {sample_proteins[0].signature: 0.5}
        codons = mapper.map_to_codons(sample_proteins, prev)
        assert codons == [1]  # 0.3 < 0.5, decreasing -> codon 1

    def test_no_previous_returns_previous_codon(
        self, sample_proteins: list[Protein]
    ) -> None:
        """Test returns previous codon when no history."""
        mapper = ConcentrationTendencyMapper()
        mapper.previous_codon = 1
        codons = mapper.map_to_codons(sample_proteins, None)
        assert codons == [1]

    def test_reset(self) -> None:
        """Test reset clears state."""
        mapper = ConcentrationTendencyMapper()
        mapper.previous_codon = 1
        mapper.reset()
        assert mapper.previous_codon == 0


class TestSortByConcentrationMapper:
    """Tests for SortByConcentrationMapper."""

    def test_method_type(self) -> None:
        """Test method type is correct."""
        mapper = SortByConcentrationMapper()
        assert mapper.method == OutputMethod.SORT_BY_CONCENTRATION

    def test_is_not_binary(self) -> None:
        """Test is_binary returns False."""
        mapper = SortByConcentrationMapper()
        assert mapper.is_binary() is False

    def test_sort_descending(self, sample_proteins: list[Protein]) -> None:
        """Test proteins sorted by concentration descending."""
        mapper = SortByConcentrationMapper(descending=True)
        codons = mapper.map_to_codons(sample_proteins)
        # Order should be: 0.7, 0.5, 0.3
        assert codons == [0x87654321, 0xABCDEF00, 0x12345678]

    def test_sort_ascending(self, sample_proteins: list[Protein]) -> None:
        """Test proteins sorted by concentration ascending."""
        mapper = SortByConcentrationMapper(descending=False)
        codons = mapper.map_to_codons(sample_proteins)
        # Order should be: 0.3, 0.5, 0.7
        assert codons == [0x12345678, 0xABCDEF00, 0x87654321]

    def test_empty_proteins(self) -> None:
        """Test empty protein list returns empty list."""
        mapper = SortByConcentrationMapper()
        codons = mapper.map_to_codons([])
        assert codons == []


class TestSortByTendencyMapper:
    """Tests for SortByTendencyMapper."""

    def test_method_type(self) -> None:
        """Test method type is correct."""
        mapper = SortByTendencyMapper()
        assert mapper.method == OutputMethod.SORT_BY_TENDENCY

    def test_is_not_binary(self) -> None:
        """Test is_binary returns False."""
        mapper = SortByTendencyMapper()
        assert mapper.is_binary() is False

    def test_sort_by_change(self, sample_proteins: list[Protein]) -> None:
        """Test proteins sorted by concentration change."""
        mapper = SortByTendencyMapper()
        # Previous: 0.5, 0.5, 0.5 -> Changes: -0.2, +0.2, 0.0
        prev = {p.signature: 0.5 for p in sample_proteins}
        codons = mapper.map_to_codons(sample_proteins, prev)
        # Largest positive first: +0.2, 0.0, -0.2
        assert codons == [0x87654321, 0xABCDEF00, 0x12345678]

    def test_sort_by_absolute_magnitude(self, sample_proteins: list[Protein]) -> None:
        """by_absolute=True sorts by |change| (thesis Ch.9), not signed (paper).

        Source contradiction: the paper's worked example sorts by SIGNED change (most
        negative LAST); the thesis says ABSOLUTE magnitude (largest |change| first,
        smallest last). Under absolute the zero-change protein must be LAST, whereas
        under signed it sits in the middle.
        """
        prev = {p.signature: 0.5 for p in sample_proteins}
        # changes: 0x12345678 -0.2, 0x87654321 +0.2, 0xABCDEF00 0.0
        absolute = SortByTendencyMapper(by_absolute=True).map_to_codons(
            sample_proteins, prev
        )
        signed = SortByTendencyMapper().map_to_codons(sample_proteins, prev)
        assert absolute[-1] == 0xABCDEF00  # zero-change protein last under |.|
        assert set(absolute[:2]) == {0x12345678, 0x87654321}  # both |0.2| first
        assert signed == [0x87654321, 0xABCDEF00, 0x12345678]  # 0.0 in the middle
        assert absolute != signed

    def test_empty_proteins(self) -> None:
        """Test empty protein list returns empty list."""
        mapper = SortByTendencyMapper()
        codons = mapper.map_to_codons([])
        assert codons == []

    def test_reset(self) -> None:
        """Test reset clears state."""
        mapper = SortByTendencyMapper()
        mapper._previous_concentrations = {0x12345678: 0.5}
        mapper.reset()
        assert mapper._previous_concentrations == {}
