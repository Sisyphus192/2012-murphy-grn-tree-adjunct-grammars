"""Tests for the Protein class."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grntage.grn.protein import Protein, ProteinType

UINT32 = st.integers(min_value=0, max_value=0xFFFFFFFF)


class TestProtein:
    """Tests for Protein class."""

    def test_create_tf_protein(self) -> None:
        """Test creating a TF protein."""
        protein = Protein(0x12345678, ProteinType.TF, 0.5)
        assert protein.signature == 0x12345678
        assert protein.protein_type == ProteinType.TF
        assert protein.concentration == 0.5

    def test_create_p_protein(self) -> None:
        """Test creating a P protein."""
        protein = Protein(0xABCDEF00, ProteinType.P, 0.25)
        assert protein.signature == 0xABCDEF00
        assert protein.protein_type == ProteinType.P
        assert protein.concentration == 0.25

    def test_invalid_signature_raises(self) -> None:
        """Test that invalid signatures raise ValueError."""
        with pytest.raises(ValueError, match="32-bit"):
            Protein(0x1_0000_0000, ProteinType.TF)  # 33 bits

    def test_invalid_concentration_raises(self) -> None:
        """Test that invalid concentrations raise ValueError."""
        with pytest.raises(ValueError, match="Concentration"):
            Protein(0x0, ProteinType.TF, 1.5)
        with pytest.raises(ValueError, match="Concentration"):
            Protein(0x0, ProteinType.TF, -0.1)

    def test_complementary_bits_identical(self) -> None:
        """Identical signatures have ZERO complementary (differing) bits (paper Eq. 1)."""
        protein = Protein(0xFFFFFFFF, ProteinType.TF)
        assert protein.count_complementary_bits(0xFFFFFFFF) == 0

    def test_complementary_bits_opposite(self) -> None:
        """Fully opposite signatures have 32 complementary (differing) bits."""
        protein = Protein(0xFFFFFFFF, ProteinType.TF)
        assert protein.count_complementary_bits(0x00000000) == 32

    def test_complementary_bits_half(self) -> None:
        """Complementary bits = number of differing bits = popcount(XOR)."""
        protein = Protein(0xFFFF0000, ProteinType.TF)
        assert protein.count_complementary_bits(0xFFFF0000) == 0  # identical
        assert protein.count_complementary_bits(0x0000FFFF) == 32  # fully opposite
        assert protein.count_complementary_bits(0xFFFFFFFF) == 16  # half differ

    @given(UINT32)
    def test_complementary_bits_self_is_zero(self, sig: int) -> None:
        """u(a, a) == 0 — identical signatures share no complementary bits.

        This property fails under the pre-fix ``32 - popcount(XOR)`` convention.
        """
        assert Protein(sig, ProteinType.TF).count_complementary_bits(sig) == 0

    @given(UINT32)
    def test_complementary_bits_complement_is_full(self, sig: int) -> None:
        """u(a, ~a) == 32 — a signature and its bitwise complement fully differ."""
        complement = sig ^ 0xFFFFFFFF
        assert Protein(sig, ProteinType.TF).count_complementary_bits(complement) == 32

    @given(UINT32, UINT32)
    def test_complementary_bits_symmetric(self, a: int, b: int) -> None:
        """u(a, b) == u(b, a)."""
        pa = Protein(a, ProteinType.TF)
        pb = Protein(b, ProteinType.TF)
        assert pa.count_complementary_bits(b) == pb.count_complementary_bits(a)

    def test_protein_equality(self) -> None:
        """Test protein equality comparison."""
        p1 = Protein(0x12345678, ProteinType.TF, 0.5)
        p2 = Protein(0x12345678, ProteinType.TF, 0.3)  # Different concentration
        p3 = Protein(0x12345678, ProteinType.P, 0.5)  # Different type
        p4 = Protein(0x87654321, ProteinType.TF, 0.5)  # Different signature

        assert p1 == p2  # Same signature and type
        assert p1 != p3  # Different type
        assert p1 != p4  # Different signature

    def test_protein_hash(self) -> None:
        """Test protein hashing for use in sets/dicts."""
        p1 = Protein(0x12345678, ProteinType.TF, 0.5)
        p2 = Protein(0x12345678, ProteinType.TF, 0.3)

        protein_set = {p1, p2}
        assert len(protein_set) == 1  # Same hash

    def test_protein_repr(self) -> None:
        """Test protein string representation."""
        protein = Protein(0x12345678, ProteinType.TF, 0.5)
        repr_str = repr(protein)
        assert "12345678" in repr_str.upper()
        assert "TF" in repr_str
        assert "0.5" in repr_str
