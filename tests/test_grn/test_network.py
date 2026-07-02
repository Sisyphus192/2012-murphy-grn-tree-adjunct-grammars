"""Tests for the GRN network dynamics."""

import math
import random

import pytest

from grntage.grn.constants import BETA
from grntage.grn.gene import Gene
from grntage.grn.genome import Genome
from grntage.grn.network import GRN, GRNState
from grntage.grn.protein import Protein, ProteinType


class TestGRNState:
    """Tests for GRNState class."""

    def test_max_change_identical(self) -> None:
        """Test max change between identical states."""
        state1 = GRNState([0.5, 0.5], [0.3, 0.7])
        state2 = GRNState([0.5, 0.5], [0.3, 0.7])
        assert state1.max_change(state2) == 0.0

    def test_max_change_different(self) -> None:
        """Test max change between different states."""
        state1 = GRNState([0.5, 0.5], [0.3, 0.7])
        state2 = GRNState([0.6, 0.4], [0.2, 0.8])
        assert state1.max_change(state2) == pytest.approx(0.1)


class TestGRN:
    """Tests for GRN class."""

    def test_create_grn_from_genome(self) -> None:
        """Test creating a GRN from a genome."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        assert grn.genome is genome
        assert isinstance(grn.tf_proteins, list)
        assert isinstance(grn.p_proteins, list)

    def test_tf_concentrations_sum_to_one(self) -> None:
        """Test that TF protein concentrations sum to 1.0."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        if grn.tf_proteins:
            total = sum(p.concentration for p in grn.tf_proteins)
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_p_concentrations_sum_to_one(self) -> None:
        """Test that P protein concentrations sum to 1.0."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        if grn.p_proteins:
            total = sum(p.concentration for p in grn.p_proteins)
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_inject_inputs_scales_existing(self) -> None:
        """Test that injecting inputs scales existing TF proteins."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        if not grn.tf_proteins:
            pytest.skip("No TF proteins in genome")

        # Record original concentrations (fresh produced TF proteins sum to 1.0)
        original_total = sum(p.concentration for p in grn.tf_proteins)
        assert original_total == pytest.approx(1.0, abs=1e-9)

        # Inject input with 0.2 concentration
        input_protein = Protein(0x12345678, ProteinType.TF, 0.2)
        grn.inject_inputs([input_protein])

        # Existing proteins should be scaled to 0.8 total
        new_total = sum(p.concentration for p in grn.tf_proteins)
        assert new_total == pytest.approx(0.8, abs=1e-9)

    def test_iterate_maintains_normalization(self) -> None:
        """Test that iteration maintains concentration normalization."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        grn.iterate(100)

        if grn.tf_proteins:
            total_tf = sum(p.concentration for p in grn.tf_proteins)
            total_tf += sum(p.concentration for p in grn.free_tf_proteins)
            assert total_tf == pytest.approx(1.0, abs=1e-6)

        if grn.p_proteins:
            total_p = sum(p.concentration for p in grn.p_proteins)
            assert total_p == pytest.approx(1.0, abs=1e-6)

    def test_stabilize_returns_iterations(self) -> None:
        """Test that stabilize returns iteration count."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        iterations = grn.stabilize(max_iterations=1000)
        assert 1 <= iterations <= 1000

    def test_get_p_protein_concentrations_sorted(self) -> None:
        """Test that P proteins are returned sorted by concentration."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        if len(grn.p_proteins) < 2:
            pytest.skip("Need at least 2 P proteins")

        grn.iterate(100)
        sorted_proteins = grn.get_p_protein_concentrations()

        # Verify descending order
        for i in range(len(sorted_proteins) - 1):
            assert sorted_proteins[i][1] >= sorted_proteins[i + 1][1]

    def test_run_with_input(self) -> None:
        """Test running GRN with input injection."""
        genome = Genome.random(4096, random.Random(42))
        grn = GRN(genome)

        input_protein = Protein(0xFFFFFFFF, ProteinType.TF, 0.1)
        grn.run_with_input([input_protein], iterations=100)

        # Verify input was injected
        assert len(grn.free_tf_proteins) == 1
        assert grn.free_tf_proteins[0].concentration == 0.1


class TestGRNEquations:
    """Pin the regulation equations (paper Eq. 1-3) and normalization invariants."""

    def _empty_grn(self) -> GRN:
        # An alternating-bit genome has no promoters, so the GRN starts with no
        # genes/proteins and we can install controlled proteins by hand.
        return GRN(Genome(int("AA" * 32, 16), 256))

    def test_compute_signal_eq1(self) -> None:
        """Eq. 1: e_i = (1/N) * sum_j c_j*exp(beta*(u_j - u_max)), u_j = popcount(XOR)."""
        grn = self._empty_grn()
        grn.tf_proteins = [
            Protein(0x00000000, ProteinType.TF, 0.6),
            Protein(0xFFFFFFFF, ProteinType.TF, 0.4),
        ]
        grn.free_tf_proteins = []
        grn._u_max = 32  # max complementary bits in this system

        # vs site 0x0: protein A (0x0) has 0 differing bits, protein B has 32.
        expected = 0.5 * (
            0.6 * math.exp(BETA * (0 - 32)) + 0.4 * math.exp(BETA * (32 - 32))
        )
        assert grn._compute_signal(0x00000000) == pytest.approx(expected)

    def test_eq2_has_ci_term_eq3_does_not(self) -> None:
        """Eq. 2 (TF rate) = (e-h)*c_i; Eq. 3 (P rate) = (e-h), no c_i.

        The setup forces e != h (the gene's enhancer matches the lone TF protein
        poorly, its inhibitor matches it perfectly), so the c_i factor is
        observable, and the two update rules are compared against each other
        rather than re-derived -- so the test fails if c_i is dropped from Eq. 2
        (or wrongly added to Eq. 3).
        """
        grn = self._empty_grn()
        # A single TF protein drives the signals; e-h depends only on the gene's
        # sites, so it is identical for the TF and P updates below.
        tf_protein = Protein(0x00000000, ProteinType.TF, 0.5)  # c_i = 0.5
        p_protein = Protein(0x12345678, ProteinType.P, 0.3)
        grn.tf_proteins = [tf_protein]
        grn.p_proteins = [p_protein]
        grn.free_tf_proteins = []
        grn._u_max = 32

        # enhancer 0x0 -> 0 complementary bits (weak); inhibitor 0xFFFFFFFF -> 32
        # (strong) => e != h.
        gene = Gene(0x00000000, 0xFFFFFFFF, 0x00, 0)
        tf_delta = grn._update_tf_protein(tf_protein, gene) - tf_protein.concentration
        p_delta = grn._update_p_protein(p_protein, gene) - p_protein.concentration

        assert abs(p_delta) > 1e-3  # e - h is meaningfully non-zero
        # TF rate carries the extra c_i factor; P rate does not.
        assert tf_delta == pytest.approx(p_delta * tf_protein.concentration)
        assert tf_delta != pytest.approx(p_delta)  # would be equal if c_i were dropped

    def test_pure_python_normalization_with_input(self) -> None:
        """Produced TF + free TF sum to 1.0; P sum to 1.0 (pure-Python path)."""
        grn = GRN(Genome.random(4096, random.Random(7)))
        grn.set_input_concentration(0x12345678, 0.1)
        grn.iterate_pure_python(50)

        if grn.tf_proteins:
            total_tf = sum(p.concentration for p in grn.tf_proteins) + sum(
                p.concentration for p in grn.free_tf_proteins
            )
            assert total_tf == pytest.approx(1.0, abs=1e-9)
        if grn.p_proteins:
            assert sum(p.concentration for p in grn.p_proteins) == pytest.approx(
                1.0, abs=1e-9
            )

    def test_jit_matches_pure_python(self) -> None:
        """The JIT and pure-Python iteration paths agree (both implement Eq. 1-3)."""
        genome = Genome.random(4096, random.Random(99))
        grn_jit = GRN(genome)
        grn_pp = GRN(genome)
        grn_jit.set_input_concentration(0xDEADBEEF, 0.05)
        grn_pp.set_input_concentration(0xDEADBEEF, 0.05)

        grn_jit.iterate(5)
        grn_pp.iterate_pure_python(5)

        for a, b in zip(grn_jit.tf_proteins, grn_pp.tf_proteins, strict=True):
            assert a.concentration == pytest.approx(b.concentration, abs=1e-10)
        for a, b in zip(grn_jit.p_proteins, grn_pp.p_proteins, strict=True):
            assert a.concentration == pytest.approx(b.concentration, abs=1e-10)
