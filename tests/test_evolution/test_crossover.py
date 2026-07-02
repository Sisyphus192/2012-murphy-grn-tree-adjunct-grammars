"""Tests for one-point crossover operator."""

import random

from grntage.evolution.crossover import OnePointCrossover
from grntage.evolution.individual import Individual
from grntage.grn.genome import Genome


def _make_individual(bits: int, length: int, fitness: float = 1.0) -> Individual:
    """Helper: create an Individual with a hand-crafted genome."""
    genome = Genome(bits, length)
    ind = Individual(genome=genome, fitness=fitness)
    return ind


class TestOnePointCrossover:
    """Tests for OnePointCrossover class."""

    def test_default_crossover_rate(self) -> None:
        """Test default crossover rate."""
        op = OnePointCrossover()
        assert op.crossover_rate == 0.9

    def test_custom_crossover_rate(self) -> None:
        """Test custom crossover rate is stored."""
        op = OnePointCrossover(crossover_rate=0.5)
        assert op.crossover_rate == 0.5

    # ------------------------------------------------------------------
    # (a) Determinism — same global seed + same parents => identical bits
    # ------------------------------------------------------------------

    def test_determinism(self) -> None:
        """Same global seed and same parents produce identical children."""
        length = 64
        p1 = _make_individual(0xAAAA_AAAA_AAAA_AAAA & ((1 << length) - 1), length)
        p2 = _make_individual(0x5555_5555_5555_5555 & ((1 << length) - 1), length)

        op = OnePointCrossover(crossover_rate=1.0)

        random.seed(7)
        c1a, c2a = op.cross(p1, p2)

        random.seed(7)
        c1b, c2b = op.cross(p1, p2)

        assert c1a.genome.bits == c1b.genome.bits
        assert c2a.genome.bits == c2b.genome.bits

    # ------------------------------------------------------------------
    # (b) At crossover_rate=1.0, children differ from parents (or equal the
    #     documented mask result) and genome length is preserved for both.
    # ------------------------------------------------------------------

    def test_crossover_rate_one_produces_recombination(self) -> None:
        """At rate=1.0, children are recombined and keep the correct length."""
        length = 64
        full_mask = (1 << length) - 1
        # All-ones parent and all-zeros parent so any split is visible.
        p1 = _make_individual(full_mask, length)
        p2 = _make_individual(0, length)

        op = OnePointCrossover(crossover_rate=1.0)
        random.seed(42)
        c1, c2 = op.cross(p1, p2)

        # Length must be preserved
        assert c1.genome.length == length
        assert c2.genome.length == length

        # With p1=all-ones, p2=all-zeros and a valid split point in [1, 63]:
        #   c1 = (all-ones & low_mask) | (all-zeros & high_mask) = low_mask
        #   c2 = (all-zeros & low_mask) | (all-ones & high_mask) = high_mask
        # So c1 | c2 must equal full_mask, and both must differ from both parents.
        assert c1.genome.bits != p1.genome.bits
        assert c1.genome.bits != p2.genome.bits
        assert c2.genome.bits != p1.genome.bits
        assert c2.genome.bits != p2.genome.bits
        assert (c1.genome.bits | c2.genome.bits) == full_mask
        assert (c1.genome.bits & c2.genome.bits) == 0

    # ------------------------------------------------------------------
    # (c) At crossover_rate=0.0, cross returns pure copies of the parents.
    # ------------------------------------------------------------------

    def test_crossover_rate_zero_returns_copies(self) -> None:
        """At rate=0.0, children have the same bits as their respective parents."""
        length = 64
        p1 = _make_individual(0xDEAD_BEEF_DEAD_BEEF & ((1 << length) - 1), length)
        p2 = _make_individual(0xCAFE_BABE_CAFE_BABE & ((1 << length) - 1), length)

        op = OnePointCrossover(crossover_rate=0.0)
        random.seed(0)
        c1, c2 = op.cross(p1, p2)

        assert c1.genome.bits == p1.genome.bits
        assert c2.genome.bits == p2.genome.bits

    # ------------------------------------------------------------------
    # (d) Both children have fitness == inf after recombination.
    # ------------------------------------------------------------------

    def test_fitness_reset_after_recombination(self) -> None:
        """Children have fitness=inf after crossover (both recombined and copied)."""
        length = 64
        p1 = _make_individual(
            0xFFFF_FFFF_FFFF_FFFF & ((1 << length) - 1), length, fitness=0.5
        )
        p2 = _make_individual(0x0000_0000_0000_0000, length, fitness=0.3)

        op = OnePointCrossover(crossover_rate=1.0)
        random.seed(1)
        c1, c2 = op.cross(p1, p2)

        assert c1.fitness == float("inf")
        assert c2.fitness == float("inf")

    def test_no_crossover_copies_parent_fitness(self) -> None:
        """When no crossover occurs (rate=0.0), children are pure copies including fitness.

        Fitness invalidation is the responsibility of the subsequent mutation step;
        crossover only resets fitness when it actually recombines bits.
        """
        length = 64
        p1 = _make_individual(0xAAAA & ((1 << length) - 1), length, fitness=2.0)
        p2 = _make_individual(0x5555 & ((1 << length) - 1), length, fitness=3.0)

        op = OnePointCrossover(crossover_rate=0.0)
        c1, c2 = op.cross(p1, p2)

        # No recombination: children are pure copies retaining parent fitness
        assert c1.fitness == 2.0
        assert c2.fitness == 3.0

    def test_grn_invalidated_after_crossover(self) -> None:
        """GRN cache is invalidated on children after recombination."""
        length = 64
        p1 = _make_individual(0xFFFF_FFFF_FFFF_FFFF & ((1 << length) - 1), length)
        p2 = _make_individual(0, length)
        # Prime GRN caches on parents (children are copies, so cache would transfer)
        _ = p1.grn
        _ = p2.grn

        op = OnePointCrossover(crossover_rate=1.0)
        random.seed(99)
        c1, c2 = op.cross(p1, p2)

        assert c1._grn is None
        assert c2._grn is None

    def test_parents_not_modified(self) -> None:
        """Cross does not mutate the parent individuals."""
        length = 64
        bits1 = 0xAAAA_AAAA_AAAA_AAAA & ((1 << length) - 1)
        bits2 = 0x5555_5555_5555_5555 & ((1 << length) - 1)
        p1 = _make_individual(bits1, length, fitness=1.5)
        p2 = _make_individual(bits2, length, fitness=2.5)

        op = OnePointCrossover(crossover_rate=1.0)
        random.seed(3)
        op.cross(p1, p2)

        assert p1.genome.bits == bits1
        assert p1.fitness == 1.5
        assert p2.genome.bits == bits2
        assert p2.fitness == 2.5

    def test_mismatched_lengths_raise(self) -> None:
        """Crossing parents with different genome lengths raises AssertionError."""
        p1 = _make_individual(0xFF, 8)
        p2 = _make_individual(0xFFFF, 16)
        op = OnePointCrossover()
        try:
            op.cross(p1, p2)
            assert False, "Expected AssertionError"
        except AssertionError:
            pass
