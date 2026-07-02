"""Tests for generalisation testing."""

import random

from grntage.evolution.individual import Individual
from grntage.grn.genome import Genome
from grntage.experiments.generalisation import (
    POSITION_LEVELS,
    GeneralisationCase,
    GeneralisationResult,
    GeneralisationTester,
    _interpolate,
)
from grntage.grammar.definitions import DIRECT_MAPPING_GRAMMAR
from grntage.mapping.concentration import ConcentrationValueMapper
from grntage.simulation.cartpole import CartPoleState


class TestInterpolate:
    """Tests for interpolation function."""

    def test_min_value(self) -> None:
        """Test interpolation at 0.0 level."""
        assert _interpolate(0.0, -2.4, 2.4) == -2.4

    def test_max_value(self) -> None:
        """Test interpolation at 1.0 level."""
        assert _interpolate(1.0, -2.4, 2.4) == 2.4

    def test_mid_value(self) -> None:
        """Test interpolation at 0.5 level."""
        assert _interpolate(0.5, -2.4, 2.4) == 0.0


class TestPositionLevels:
    """Tests for position level constants."""

    def test_levels_count(self) -> None:
        """Test there are 5 levels."""
        assert len(POSITION_LEVELS) == 5

    def test_levels_values(self) -> None:
        """Test specific level values."""
        assert POSITION_LEVELS[0] == 0.05
        assert POSITION_LEVELS[2] == 0.5
        assert POSITION_LEVELS[4] == 0.95


class TestGeneralisationCase:
    """Tests for GeneralisationCase class."""

    def test_create_test_case(self) -> None:
        """Test creating a test case."""
        state = CartPoleState()
        case = GeneralisationCase(case_id=0, state=state)
        assert case.case_id == 0
        assert case.state is state


class TestGeneralisationResult:
    """Tests for GeneralisationResult class."""

    def test_default_values(self) -> None:
        """Test default result values."""
        ind = Individual.random()
        result = GeneralisationResult(individual=ind)
        assert result.total_cases == 625
        assert result.successful_cases == 0
        assert result.solvable_cases == 457

    def test_success_rate_empty(self) -> None:
        """Test success rate with no steps."""
        ind = Individual.random()
        result = GeneralisationResult(individual=ind)
        assert result.success_rate == 0.0

    def test_success_rate_computed(self) -> None:
        """Test success rate computation."""
        ind = Individual.random()
        result = GeneralisationResult(individual=ind)
        result.successful_cases = 100
        result.steps_per_case = [100] * 625
        assert result.success_rate == 100 / 625

    def test_mean_steps(self) -> None:
        """Test mean steps computation."""
        ind = Individual.random()
        result = GeneralisationResult(individual=ind)
        result.steps_per_case = [100, 200, 300]
        assert result.mean_steps == 200.0


class TestGeneralisationTester:
    """Tests for GeneralisationTester class."""

    def test_create_tester(self) -> None:
        """Test creating a tester."""
        tester = GeneralisationTester(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        assert tester.max_time_steps == 1000

    def test_generate_625_cases(self) -> None:
        """Test that 625 cases are generated."""
        tester = GeneralisationTester(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        cases = tester.test_cases
        assert len(cases) == 625

    def test_case_ids_unique(self) -> None:
        """Test that case IDs are unique."""
        tester = GeneralisationTester(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        case_ids = [c.case_id for c in tester.test_cases]
        assert len(set(case_ids)) == 625

    def test_cases_cached(self) -> None:
        """Test that test cases are cached."""
        tester = GeneralisationTester(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        cases1 = tester.test_cases
        cases2 = tester.test_cases
        assert cases1 is cases2

    def test_states_within_bounds(self) -> None:
        """Test that all states are within bounds."""
        tester = GeneralisationTester(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        for case in tester.test_cases:
            assert -2.4 <= case.state.x <= 2.4
            assert -1.0 <= case.state.x_dot <= 1.0
            assert -0.21 <= case.state.theta <= 0.21
            assert -1.5 <= case.state.theta_dot <= 1.5

    def test_parallel_matches_sequential(self) -> None:
        """Parallel generalisation gives bit-identical results to sequential.

        Each of the 625 cases is deterministic given (genome, initial state) and
        results are collected in case order, so the parallel path must reproduce
        the sequential steps_per_case and success count exactly. Tiny sim params
        keep the full 625-case sweep fast.
        """
        kwargs = dict(
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
            max_time_steps=3,
            grn_iterations_per_step=3,
            initial_grn_iterations=3,
        )
        individual = Individual(Genome.random(512, random.Random(0)))
        seq = GeneralisationTester(parallel=False, **kwargs).test_individual(individual)
        par = GeneralisationTester(parallel=True, **kwargs).test_individual(individual)
        assert par.steps_per_case == seq.steps_per_case
        assert par.successful_cases == seq.successful_cases
