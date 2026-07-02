"""Generalisation testing for evolved controllers.

Generates 625 test cases from 5 values per variable
and tests controllers on each case.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product

from grntage.evolution.individual import Individual
from grntage.grammar.definitions import Grammar
from grntage.mapping.base import OutputMapper
from grntage.simulation.cartpole import (
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
    CartPoleState,
)
from grntage.simulation.controller import evaluate_best_protein_steps

# 5 values at 0.05, 0.275, 0.5, 0.725, 0.95 of each variable's range
POSITION_LEVELS = [0.05, 0.275, 0.5, 0.725, 0.95]


def _interpolate(level: float, min_val: float, max_val: float) -> float:
    """Interpolate within a range at a given level."""
    return min_val + level * (max_val - min_val)


def _run_generalisation_case_worker(
    genome_bits: int,
    genome_length: int,
    state_tuple: tuple[float, float, float, float],
    grammar: Grammar,
    output_mapper: OutputMapper,
    max_time_steps: int,
    grn_iterations_per_step: int,
    initial_grn_iterations: int,
    physics_dt: float,
    physics_substeps: int,
) -> int:
    """Run one generalisation case in a worker process; return steps survived.

    Reconstructs the GRN from the (picklable) genome bits, mirroring
    ``GeneralisationTester.test_case`` exactly so the parallel result is identical
    to the sequential one. The output mapper is a fresh pickled copy per task and
    is reset here, so tendency state never leaks across cases.
    """
    from grntage.grn.genome import Genome
    from grntage.grn.network import GRN

    grn = GRN(Genome(genome_bits, genome_length))
    return evaluate_best_protein_steps(
        grn,
        output_mapper,
        grammar,
        CartPoleState(*state_tuple),
        physics_dt=physics_dt,
        physics_substeps=physics_substeps,
        grn_iterations_per_step=grn_iterations_per_step,
        initial_grn_iterations=initial_grn_iterations,
        max_time_steps=max_time_steps,
    )


@dataclass
class GeneralisationCase:
    """A single generalization test case."""

    case_id: int
    state: CartPoleState


@dataclass
class GeneralisationResult:
    """Result of generalisation testing.

    Attributes:
        individual: The tested individual
        total_cases: Total number of test cases (625)
        successful_cases: Number of cases solved (1000+ steps)
        solvable_cases: Number of actually solvable cases (457)
        steps_per_case: Steps survived in each case
    """

    individual: Individual
    total_cases: int = 625
    successful_cases: int = 0
    solvable_cases: int = 457  # From paper
    steps_per_case: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Generalisation success rate."""
        if not self.steps_per_case:
            return 0.0
        return self.successful_cases / self.total_cases

    @property
    def mean_steps(self) -> float:
        """Mean steps across all cases."""
        if not self.steps_per_case:
            return 0.0
        return sum(self.steps_per_case) / len(self.steps_per_case)


class GeneralisationTester:
    """Tests evolved controllers on 625 generalisation cases."""

    def __init__(
        self,
        grammar: Grammar,
        output_mapper: OutputMapper,
        max_time_steps: int = 1000,
        grn_iterations_per_step: int = 2000,
        initial_grn_iterations: int = 10000,
        physics_dt: float = 0.02,
        physics_substeps: int = 1,
        parallel: bool = True,
        num_workers: int | None = None,
    ) -> None:
        """Initialize the tester.

        Args:
            grammar: Grammar for phenotype mapping
            output_mapper: Output mapping method
            max_time_steps: Maximum steps per test (default 1000)
            grn_iterations_per_step: GRN iterations per physics step
            initial_grn_iterations: Initial GRN stabilization iterations
            physics_dt: Cart-pole control interval in seconds (A.3.4/D1, default
                0.02 = classic Barto step). Must match the training regime for
                comparability.
            physics_substeps: Euler sub-steps per control interval (default 1).
            parallel: Evaluate the 625 cases across processes (default True). The
                cases are independent and collected in order, so the result is
                identical to the sequential path.
            num_workers: Worker processes for the parallel path (default cpu_count).
        """
        self.grammar = grammar
        self.output_mapper = output_mapper
        self.max_time_steps = max_time_steps
        self.grn_iterations_per_step = grn_iterations_per_step
        self.initial_grn_iterations = initial_grn_iterations
        self.physics_dt = physics_dt
        self.physics_substeps = physics_substeps
        self.parallel = parallel
        self.num_workers = num_workers
        self._test_cases: list[GeneralisationCase] | None = None

    @property
    def test_cases(self) -> list[GeneralisationCase]:
        """Get all 625 test cases."""
        if self._test_cases is None:
            self._test_cases = self._generate_test_cases()
        return self._test_cases

    def _generate_test_cases(self) -> list[GeneralisationCase]:
        """Generate 625 test cases."""
        cases = []
        case_id = 0

        for x_level, xdot_level, theta_level, thetadot_level in product(
            POSITION_LEVELS, repeat=4
        ):
            state = CartPoleState(
                x=_interpolate(x_level, *X_RANGE),
                x_dot=_interpolate(xdot_level, *X_DOT_RANGE),
                theta=_interpolate(theta_level, *THETA_RANGE),
                theta_dot=_interpolate(thetadot_level, *THETA_DOT_RANGE),
            )
            cases.append(GeneralisationCase(case_id=case_id, state=state))
            case_id += 1

        return cases

    def test_case(self, individual: Individual, test_case: GeneralisationCase) -> int:
        """Test individual on a single case.

        Args:
            individual: Individual to test
            test_case: Test case to run

        Returns:
            Number of steps survived
        """
        return evaluate_best_protein_steps(
            individual.grn,
            self.output_mapper,
            self.grammar,
            test_case.state,
            physics_dt=self.physics_dt,
            physics_substeps=self.physics_substeps,
            grn_iterations_per_step=self.grn_iterations_per_step,
            initial_grn_iterations=self.initial_grn_iterations,
            max_time_steps=self.max_time_steps,
        )

    def test_individual(self, individual: Individual) -> GeneralisationResult:
        """Test individual on all 625 cases (parallel by default).

        Args:
            individual: Individual to test

        Returns:
            GeneralisationResult with statistics
        """
        if self.parallel and len(self.test_cases) > 1:
            return self._test_individual_parallel(individual)

        result = GeneralisationResult(individual=individual)
        for case in self.test_cases:
            steps = self.test_case(individual, case)
            result.steps_per_case.append(steps)
            if steps >= self.max_time_steps:
                result.successful_cases += 1
        return result

    def _test_individual_parallel(self, individual: Individual) -> GeneralisationResult:
        """Evaluate the 625 cases across processes; identical result to sequential.

        Results are collected in case order, so steps_per_case and the success
        count match the sequential path exactly (each case is deterministic given
        the genome + initial state).
        """
        # Pin numeric libraries to one thread per worker (set before the
        # forkserver context starts) to avoid CPU oversubscription.
        for _thread_var in (
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            os.environ.setdefault(_thread_var, "1")

        cases = self.test_cases
        bits = individual.genome.bits
        length = individual.genome.length
        n_workers = self.num_workers or os.cpu_count() or 1
        n_workers = min(n_workers, len(cases))
        ctx = mp.get_context("forkserver")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _run_generalisation_case_worker,
                    bits,
                    length,
                    (c.state.x, c.state.x_dot, c.state.theta, c.state.theta_dot),
                    self.grammar,
                    self.output_mapper,
                    self.max_time_steps,
                    self.grn_iterations_per_step,
                    self.initial_grn_iterations,
                    self.physics_dt,
                    self.physics_substeps,
                )
                for c in cases
            ]
            steps_per_case = [f.result() for f in futures]

        result = GeneralisationResult(individual=individual)
        result.steps_per_case = steps_per_case
        result.successful_cases = sum(
            1 for s in steps_per_case if s >= self.max_time_steps
        )
        return result
