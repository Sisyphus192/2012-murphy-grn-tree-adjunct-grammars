"""Main evolutionary algorithm implementation.

Implements a generational EA with elitism for evolving
GRN-controlled cart-pole controllers.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from grntage.evolution.crossover import OnePointCrossover
from grntage.evolution.individual import Individual
from grntage.evolution.mutation import BitMutator
from grntage.evolution.population import Population
from grntage.evolution.selection import TournamentSelector
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
from grntage.simulation.fitness import compute_fitness

if TYPE_CHECKING:
    pass


@dataclass
class EAConfig:
    """Configuration for the evolutionary algorithm.

    Timing semantics (decision A.3.4/D1):
    - Each control cycle = grn_iterations_per_step GRN iterations + one physics
      control interval of physics_dt seconds, integrated in physics_substeps
      Euler sub-steps.
    - The faithful regime is physics_dt=0.02s, physics_substeps=1 (the classic
      Barto control/integration step; 120000 cycles = 2400s). A 0.2s control
      interval (the literal reading of "2000 GRN iters = 0.2s") was implemented
      and FALSIFIED: bang-bang +/-10N held for 0.2s tips the pole past 12deg in a
      single cycle, so the Direct-mapping grammar cannot survive one step -- yet
      the paper reports it at 47-50/50. The paper's control interval must
      therefore be ~0.02s ("0.2s" = GRN settling, not the cart-pole update rate).
      The physics_substeps machinery is kept (correct, tested) for accurate
      integration at coarser control intervals.

    Attributes:
        population_size: Number of individuals (default 250)
        generations: Number of generations (default 50)
        elite_size: Number of elites to preserve (default 25)
        tournament_size: Tournament selection size (default 3)
        mutation_rate: Per-bit mutation probability (default 0.005)
        genome_bits: Genome size in bits (default 4096)
        max_time_steps: Maximum control cycles per evaluation (default 120000).
            Each cycle = 0.02s physics time, so 120000 = 2400s.
        initial_grn_iterations: GRN stabilization iterations at start (default 10000)
        grn_iterations_per_step: GRN iterations per control cycle (default 2000)
        physics_dt: Cart-pole control interval in seconds — force held this long
            (decision A.3.4/D1, default 0.02 = classic Barto step)
        physics_substeps: Euler sub-steps per control interval (default 1 = single
            step; >1 for accurate integration at coarser control intervals)
        parallel: Enable parallel population evaluation (default True)
        num_workers: Number of worker processes (default: CPU count)
        crossover_rate: Per-pair one-point crossover probability (default 0.0 =
            mutation-only). The GRN genome is crossover-INTOLERANT: one-point bit
            crossover shreds contiguous gene/promoter structure, and an experiment
            (2026-06-25) showed it HURTS — Continuous*Tendency 66->56%,
            SymReg*Tendency 30->4% — worse the deeper the grammar. The predecessor
            (Nicolau 2010) used "only bit-flip mutation" for the same reason. The
            OnePointCrossover operator + --crossover-rate flag are retained so the
            falsification is reproducible; keep this 0.0 for faithful replication.
    """

    population_size: int = 250
    generations: int = 50
    elite_size: int = 25
    tournament_size: int = 3
    mutation_rate: float = 0.005
    crossover_rate: float = (
        0.0  # mutation-only is faithful; crossover falsified (see docstring)
    )
    genome_bits: int = 4096
    max_time_steps: int = 120000  # Control cycles (each = 0.02s physics time)
    initial_grn_iterations: int = 10000
    grn_iterations_per_step: int = 2000
    physics_dt: float = 0.02  # A.3.4/D1: control interval (s) = integration step
    physics_substeps: int = 1  # single Euler step (0.2s control unsolvable; see plan)
    parallel: bool = True
    num_workers: int | None = None  # None = use os.cpu_count()


@dataclass
class GenerationStats:
    """Statistics for a generation."""

    generation: int
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    best_steps: int


@dataclass
class EAResult:
    """Result of an evolutionary run.

    Attributes:
        best_individual: Best individual found
        generation_stats: Statistics per generation
        success: Whether a solution was found
        solution_generation: Generation solution was found (-1 if not)
    """

    best_individual: Individual
    generation_stats: list[GenerationStats] = field(default_factory=list)
    success: bool = False
    solution_generation: int = -1


# =============================================================================
# Module-scope worker function for multiprocessing
# Must be at module scope to be picklable by forkserver/spawn contexts
# =============================================================================


def _evaluate_individual_worker(
    genome_bits: int,
    genome_length: int,
    initial_state_tuple: tuple[float, float, float, float],
    grammar: Grammar,
    output_mapper: OutputMapper,
    max_time_steps: int,
    initial_grn_iterations: int,
    grn_iterations_per_step: int,
    physics_dt: float,
    physics_substeps: int,
) -> float:
    """Evaluate a single individual in a worker process.

    This function is called by ProcessPoolExecutor. Each call creates
    its own GRN and simulation components to ensure complete isolation.

    Args:
        genome_bits: Raw bit string as integer
        genome_length: Number of bits in genome
        initial_state_tuple: (x, x_dot, theta, theta_dot) cart-pole state
        grammar: Grammar for phenotype mapping
        output_mapper: Output mapping method (fresh instance per evaluation)
        max_time_steps: Maximum simulation steps
        initial_grn_iterations: GRN stabilization iterations
        grn_iterations_per_step: GRN iterations per physics step

    Returns:
        Fitness value (lower is better)
    """
    # Import here to ensure modules are loaded in worker process
    from grntage.grn.genome import Genome
    from grntage.grn.network import GRN

    # Reconstruct objects from picklable data
    genome = Genome(genome_bits, genome_length)
    grn = GRN(genome)

    initial_state = CartPoleState(
        x=initial_state_tuple[0],
        x_dot=initial_state_tuple[1],
        theta=initial_state_tuple[2],
        theta_dot=initial_state_tuple[3],
    )

    # Single-output (binary) methods select the best P-protein per individual
    # ("Best Product"/"Best Single Output"); multi-codon methods run once.
    steps = evaluate_best_protein_steps(
        grn,
        output_mapper,
        grammar,
        initial_state,
        physics_dt=physics_dt,
        physics_substeps=physics_substeps,
        grn_iterations_per_step=grn_iterations_per_step,
        initial_grn_iterations=initial_grn_iterations,
        max_time_steps=max_time_steps,
    )

    return compute_fitness(steps)


class EvolutionaryAlgorithm:
    """Evolutionary algorithm for cart-pole control.

    Uses tournament selection, bit mutation, and elitism.
    """

    def __init__(
        self,
        grammar: Grammar,
        output_mapper: OutputMapper,
        config: EAConfig | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the EA.

        Args:
            grammar: Grammar for phenotype mapping
            output_mapper: Output mapping method
            config: EA configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config or EAConfig()
        self.grammar = grammar
        self.output_mapper = output_mapper

        if random_seed is not None:
            random.seed(random_seed)

        self.selector = TournamentSelector(self.config.tournament_size)
        self.mutator = BitMutator(self.config.mutation_rate)
        self.crossover = OnePointCrossover(self.config.crossover_rate)

        self._on_generation: Callable[[int, GenerationStats], None] | None = None

    def set_generation_callback(
        self, callback: Callable[[int, GenerationStats], None]
    ) -> None:
        """Set callback for generation completion.

        Args:
            callback: Function called after each generation with (gen, stats)
        """
        self._on_generation = callback

    def _random_initial_state(self) -> CartPoleState:
        """Generate random initial cart-pole state.

        Returns:
            Random state within valid bounds
        """
        x = random.uniform(*X_RANGE)
        x_dot = random.uniform(*X_DOT_RANGE)
        theta = random.uniform(*THETA_RANGE)
        theta_dot = random.uniform(*THETA_DOT_RANGE)
        return CartPoleState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)

    def evaluate_individual(
        self, individual: Individual, initial_state: CartPoleState
    ) -> float:
        """Evaluate an individual's fitness.

        Args:
            individual: Individual to evaluate
            initial_state: Initial cart-pole state

        Returns:
            Fitness value (lower is better)
        """
        # Single-output (binary) methods select the best P-protein per individual
        # ("Best Product"/"Best Single Output"); multi-codon methods run once.
        steps = evaluate_best_protein_steps(
            individual.grn,
            self.output_mapper,
            self.grammar,
            initial_state,
            physics_dt=self.config.physics_dt,
            physics_substeps=self.config.physics_substeps,
            grn_iterations_per_step=self.config.grn_iterations_per_step,
            initial_grn_iterations=self.config.initial_grn_iterations,
            max_time_steps=self.config.max_time_steps,
        )

        return compute_fitness(steps)

    def evaluate_population(
        self, population: Population, initial_state: CartPoleState
    ) -> None:
        """Evaluate all unevaluated individuals in the population.

        Uses parallel evaluation if enabled in config and there are
        unevaluated individuals. Falls back to sequential evaluation
        when parallel is disabled or for small populations.

        Args:
            population: Population to evaluate
            initial_state: Initial cart-pole state for this generation
        """
        # Get unevaluated individuals
        unevaluated = [ind for ind in population if not ind.is_evaluated()]

        if not unevaluated:
            return

        # Use parallel evaluation if enabled and worthwhile
        if self.config.parallel and len(unevaluated) > 1:
            self._evaluate_population_parallel(unevaluated, initial_state)
        else:
            # Sequential evaluation
            for individual in unevaluated:
                individual.fitness = self.evaluate_individual(individual, initial_state)

    def _evaluate_population_parallel(
        self,
        individuals: list[Individual],
        initial_state: CartPoleState,
    ) -> None:
        """Evaluate individuals in parallel using ProcessPoolExecutor.

        Uses forkserver context for fork-safety with Numba JIT caching.
        Each worker process creates its own GRN and simulation components
        to ensure complete isolation.

        Args:
            individuals: List of individuals to evaluate
            initial_state: Initial cart-pole state for evaluation
        """
        # Pin each worker's numeric libraries to a single thread so that
        # N worker processes use N threads total, not N * cpu_count. The GRN
        # kernel is single-threaded numba today (no prange), but this prevents
        # oversubscription if a parallel kernel or a threaded BLAS call is ever
        # added. Set BEFORE the forkserver context starts so the server (and the
        # workers it forks) inherit these and numba reads them at its fresh
        # import; setdefault respects an explicit operator override.
        for _thread_var in (
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            os.environ.setdefault(_thread_var, "1")

        # Determine number of workers
        num_workers = self.config.num_workers
        if num_workers is None:
            num_workers = os.cpu_count() or 1

        # Don't use more workers than individuals
        num_workers = min(num_workers, len(individuals))

        # Convert initial state to tuple for pickling
        initial_state_tuple = (
            initial_state.x,
            initial_state.x_dot,
            initial_state.theta,
            initial_state.theta_dot,
        )

        # Use forkserver context (Python 3.14 default, fork-safe with Numba)
        ctx = mp.get_context("forkserver")

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
        ) as executor:
            # Submit all evaluation tasks
            futures = [
                executor.submit(
                    _evaluate_individual_worker,
                    genome_bits=ind.genome.bits,
                    genome_length=ind.genome.length,
                    initial_state_tuple=initial_state_tuple,
                    grammar=self.grammar,
                    output_mapper=self.output_mapper,
                    max_time_steps=self.config.max_time_steps,
                    initial_grn_iterations=self.config.initial_grn_iterations,
                    grn_iterations_per_step=self.config.grn_iterations_per_step,
                    physics_dt=self.config.physics_dt,
                    physics_substeps=self.config.physics_substeps,
                )
                for ind in individuals
            ]

            # Collect results and assign fitness values
            for individual, future in zip(individuals, futures):
                individual.fitness = future.result()

    def evaluate_population_sequential(
        self, population: Population, initial_state: CartPoleState
    ) -> None:
        """Evaluate all unevaluated individuals sequentially.

        This method is kept for testing and comparison purposes.

        Args:
            population: Population to evaluate
            initial_state: Initial cart-pole state for this generation
        """
        for individual in population:
            if not individual.is_evaluated():
                individual.fitness = self.evaluate_individual(individual, initial_state)

    def create_next_generation(self, population: Population) -> list[Individual]:
        """Create the next generation using selection, crossover, mutation, and elitism.

        Each iteration selects two parents, applies one-point crossover to produce
        two children, then mutates each child in place.  An odd ``num_offspring``
        is handled by truncating the last pair to one child.

        Args:
            population: Current population

        Returns:
            List of individuals for next generation (length == population_size)
        """
        population.sort_by_fitness()

        # Elites pass through unchanged (load-bearing: first elite_size entries)
        elites = [ind.copy() for ind in population.get_best(self.config.elite_size)]
        for elite in elites:
            elite.generation = population.generation + 1

        # Create offspring via selection, crossover, and mutation
        num_offspring = self.config.population_size - self.config.elite_size
        offspring: list[Individual] = []
        next_gen = population.generation + 1

        while len(offspring) < num_offspring:
            p1 = self.selector.select_one(population)
            p2 = self.selector.select_one(population)
            child1, child2 = self.crossover.cross(p1, p2)
            self.mutator.mutate_in_place(child1)
            child1.generation = next_gen
            offspring.append(child1)
            if len(offspring) < num_offspring:
                self.mutator.mutate_in_place(child2)
                child2.generation = next_gen
                offspring.append(child2)

        return elites + offspring

    def run(self, initial_population: Population | None = None) -> EAResult:
        """Run the evolutionary algorithm.

        Args:
            initial_population: Starting population (default: random)

        Returns:
            EAResult with best individual and statistics
        """
        # Initialize population
        if initial_population is None:
            population = Population.random(
                size=self.config.population_size,
                genome_bits=self.config.genome_bits,
            )
        else:
            population = initial_population

        result = EAResult(best_individual=population[0])
        best_ever = float("inf")

        for gen in range(self.config.generations):
            # Random initial state for this generation
            initial_state = self._random_initial_state()

            # A fresh random state each generation: re-score EVERY individual
            # (including carried-over elites) against it so the generation is a
            # level comparison (paper: "initialised with a random state at the
            # start of each generation").
            for individual in population:
                individual.fitness = float("inf")

            # Evaluate population
            self.evaluate_population(population, initial_state)

            # Collect statistics
            best_fitness = population.best_fitness()
            stats = GenerationStats(
                generation=gen,
                best_fitness=best_fitness,
                worst_fitness=population.worst_fitness(),
                mean_fitness=population.mean_fitness(),
                best_steps=int(120000 / (best_fitness + 1))
                if best_fitness >= 0
                else 120000,
            )
            result.generation_stats.append(stats)

            # Update best ever
            if best_fitness < best_ever:
                best_ever = best_fitness
                result.best_individual = population.get_best(1)[0].copy()

            # Check for solution
            if best_fitness <= 0.0:
                result.success = True
                result.solution_generation = gen
                break

            # Callback
            if self._on_generation:
                self._on_generation(gen, stats)

            # Create next generation (unless last)
            if gen < self.config.generations - 1:
                next_gen = self.create_next_generation(population)
                population.replace(next_gen)

        return result
