"""Tests for evolutionary algorithm."""

import random


from grntage.evolution.algorithm import (
    EAConfig,
    EAResult,
    EvolutionaryAlgorithm,
    GenerationStats,
)
from grntage.evolution.population import Population
from grntage.grammar.definitions import DIRECT_MAPPING_GRAMMAR
from grntage.mapping.concentration import ConcentrationValueMapper
from grntage.simulation.cartpole import (
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
)


class TestEAConfig:
    """Tests for EAConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = EAConfig()
        assert config.population_size == 250
        assert config.generations == 50
        assert config.elite_size == 25
        assert config.tournament_size == 3
        assert config.mutation_rate == 0.005
        assert config.genome_bits == 4096
        assert config.max_time_steps == 120000
        assert config.physics_dt == 0.02  # A.3.4/D1: classic Barto control step (s)
        assert config.physics_substeps == 1  # single Euler step (0.2s unsolvable)
        assert (
            config.crossover_rate == 0.0
        )  # mutation-only faithful; crossover falsified

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EAConfig(
            population_size=100,
            generations=10,
            elite_size=5,
        )
        assert config.population_size == 100
        assert config.generations == 10
        assert config.elite_size == 5


class TestGenerationStats:
    """Tests for GenerationStats."""

    def test_creation(self) -> None:
        """Test creating generation stats."""
        stats = GenerationStats(
            generation=5,
            best_fitness=0.5,
            worst_fitness=10.0,
            mean_fitness=2.5,
            best_steps=60000,
        )
        assert stats.generation == 5
        assert stats.best_fitness == 0.5


class TestEvolutionaryAlgorithm:
    """Tests for EvolutionaryAlgorithm."""

    def test_create_ea(self) -> None:
        """Test creating EA with grammar and mapper."""
        grammar = DIRECT_MAPPING_GRAMMAR
        mapper = ConcentrationValueMapper()
        ea = EvolutionaryAlgorithm(grammar, mapper)
        assert ea.grammar is grammar
        assert ea.output_mapper is mapper

    def test_random_initial_state(self) -> None:
        """Random initial states lie within the canonical cart-pole ranges the
        input encoder uses, so no state-variable input saturates at init.

        Regression for the theta_dot range/units mismatch (encoder used
        +/-radians(1.5) while random-init used +/-1.5 rad/s, ~57x wider).
        """
        random.seed(42)
        ea = EvolutionaryAlgorithm(DIRECT_MAPPING_GRAMMAR, ConcentrationValueMapper())

        for _ in range(300):
            state = ea._random_initial_state()
            assert X_RANGE[0] <= state.x <= X_RANGE[1]
            assert X_DOT_RANGE[0] <= state.x_dot <= X_DOT_RANGE[1]
            assert THETA_RANGE[0] <= state.theta <= THETA_RANGE[1]
            assert THETA_DOT_RANGE[0] <= state.theta_dot <= THETA_DOT_RANGE[1]

    def test_short_run(self) -> None:
        """Test running EA for a few generations."""
        random.seed(42)
        config = EAConfig(
            population_size=10,
            generations=2,
            elite_size=2,
            genome_bits=256,  # Small genome
            max_time_steps=100,  # Short sim
            initial_grn_iterations=10,
            grn_iterations_per_step=10,
        )
        grammar = DIRECT_MAPPING_GRAMMAR
        mapper = ConcentrationValueMapper()
        ea = EvolutionaryAlgorithm(grammar, mapper, config=config)

        result = ea.run()
        assert isinstance(result, EAResult)
        assert len(result.generation_stats) == 2
        assert result.best_individual is not None

    def test_generation_callback(self) -> None:
        """Test generation callback is called."""
        random.seed(42)
        config = EAConfig(
            population_size=5,
            generations=3,
            elite_size=1,
            genome_bits=128,
            max_time_steps=50,
            initial_grn_iterations=5,
            grn_iterations_per_step=5,
        )
        grammar = DIRECT_MAPPING_GRAMMAR
        mapper = ConcentrationValueMapper()
        ea = EvolutionaryAlgorithm(grammar, mapper, config=config)

        generations_seen = []

        def callback(gen: int, stats: GenerationStats) -> None:
            generations_seen.append(gen)

        ea.set_generation_callback(callback)
        _ = ea.run()

        # Callback should be called for all but last generation
        assert 0 in generations_seen
        assert 1 in generations_seen

    def test_elitism_preserves_best(self) -> None:
        """Test that elitism preserves best individuals."""
        random.seed(42)
        config = EAConfig(
            population_size=10,
            generations=2,
            elite_size=3,
            genome_bits=128,
            max_time_steps=50,
            initial_grn_iterations=5,
            grn_iterations_per_step=5,
        )
        grammar = DIRECT_MAPPING_GRAMMAR
        mapper = ConcentrationValueMapper()
        ea = EvolutionaryAlgorithm(grammar, mapper, config=config)

        # Create population with known fitness
        pop = Population.random(size=10, genome_bits=128)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)

        next_gen = ea.create_next_generation(pop)
        assert len(next_gen) == 10
        # First 3 should be elites with same genome
        best = pop.get_best(3)
        for i in range(3):
            assert next_gen[i].genome.bits == best[i].genome.bits

    def test_elites_reevaluated_each_generation(self) -> None:
        """Every individual (incl. elites) is re-scored against each gen's state.

        With a fresh random state per generation, fitness cannot carry over, so
        the count of evaluations must be population_size * generations -- not the
        smaller offspring-only count that elite fitness caching would produce.
        """
        random.seed(0)
        config = EAConfig(
            population_size=6,
            generations=3,
            elite_size=2,
            genome_bits=128,
            max_time_steps=20,  # cannot "solve" (needs 120000), so no early stop
            initial_grn_iterations=5,
            grn_iterations_per_step=5,
            parallel=False,
        )
        ea = EvolutionaryAlgorithm(
            DIRECT_MAPPING_GRAMMAR, ConcentrationValueMapper(), config=config
        )

        calls = 0
        original = ea.evaluate_individual

        def counting(ind: object, state: object) -> float:
            nonlocal calls
            calls += 1
            return original(ind, state)  # type: ignore[arg-type]

        ea.evaluate_individual = counting  # type: ignore[method-assign]
        result = ea.run()

        assert calls == config.population_size * len(result.generation_stats)
        # Sanity: with elite caching this would be 6 + 4*2 = 14, not 18.
        assert calls == 18
