"""Tests for experiment runner."""

from grntage.evolution.algorithm import EAConfig
from grntage.experiments.runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)
from grntage.grammar.definitions import DIRECT_MAPPING_GRAMMAR
from grntage.mapping.concentration import ConcentrationValueMapper


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_config(self) -> None:
        """Test creating config with defaults."""
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        assert config.name == "test"
        assert config.num_runs == 50
        assert config.base_seed == 0

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
            num_runs=10,
            base_seed=100,
        )
        assert config.num_runs == 10
        assert config.base_seed == 100


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    def test_create_runner(self) -> None:
        """Test creating experiment runner."""
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        runner = ExperimentRunner(config)
        assert runner.config is config
        assert len(runner.results) == 0

    def test_run_single(self) -> None:
        """Test running a single experiment."""
        ea_config = EAConfig(
            population_size=5,
            generations=2,
            elite_size=1,
            genome_bits=128,
            max_time_steps=50,
            initial_grn_iterations=5,
            grn_iterations_per_step=5,
        )
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
            ea_config=ea_config,
        )
        runner = ExperimentRunner(config)

        result = runner.run_single(0, 42)
        assert isinstance(result, ExperimentResult)
        assert result.run_id == 0
        assert result.seed == 42

    def test_run_callback(self) -> None:
        """Test run completion callback."""
        ea_config = EAConfig(
            population_size=3,
            generations=1,
            elite_size=1,
            genome_bits=64,
            max_time_steps=20,
            initial_grn_iterations=2,
            grn_iterations_per_step=2,
        )
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
            ea_config=ea_config,
            num_runs=2,
        )
        runner = ExperimentRunner(config)

        callbacks_received = []

        def callback(run_id: int, result: ExperimentResult) -> None:
            callbacks_received.append(run_id)

        runner.set_run_callback(callback)
        runner.run_all()

        assert callbacks_received == [0, 1]

    def test_success_rate_empty(self) -> None:
        """Test success rate with no results."""
        config = ExperimentConfig(
            name="test",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
        )
        runner = ExperimentRunner(config)
        assert runner.success_rate() == 0.0
