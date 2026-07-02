"""Experiment runner for multi-run experiments.

Executes multiple evolutionary runs with different seeds
and collects results for analysis.
"""

from dataclasses import dataclass, field
from typing import Callable

from grntage.evolution.algorithm import (
    EAConfig,
    EAResult,
    EvolutionaryAlgorithm,
)
from grntage.grammar.definitions import Grammar
from grntage.mapping.base import OutputMapper


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Attributes:
        name: Experiment name
        grammar: Grammar to use
        output_mapper: Output mapping method
        ea_config: EA configuration
        num_runs: Number of runs (default 50)
        base_seed: Base random seed (default 0)
    """

    name: str
    grammar: Grammar
    output_mapper: OutputMapper
    ea_config: EAConfig = field(default_factory=EAConfig)
    num_runs: int = 50
    base_seed: int = 0


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        run_id: Run number
        seed: Random seed used
        ea_result: EA result
        success: Whether solution was found
        solution_generation: Generation solution found (-1 if not)
        best_fitness: Best fitness achieved
    """

    run_id: int
    seed: int
    ea_result: EAResult
    success: bool
    solution_generation: int
    best_fitness: float


class ExperimentRunner:
    """Runs multiple experiments with different configurations.

    Attributes:
        config: Experiment configuration
        results: Results from all runs
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results: list[ExperimentResult] = []
        self._on_run_complete: Callable[[int, ExperimentResult], None] | None = None

    def set_run_callback(
        self, callback: Callable[[int, ExperimentResult], None]
    ) -> None:
        """Set callback for run completion.

        Args:
            callback: Function called after each run with (run_id, result)
        """
        self._on_run_complete = callback

    def run_single(self, run_id: int, seed: int) -> ExperimentResult:
        """Run a single experiment.

        Args:
            run_id: Run identifier
            seed: Random seed

        Returns:
            ExperimentResult for this run
        """
        ea = EvolutionaryAlgorithm(
            grammar=self.config.grammar,
            output_mapper=self.config.output_mapper,
            config=self.config.ea_config,
            random_seed=seed,
        )

        ea_result = ea.run()

        return ExperimentResult(
            run_id=run_id,
            seed=seed,
            ea_result=ea_result,
            success=ea_result.success,
            solution_generation=ea_result.solution_generation,
            best_fitness=ea_result.best_individual.fitness,
        )

    def run_all(self) -> list[ExperimentResult]:
        """Run all experiments.

        Returns:
            List of all experiment results
        """
        self.results = []

        for run_id in range(self.config.num_runs):
            seed = self.config.base_seed + run_id
            result = self.run_single(run_id, seed)
            self.results.append(result)

            if self._on_run_complete:
                self._on_run_complete(run_id, result)

        return self.results

    def get_successful_runs(self) -> list[ExperimentResult]:
        """Get all successful runs."""
        return [r for r in self.results if r.success]

    def success_rate(self) -> float:
        """Compute success rate."""
        if not self.results:
            return 0.0
        return len(self.get_successful_runs()) / len(self.results)
