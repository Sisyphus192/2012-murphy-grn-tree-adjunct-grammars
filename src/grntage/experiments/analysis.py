"""Results analysis and reporting.

Computes statistics from experiment results and generates reports.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

from grntage.experiments.generalisation import GeneralisationResult
from grntage.experiments.runner import ExperimentResult


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment.

    Attributes:
        name: Experiment name
        num_runs: Total number of runs
        successful_runs: Number of successful runs
        success_rate: Proportion of successful runs
        mean_solution_generation: Mean generation solution was found
        median_solution_generation: Median generation solution was found
        best_fitness: Best fitness across all runs
        mean_best_fitness: Mean of best fitness per run
        generalisation_stats: Optional generalisation statistics
    """

    name: str
    num_runs: int
    successful_runs: int
    success_rate: float
    mean_solution_generation: float
    median_solution_generation: float
    best_fitness: float
    mean_best_fitness: float
    generalisation_mean: float = 0.0
    generalisation_median: float = 0.0
    generalisation_std: float = 0.0
    generalisation_best: int = 0
    generalisation_worst: int = 0


class ExperimentAnalyzer:
    """Analyzes experiment results and generates summaries."""

    def __init__(self, name: str) -> None:
        """Initialize analyzer.

        Args:
            name: Experiment name
        """
        self.name = name
        self.results: list[ExperimentResult] = []
        self.generalisation_results: list[GeneralisationResult] = []

    def add_results(self, results: list[ExperimentResult]) -> None:
        """Add experiment results.

        Args:
            results: List of experiment results
        """
        self.results.extend(results)

    def add_generalisation_results(self, results: list[GeneralisationResult]) -> None:
        """Add generalisation results.

        Args:
            results: List of generalisation results
        """
        self.generalisation_results.extend(results)

    def summarize(self) -> ExperimentSummary:
        """Generate experiment summary.

        Returns:
            ExperimentSummary with computed statistics
        """
        successful = [r for r in self.results if r.success]
        solution_gens = [r.solution_generation for r in successful]
        all_fitness = [r.best_fitness for r in self.results]

        summary = ExperimentSummary(
            name=self.name,
            num_runs=len(self.results),
            successful_runs=len(successful),
            success_rate=len(successful) / len(self.results) if self.results else 0.0,
            mean_solution_generation=(
                statistics.mean(solution_gens) if solution_gens else -1
            ),
            median_solution_generation=(
                statistics.median(solution_gens) if solution_gens else -1
            ),
            best_fitness=min(all_fitness) if all_fitness else float("inf"),
            mean_best_fitness=(
                statistics.mean(all_fitness) if all_fitness else float("inf")
            ),
        )

        # Add generalisation stats if available
        if self.generalisation_results:
            gen_cases = [r.successful_cases for r in self.generalisation_results]
            summary.generalisation_mean = statistics.mean(gen_cases)
            summary.generalisation_median = statistics.median(gen_cases)
            summary.generalisation_std = (
                statistics.stdev(gen_cases) if len(gen_cases) > 1 else 0.0
            )
            summary.generalisation_best = max(gen_cases)
            summary.generalisation_worst = min(gen_cases)

        return summary

    def get_convergence_data(self) -> list[list[float]]:
        """Get fitness convergence data per run.

        Returns:
            List of fitness values per generation per run
        """
        data = []
        for result in self.results:
            run_data = [s.best_fitness for s in result.ea_result.generation_stats]
            data.append(run_data)
        return data

    def to_csv(self, path: Path) -> None:
        """Export results to CSV.

        Args:
            path: Output file path
        """
        lines = ["run_id,seed,success,solution_generation,best_fitness"]
        for r in self.results:
            lines.append(
                f"{r.run_id},{r.seed},{r.success},{r.solution_generation},{r.best_fitness}"
            )
        path.write_text("\n".join(lines))

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.summarize()
        print(f"\n{'=' * 60}")
        print(f"Experiment: {summary.name}")
        print(f"{'=' * 60}")
        print(f"Runs: {summary.num_runs}")
        print(
            f"Successful: {summary.successful_runs} ({summary.success_rate * 100:.1f}%)"
        )
        print(f"Mean Solution Gen: {summary.mean_solution_generation:.1f}")
        print(f"Best Fitness: {summary.best_fitness:.4f}")
        if summary.generalisation_mean > 0:
            print("\nGeneralisation:")
            print(f"  Mean: {summary.generalisation_mean:.1f}")
            print(f"  Median: {summary.generalisation_median:.1f}")
            print(f"  Std: {summary.generalisation_std:.1f}")
            print(f"  Best: {summary.generalisation_best}")
            print(f"  Worst: {summary.generalisation_worst}")
        print(f"{'=' * 60}\n")
