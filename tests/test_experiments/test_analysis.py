"""Tests for experiment analysis."""

import tempfile
from pathlib import Path


from grntage.evolution.algorithm import EAResult
from grntage.evolution.individual import Individual
from grntage.experiments.analysis import ExperimentAnalyzer, ExperimentSummary
from grntage.experiments.generalisation import GeneralisationResult
from grntage.experiments.runner import ExperimentResult


def make_result(
    run_id: int, success: bool, solution_gen: int, best_fitness: float
) -> ExperimentResult:
    """Create a mock experiment result."""
    ind = Individual.random()
    ind.fitness = best_fitness
    ea_result = EAResult(
        best_individual=ind,
        success=success,
        solution_generation=solution_gen,
    )
    return ExperimentResult(
        run_id=run_id,
        seed=run_id,
        ea_result=ea_result,
        success=success,
        solution_generation=solution_gen,
        best_fitness=best_fitness,
    )


class TestExperimentSummary:
    """Tests for ExperimentSummary."""

    def test_create_summary(self) -> None:
        """Test creating a summary."""
        summary = ExperimentSummary(
            name="test",
            num_runs=10,
            successful_runs=8,
            success_rate=0.8,
            mean_solution_generation=15.0,
            median_solution_generation=12.0,
            best_fitness=-0.5,
            mean_best_fitness=0.5,
        )
        assert summary.name == "test"
        assert summary.success_rate == 0.8


class TestExperimentAnalyzer:
    """Tests for ExperimentAnalyzer."""

    def test_create_analyzer(self) -> None:
        """Test creating an analyzer."""
        analyzer = ExperimentAnalyzer("test")
        assert analyzer.name == "test"
        assert len(analyzer.results) == 0

    def test_add_results(self) -> None:
        """Test adding results."""
        analyzer = ExperimentAnalyzer("test")
        results = [
            make_result(0, True, 10, -0.5),
            make_result(1, False, -1, 5.0),
        ]
        analyzer.add_results(results)
        assert len(analyzer.results) == 2

    def test_summarize(self) -> None:
        """Test generating summary."""
        analyzer = ExperimentAnalyzer("test")
        results = [
            make_result(0, True, 10, -0.5),
            make_result(1, True, 20, 0.0),
            make_result(2, False, -1, 5.0),
        ]
        analyzer.add_results(results)

        summary = analyzer.summarize()
        assert summary.num_runs == 3
        assert summary.successful_runs == 2
        assert abs(summary.success_rate - 2 / 3) < 0.01
        assert summary.mean_solution_generation == 15.0
        assert summary.best_fitness == -0.5

    def test_summarize_empty(self) -> None:
        """Test summarizing with no results."""
        analyzer = ExperimentAnalyzer("test")
        summary = analyzer.summarize()
        assert summary.num_runs == 0
        assert summary.success_rate == 0.0

    def test_add_generalisation_results(self) -> None:
        """Test adding generalisation results."""
        analyzer = ExperimentAnalyzer("test")
        ind = Individual.random()
        gen_result = GeneralisationResult(individual=ind)
        gen_result.successful_cases = 100
        analyzer.add_generalisation_results([gen_result])
        assert len(analyzer.generalisation_results) == 1

    def test_summarize_with_generalisation(self) -> None:
        """Test summarizing with generalisation data."""
        analyzer = ExperimentAnalyzer("test")
        analyzer.add_results([make_result(0, True, 10, 0.0)])

        for i in [100, 150, 200]:
            ind = Individual.random()
            gen_result = GeneralisationResult(individual=ind)
            gen_result.successful_cases = i
            analyzer.add_generalisation_results([gen_result])

        summary = analyzer.summarize()
        assert summary.generalisation_mean == 150.0
        assert summary.generalisation_median == 150.0
        assert summary.generalisation_best == 200
        assert summary.generalisation_worst == 100

    def test_to_csv(self) -> None:
        """Test exporting to CSV."""
        analyzer = ExperimentAnalyzer("test")
        results = [
            make_result(0, True, 10, -0.5),
            make_result(1, False, -1, 5.0),
        ]
        analyzer.add_results(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            analyzer.to_csv(path)

            content = path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 3  # Header + 2 data rows
            assert "run_id" in lines[0]
