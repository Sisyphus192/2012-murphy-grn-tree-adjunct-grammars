"""Script to run full experiments replicating the paper.

Runs 8 configurations (4 grammars × 2 output methods) with 50 runs each.

Usage:
    python -m grntage.experiments.run_experiments [--quick]
"""

import argparse
from pathlib import Path

from grntage.evolution.algorithm import EAConfig
from grntage.experiments.analysis import ExperimentAnalyzer
from grntage.experiments.generalisation import GeneralisationTester
from grntage.experiments.runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)
from grntage.grammar.definitions import (
    CONTINUOUS_DIGITS_GRAMMAR,
    DIRECT_MAPPING_GRAMMAR,
    DISCRETE_DIGITS_GRAMMAR,
    SYMBOLIC_REGRESSION_GRAMMAR,
)
from grntage.mapping.concentration import ConcentrationValueMapper
from grntage.mapping.sort_concentration import SortByConcentrationMapper
from grntage.mapping.sort_tendency import SortByTendencyMapper
from grntage.mapping.tendency import ConcentrationTendencyMapper


def get_configurations(
    ea_config: EAConfig, tendency_by_absolute: bool = False
) -> list[ExperimentConfig]:
    """Get all 8 experiment configurations.

    Mapper-Grammar compatibility:
    - Binary mappers (single codon): ConcentrationValueMapper, ConcentrationTendencyMapper
      -> Only compatible with DIRECT_MAPPING_GRAMMAR (2 choices: -1.0 or 1.0)
    - Multi-codon mappers: SortByConcentrationMapper, SortByTendencyMapper
      -> Compatible with Discrete/Continuous/Symbolic grammars (need multiple codons)

    The paper's "Concentration" and "Tendency" methods refer to multi-codon mappers
    (Sort by Concentration, Sort by Concentration Tendency) when used with non-direct grammars.
    """
    configs = []

    # Direct Mapping Grammar: uses binary mappers (single codon output)
    # These produce bang-bang control: -1.0 or 1.0
    configs.append(
        ExperimentConfig(
            name="DirectMapping_Concentration",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationValueMapper(),
            ea_config=ea_config,
        )
    )
    configs.append(
        ExperimentConfig(
            name="DirectMapping_Tendency",
            grammar=DIRECT_MAPPING_GRAMMAR,
            output_mapper=ConcentrationTendencyMapper(),
            ea_config=ea_config,
        )
    )

    # Non-direct grammars: use multi-codon mappers (Sort by X)
    # These produce variable force values based on protein signatures
    multi_codon_grammars = [
        ("DiscreteDigits", DISCRETE_DIGITS_GRAMMAR),
        ("ContinuousDigits", CONTINUOUS_DIGITS_GRAMMAR),
        ("SymbolicRegression", SYMBOLIC_REGRESSION_GRAMMAR),
    ]
    multi_codon_mappers = [
        ("Concentration", SortByConcentrationMapper()),
        ("Tendency", SortByTendencyMapper(by_absolute=tendency_by_absolute)),
    ]

    for grammar_name, grammar in multi_codon_grammars:
        for mapper_name, mapper in multi_codon_mappers:
            name = f"{grammar_name}_{mapper_name}"
            configs.append(
                ExperimentConfig(
                    name=name,
                    grammar=grammar,
                    output_mapper=mapper,
                    ea_config=ea_config,
                )
            )

    return configs


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    run_generalisation: bool = True,
) -> ExperimentAnalyzer:
    """Run a single experiment configuration."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config.name}")
    print(f"{'=' * 60}")

    runner = ExperimentRunner(config)

    def on_run(run_id: int, result: ExperimentResult) -> None:
        status = "✓" if result.success else "✗"
        gen = result.solution_generation if result.success else "-"
        print(f"  Run {run_id + 1:2}/{config.num_runs}: {status} gen={gen}")

    runner.set_run_callback(on_run)
    results = runner.run_all()

    analyzer = ExperimentAnalyzer(config.name)
    analyzer.add_results(results)

    # Run generalisation on successful runs
    if run_generalisation:
        successful = [r for r in results if r.success]
        if successful:
            print(f"\n  Generalisation testing ({len(successful)} solutions)...")
            # Forward the training EA's physics + GRN regime so generalisation is
            # measured under the SAME conditions the controllers were evolved in
            # (otherwise an EAConfig override would silently desync the two).
            tester = GeneralisationTester(
                grammar=config.grammar,
                output_mapper=config.output_mapper,
                grn_iterations_per_step=config.ea_config.grn_iterations_per_step,
                initial_grn_iterations=config.ea_config.initial_grn_iterations,
                physics_dt=config.ea_config.physics_dt,
                physics_substeps=config.ea_config.physics_substeps,
            )
            for result in successful:  # paper: generalise the best of every run
                gen_result = tester.test_individual(result.ea_result.best_individual)
                analyzer.add_generalisation_results([gen_result])
                print(f"    {gen_result.successful_cases}/625 cases solved")

    # Save results
    csv_path = output_dir / f"{config.name}_results.csv"
    analyzer.to_csv(csv_path)
    analyzer.print_summary()

    return analyzer


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run TAGE-GRN experiments")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: fewer runs/generations"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output directory"
    )
    parser.add_argument(
        "--configs", type=str, nargs="*", help="Specific configs to run"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=None,
        help="Per-pair one-point crossover probability (default: 0.0, mutation-only)",
    )
    parser.add_argument(
        "--tendency-sort",
        choices=["signed", "absolute"],
        default="signed",
        help="Sort-by-Tendency key: 'signed' (paper worked example, default) or "
        "'absolute' (thesis Ch.9 wording). Affects only the *_Tendency Sort configs.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Configure EA
    if args.quick:
        ea_config = EAConfig(
            population_size=50,
            generations=10,
            elite_size=5,
            genome_bits=2048,
            max_time_steps=10000,
            initial_grn_iterations=1000,
            grn_iterations_per_step=200,
        )
        num_runs = 5
    else:
        ea_config = EAConfig()  # Default paper parameters
        num_runs = 50

    if args.crossover_rate is not None:
        ea_config.crossover_rate = args.crossover_rate

    configs = get_configurations(
        ea_config, tendency_by_absolute=(args.tendency_sort == "absolute")
    )
    if args.configs:
        configs = [c for c in configs if c.name in args.configs]

    for config in configs:
        config.num_runs = num_runs

    # Run experiments
    analyzers = []
    for config in configs:
        analyzer = run_experiment(config, args.output, run_generalisation=True)
        analyzers.append(analyzer)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for analyzer in analyzers:
        summary = analyzer.summarize()
        gen = (
            f", gen={summary.generalisation_mean:.0f}"
            if summary.generalisation_mean
            else ""
        )
        print(f"{summary.name}: {summary.success_rate * 100:.0f}% success{gen}")


if __name__ == "__main__":
    main()
