#!/usr/bin/env python3
"""Scaled-up end-to-end validation of the evolutionary experiments.

This script runs a near-paper-scale version of the experiments to validate
the complete pipeline: GRN → Grammar Mapping → Cart-Pole → Fitness → Evolution

Parameters (targeting ~15 minute runtime):
- Population size: 200 (paper: 250)
- Generations: 50 (paper: 50, same as paper)
- Max time steps: 60,000 (paper: 120,000)
- GRN iterations/step: 2,000 (paper: 2,000, same as paper)
- Genome bits: 4,096 (same as paper)
- Single seed: 42 (paper: 50 runs with seeds 0-49)

Usage:
    uv run python scripts/validate_e2e.py
"""

import time

from grntage.evolution.algorithm import (
    EAConfig,
    EvolutionaryAlgorithm,
    GenerationStats,
)
from grntage.grammar.definitions import DISCRETE_DIGITS_GRAMMAR
from grntage.mapping.sort_tendency import SortByTendencyMapper


def main() -> None:
    """Run scaled-down validation experiment."""
    print("=" * 70)
    print("END-TO-END VALIDATION: Scaled-down Evolutionary Experiment")
    print("=" * 70)
    print()

    # Configuration matching ~100 second run for analysis
    config = EAConfig(
        population_size=200,  # Paper: 250
        generations=50,  # Paper: 50 (same as paper)
        elite_size=20,  # Paper: 25 (scaled with population)
        tournament_size=3,  # Same as paper
        mutation_rate=0.005,  # Same as paper
        genome_bits=4096,  # Paper: 4096 (same as paper)
        max_time_steps=60000,  # Paper: 120,000 (half for faster validation)
        initial_grn_iterations=10000,  # Paper: 10,000 (same as paper)
        grn_iterations_per_step=2000,  # Paper: 2,000 (same as paper)
        parallel=True,  # Use parallel evaluation
    )

    print("Configuration:")
    print(f"  Population size:      {config.population_size}")
    print(f"  Generations:          {config.generations}")
    print(f"  Elite size:           {config.elite_size}")
    print(f"  Max time steps:       {config.max_time_steps} control cycles")
    print("  Physics time/step:    0.2s")
    print(f"  Max simulated time:   {config.max_time_steps * 0.2:.0f}s")
    print(f"  GRN iters/step:       {config.grn_iterations_per_step}")
    print(f"  Genome bits:          {config.genome_bits}")
    print()

    # Use Sort by Tendency mapper (multi-codon) with Discrete Digits grammar
    # Note: Binary mappers (ConcentrationTendencyMapper) only work with DIRECT_MAPPING_GRAMMAR
    # Multi-codon mappers (SortByTendencyMapper) work with Discrete/Continuous/Symbolic grammars
    grammar = DISCRETE_DIGITS_GRAMMAR
    output_mapper = SortByTendencyMapper()

    print("Grammar:         Discrete Digits")
    print("Output Mapper:   Sort by Tendency (multi-codon)")
    print("Random Seed:     42")
    print()

    # Track fitness progression
    fitness_history: list[tuple[int, float, float, int]] = []

    def on_generation(gen: int, stats: GenerationStats) -> None:
        """Callback for each generation."""
        fitness_history.append(
            (gen, stats.best_fitness, stats.mean_fitness, stats.best_steps)
        )
        status = "SOLUTION!" if stats.best_fitness <= 0 else ""
        print(
            f"  Gen {gen:2d}: best={stats.best_fitness:8.2f}, "
            f"mean={stats.mean_fitness:8.2f}, steps={stats.best_steps:4d} {status}"
        )

    # Create and run EA
    ea = EvolutionaryAlgorithm(
        grammar=grammar,
        output_mapper=output_mapper,
        config=config,
        random_seed=42,
    )
    ea.set_generation_callback(on_generation)

    print("Running evolutionary algorithm...")
    print("-" * 70)

    start_time = time.time()
    result = ea.run()
    elapsed = time.time() - start_time

    print("-" * 70)
    print()

    # Report results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Elapsed time:         {elapsed:.1f}s")
    print(f"  Total generations:    {len(result.generation_stats)}")
    print(f"  Solution found:       {result.success}")
    if result.success:
        print(f"  Solution generation:  {result.solution_generation}")
    print(f"  Best fitness:         {result.best_individual.fitness:.4f}")
    print()

    # Verify fitness improved
    if len(fitness_history) >= 2:
        first_best = fitness_history[0][1]
        last_best = fitness_history[-1][1]
        improved = last_best < first_best
        print(f"  Fitness improved:     {improved}")
        print(f"    Initial best:       {first_best:.4f}")
        print(f"    Final best:         {last_best:.4f}")
        print(f"    Improvement:        {first_best - last_best:.4f}")
    print()

    # Summary
    print("=" * 70)
    if elapsed < 30 * 60:  # Under 30 minutes
        print("✓ VALIDATION PASSED: Experiment completed within time budget")
    else:
        print("⚠ WARNING: Experiment exceeded 30-minute time budget")

    if result.success or (
        len(fitness_history) >= 2 and fitness_history[-1][1] < fitness_history[0][1]
    ):
        print("✓ VALIDATION PASSED: Fitness improved over generations")
    else:
        print("⚠ NOTE: Fitness did not improve (may need more generations)")

    print("=" * 70)


if __name__ == "__main__":
    main()
