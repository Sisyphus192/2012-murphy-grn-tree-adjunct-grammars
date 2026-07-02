"""Fitness evaluation for cart-pole control.

Implements the fitness function from Equation 5:
F(x) = 120000 / successful_time_steps - 1

Lower fitness is better. Goal is to maximize time steps.

Timing context:
- Each time step = one control cycle = 0.2s physics time (per paper p. 383)
- 120,000 steps = 24,000 seconds = 6.67 hours of simulated physics time
- Perfect fitness (0.0) requires balancing for the full 6.67 hours
"""


def compute_fitness(successful_steps: int) -> float:
    """Compute fitness from successful control cycles.

    Implements Equation 5: F(x) = 120000 / successful_time_steps - 1

    Args:
        successful_steps: Number of control cycles before failure.
            Each cycle = 0.2s physics time, so 120000 steps = 24000s.

    Returns:
        Fitness value (lower is better, 0.0 = perfect)
    """
    if successful_steps <= 0:
        return float("inf")

    return 120000.0 / successful_steps - 1.0


class FitnessEvaluator:
    """Evaluator for cart-pole controller fitness.

    Runs simulations and computes fitness scores.

    Timing context:
    - Each step = one control cycle = 0.2s physics time (per paper p. 383)
    - 120,000 steps = 24,000 seconds = 6.67 hours of simulated physics time

    Attributes:
        target_steps: Number of control cycles for perfect fitness (default 120000)
    """

    # Fitness thresholds
    PERFECT_FITNESS = 0.0  # 120000 control cycles = 24000s physics time
    SOLUTION_THRESHOLD = 0.0  # Success when fitness = 0

    def __init__(self, target_steps: int = 120000) -> None:
        """Initialize the evaluator.

        Args:
            target_steps: Number of control cycles for zero fitness.
                Default 120000 = 24000s = 6.67 hours physics time.
        """
        self.target_steps = target_steps

    def compute(self, successful_steps: int) -> float:
        """Compute fitness from successful control cycles.

        Args:
            successful_steps: Number of control cycles before failure

        Returns:
            Fitness value (lower is better)
        """
        return compute_fitness(successful_steps)

    def is_solution(self, fitness: float) -> bool:
        """Check if fitness represents a solution.

        Args:
            fitness: Fitness value

        Returns:
            True if this is considered a solution
        """
        return fitness <= self.SOLUTION_THRESHOLD

    def steps_for_fitness(self, fitness: float) -> int:
        """Calculate steps required for a given fitness.

        Inverse of fitness function.

        Args:
            fitness: Target fitness value

        Returns:
            Required number of steps
        """
        if fitness < 0:
            return self.target_steps
        return int(120000.0 / (fitness + 1.0))
