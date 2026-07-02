"""Cart-Pole simulation module.

Implements the cart-pole (inverted pendulum) physics model
and simulation controller for the GRN-TAGE system.

Based on equations from:
"Differential Gene Expression with Tree-Adjunct Grammars"
Murphy et al., PPSN 2012
"""

from grntage.simulation.cartpole import CartPole, CartPoleState
from grntage.simulation.controller import SimulationController
from grntage.simulation.fitness import FitnessEvaluator, compute_fitness

__all__ = [
    "CartPole",
    "CartPoleState",
    "FitnessEvaluator",
    "SimulationController",
    "compute_fitness",
]
