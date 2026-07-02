"""Evolutionary Algorithm module.

Implements the evolutionary algorithm for evolving GRN-controlled
cart-pole controllers using TAGE grammars.

Based on:
"Differential Gene Expression with Tree-Adjunct Grammars"
Murphy et al., PPSN 2012
"""

from grntage.evolution.algorithm import EvolutionaryAlgorithm
from grntage.evolution.individual import Individual
from grntage.evolution.mutation import BitMutator
from grntage.evolution.population import Population
from grntage.evolution.selection import TournamentSelector

__all__ = [
    "BitMutator",
    "EvolutionaryAlgorithm",
    "Individual",
    "Population",
    "TournamentSelector",
]
