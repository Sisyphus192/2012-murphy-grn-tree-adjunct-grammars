"""Experimentation Framework module.

Provides tools for running multi-run experiments, generalization tests,
and results analysis.

Based on:
"Differential Gene Expression with Tree-Adjunct Grammars"
Murphy et al., PPSN 2012
"""

from grntage.experiments.analysis import ExperimentAnalyzer, ExperimentSummary
from grntage.experiments.generalisation import (
    GeneralisationCase,
    GeneralisationResult,
    GeneralisationTester,
)
from grntage.experiments.runner import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
)

__all__ = [
    "ExperimentAnalyzer",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSummary",
    "GeneralisationCase",
    "GeneralisationResult",
    "GeneralisationTester",
]
