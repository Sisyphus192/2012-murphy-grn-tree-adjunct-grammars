"""Output mapping module for P-protein to codon conversion.

This module implements the four output mapping methods from:
"Differential Gene Expression with Tree-Adjunct Grammars"
Murphy et al., PPSN 2012

Methods:
1. Concentration Value (binary): Threshold at 0.5
2. Concentration Tendency (binary): Track concentration change
3. Sort by Concentration (multi-codon): Sort proteins by concentration
4. Sort by Concentration Tendency (multi-codon): Sort by change magnitude
"""

from grntage.mapping.base import OutputMapper, OutputMethod
from grntage.mapping.concentration import ConcentrationValueMapper
from grntage.mapping.sort_concentration import SortByConcentrationMapper
from grntage.mapping.sort_tendency import SortByTendencyMapper
from grntage.mapping.tendency import ConcentrationTendencyMapper

__all__ = [
    "ConcentrationTendencyMapper",
    "ConcentrationValueMapper",
    "OutputMapper",
    "OutputMethod",
    "SortByConcentrationMapper",
    "SortByTendencyMapper",
]
