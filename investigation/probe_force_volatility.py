"""Probe #3: confirm the code-audit's central claim — Sort-by-Tendency yields a JUMPIER
step-to-step force than Sort-by-Concentration, and it matters most for the finely-graded
deep grammars.

For each grammar x method, run a real multi-step rollout from a near-upright state and measure
mean step-to-step |delta-alpha| (force volatility) and the number of distinct force levels
(force resolution). Hypothesis: tendency >> concentration volatility, and Continuous/SymReg
have the finest force resolution -> a jumpy fine-graded force can't hold the pole over 120k steps.
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import random  # noqa: E402
import statistics  # noqa: E402

from grntage.grn.genome import Genome  # noqa: E402
from grntage.grn.network import GRN  # noqa: E402
from grntage.simulation.cartpole import CartPole, CartPoleState  # noqa: E402
from grntage.grammar.definitions import (  # noqa: E402
    CONTINUOUS_DIGITS_GRAMMAR,
    DISCRETE_DIGITS_GRAMMAR,
    SYMBOLIC_REGRESSION_GRAMMAR,
)
from grntage.mapping.sort_concentration import SortByConcentrationMapper  # noqa: E402
from grntage.mapping.sort_tendency import SortByTendencyMapper  # noqa: E402
from grntage.simulation.controller import SimulationController  # noqa: E402

GRAMMARS = [
    ("Discrete", DISCRETE_DIGITS_GRAMMAR),
    ("Continuous", CONTINUOUS_DIGITS_GRAMMAR),
    ("SymReg", SYMBOLIC_REGRESSION_GRAMMAR),
]
INIT = CartPoleState(theta=0.02)  # mild near-upright tip
MAX_STEPS = 120


def rollout_alphas(grn: GRN, mapper, grammar) -> list[float]:
    grn.reset()
    grn.stabilize(max_iterations=10000)
    ctrl = SimulationController(grn, CartPole(INIT), mapper, grammar)
    alphas = []
    for _ in range(MAX_STEPS):
        alpha, ok = ctrl.step()
        alphas.append(alpha)
        if not ok:
            break
    return alphas


print("=== force volatility over a real rollout (mean |delta-alpha| step-to-step) ===")
print(
    "hypothesis: tendency jumpier than concentration; worst for fine-graded deep grammars\n"
)
for gname, grammar in GRAMMARS:
    print(f"--- {gname} ---")
    for seed in [0, 3, 5]:
        row = {}
        for label, mcls in [
            ("tend", SortByTendencyMapper),
            ("conc", SortByConcentrationMapper),
        ]:
            g = GRN(Genome.random(4096, random.Random(seed)))
            a = rollout_alphas(g, mcls(), grammar)
            deltas = [abs(a[i] - a[i - 1]) for i in range(1, len(a))]
            vol = statistics.mean(deltas) if deltas else 0.0
            row[label] = (vol, len(set(round(x, 4) for x in a)), len(a))
        tv, tl, tn = row["tend"]
        cv, cl, cn = row["conc"]
        ratio = (tv / cv) if cv > 1e-9 else float("inf")
        print(
            f"  seed {seed}: TEND vol={tv:.3f} levels={tl:2d} steps={tn:3d} | "
            f"CONC vol={cv:.3f} levels={cl:2d} steps={cn:3d} | tend/conc vol ratio={ratio:.1f}x"
        )
