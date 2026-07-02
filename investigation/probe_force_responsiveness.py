"""Probe #2: does degenerate tendency ordering make the control FORCE state-blind?

A balancer needs force that RESPONDS to the cart state (feedback). If the Sort-by-Tendency
codon order is degenerate (many P-proteins tied at ~0 change), the deep-grammar force may be
nearly constant across states -> no feedback -> cannot balance -> under-solve.

For each genome + the SymbolicRegression grammar, measure the std of the first-step force alpha
across 15 random initial states, for Sort-by-Tendency vs Sort-by-Concentration. Low std = the
force barely depends on the state (state-blind). Compare against the tendency tie-fraction.
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import random  # noqa: E402
import statistics  # noqa: E402

from grntage.grn.genome import Genome  # noqa: E402
from grntage.grn.network import GRN  # noqa: E402
from grntage.simulation.cartpole import (  # noqa: E402
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
    CartPole,
    CartPoleState,
)
from grntage.grammar.definitions import SYMBOLIC_REGRESSION_GRAMMAR  # noqa: E402
from grntage.mapping.sort_concentration import SortByConcentrationMapper  # noqa: E402
from grntage.mapping.sort_tendency import SortByTendencyMapper  # noqa: E402
from grntage.simulation.controller import SimulationController  # noqa: E402


def random_state(r: random.Random) -> CartPoleState:
    return CartPoleState(
        x=r.uniform(*X_RANGE),
        x_dot=r.uniform(*X_DOT_RANGE),
        theta=r.uniform(*THETA_RANGE),
        theta_dot=r.uniform(*THETA_DOT_RANGE),
    )


print("=== first-step force alpha responsiveness across 15 states (SymReg grammar) ===")
print("low std(alpha) => force is state-blind (no feedback => cannot balance)\n")
rng = random.Random(123)
for seed in [0, 3, 7, 12, 5, 9]:
    g = GRN(Genome.random(4096, random.Random(seed)))
    n_p = len(g.get_p_proteins())
    if n_p < 2:
        continue
    g.stabilize(max_iterations=10000)
    tf_snap = [p.concentration for p in g.tf_proteins]
    p_snap = [p.concentration for p in g.p_proteins]

    results = {}
    for label, mapper_cls in [
        ("tendency", SortByTendencyMapper),
        ("concentration", SortByConcentrationMapper),
    ]:
        alphas = []
        state_rng = random.Random(999)  # SAME states for both mappers
        for _ in range(15):
            for prot, c in zip(g.tf_proteins, tf_snap, strict=True):
                prot.concentration = c
            for prot, c in zip(g.p_proteins, p_snap, strict=True):
                prot.concentration = c
            g.free_tf_proteins = []
            state = random_state(state_rng)
            ctrl = SimulationController(
                g, CartPole(state), mapper_cls(), SYMBOLIC_REGRESSION_GRAMMAR
            )
            alpha, _ = ctrl.step()
            alphas.append(alpha)
        results[label] = alphas

    t = results["tendency"]
    c = results["concentration"]
    print(
        f"seed {seed:2d} (n_p={n_p:2d}): "
        f"TEND std(alpha)={statistics.pstdev(t):.4f} range=[{min(t):+.3f},{max(t):+.3f}] "
        f"distinct={len(set(round(a, 6) for a in t))}/15  ||  "
        f"CONC std(alpha)={statistics.pstdev(c):.4f} distinct={len(set(round(a, 6) for a in c))}/15"
    )
