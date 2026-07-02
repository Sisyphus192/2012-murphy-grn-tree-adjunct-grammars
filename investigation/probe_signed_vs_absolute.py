"""Probe #4: SIGNED vs ABSOLUTE Sort-by-Tendency — do they differ enough to matter?

KEY SOURCE DISCREPANCY (from thesis mining):
- 2012 PAPER: "sorted by the SIGNED magnitude of the tendency" + worked example puts P1(-0.128)
  LAST (despite largest |change|). => signed descending. OUR CODE matches this.
- THESIS (Ch.9): "sorted by the ABSOLUTE magnitude of the change ... the P-protein whose
  concentration changes the MOST ... first ... changed the LEAST ... last." => absolute descending.

The two contradict. Our code uses SIGNED. If ABSOLUTE produces a substantially different codon
stream (esp. the FIRST codon, which for SymReg picks the whole base expression via codon[0] % 400),
then re-running the 2 worst configs under ABSOLUTE is worth ~3h to see if it closes the gap.

Measure across genomes x states: how often the first codon and full order differ between the two
sort keys, and how often the resulting SymReg force alpha differs.
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import random  # noqa: E402

from grntage.grn.genome import Genome  # noqa: E402
from grntage.grn.network import GRN  # noqa: E402
from grntage.grammar.definitions import SYMBOLIC_REGRESSION_GRAMMAR  # noqa: E402
from grntage.grammar.mapper import GrammarMapper  # noqa: E402
from grntage.simulation.cartpole import (  # noqa: E402
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
    CartPole,
    CartPoleState,
)
from grntage.mapping.sort_tendency import SortByTendencyMapper  # noqa: E402
from grntage.simulation.controller import SimulationController  # noqa: E402


def random_state(r: random.Random) -> CartPoleState:
    return CartPoleState(
        x=r.uniform(*X_RANGE),
        x_dot=r.uniform(*X_DOT_RANGE),
        theta=r.uniform(*THETA_RANGE),
        theta_dot=r.uniform(*THETA_DOT_RANGE),
    )


gm = GrammarMapper(SYMBOLIC_REGRESSION_GRAMMAR)
rng = random.Random(123)
print(
    "=== SIGNED vs ABSOLUTE Sort-by-Tendency divergence (SymReg, 20 states/genome) ===\n"
)

tot_first_diff = tot_order_diff = tot_alpha_diff = tot = 0
for seed in [0, 3, 5, 7, 9, 12]:
    g = GRN(Genome.random(4096, random.Random(seed)))
    n_p = len(g.get_p_proteins())
    if n_p < 2:
        continue
    g.stabilize(max_iterations=10000)
    tf_snap = [p.concentration for p in g.tf_proteins]
    p_snap = [p.concentration for p in g.p_proteins]

    first_diff = order_diff = alpha_diff = 0
    for _ in range(20):
        for prot, c in zip(g.tf_proteins, tf_snap, strict=True):
            prot.concentration = c
        for prot, c in zip(g.p_proteins, p_snap, strict=True):
            prot.concentration = c
        g.free_tf_proteins = []
        state = random_state(rng)
        ctrl = SimulationController(
            g, CartPole(state), SortByTendencyMapper(), SYMBOLIC_REGRESSION_GRAMMAR
        )
        ctrl._inject_state()
        baseline = {p.signature: p.concentration for p in g.get_p_proteins()}
        g.iterate(2000)
        pj = g.get_p_proteins()
        changes = [(p.signature, p.concentration - baseline[p.signature]) for p in pj]

        signed_order = [s for s, _ in sorted(changes, key=lambda x: x[1], reverse=True)]
        absolute_order = [
            s for s, _ in sorted(changes, key=lambda x: abs(x[1]), reverse=True)
        ]

        if signed_order[0] != absolute_order[0]:
            first_diff += 1
        if signed_order != absolute_order:
            order_diff += 1
        a_signed = gm.map_and_evaluate(signed_order)
        a_abs = gm.map_and_evaluate(absolute_order)
        if abs(a_signed - a_abs) > 1e-9:
            alpha_diff += 1

    print(
        f"seed {seed:2d} (n_p={n_p:2d}): first-codon differs {first_diff:2d}/20, "
        f"full-order differs {order_diff:2d}/20, SymReg alpha differs {alpha_diff:2d}/20"
    )
    tot_first_diff += first_diff
    tot_order_diff += order_diff
    tot_alpha_diff += alpha_diff
    tot += 20

print(
    f"\nTOTAL: first-codon differs {tot_first_diff}/{tot} ({100 * tot_first_diff // tot}%), "
    f"full-order {tot_order_diff}/{tot} ({100 * tot_order_diff // tot}%), "
    f"alpha {tot_alpha_diff}/{tot} ({100 * tot_alpha_diff // tot}%)"
)
print(
    "\nIf alpha differs in a large fraction of states, ABSOLUTE is worth an EA re-run."
)
