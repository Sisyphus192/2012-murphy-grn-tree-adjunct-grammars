"""Empirical probe: is the Sort-by-Tendency codon stream degenerate/starved for deep grammars?

Tests the prime hypothesis for the deep-grammar x Sort-by-Tendency under-solve:
- (A) How many P-proteins (codons) does a 4096-bit genome yield? Are deep grammars starved?
- (B) Are P-protein signatures unique (duplicate codons collapse diversity + baseline dict)?
- (C) Per control step, how many P-protein tendencies are ~0 (tie below 1e-10) -> degenerate
      stable-sort order? And does the TENDENCY codon ORDER vary across initial states
      (state-informative) vs the CONCENTRATION order? If tendency yields few distinct orders
      and high tie-fractions while concentration yields many, the hypothesis holds.
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import random  # noqa: E402

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
from grntage.mapping.sort_tendency import SortByTendencyMapper  # noqa: E402
from grntage.simulation.controller import SimulationController  # noqa: E402


def random_state(r: random.Random) -> CartPoleState:
    return CartPoleState(
        x=r.uniform(*X_RANGE),
        x_dot=r.uniform(*X_DOT_RANGE),
        theta=r.uniform(*THETA_RANGE),
        theta_dot=r.uniform(*THETA_DOT_RANGE),
    )


# (A) + (B) P-protein count & signature-uniqueness distribution over 30 seeds
print("=== (A/B) P-protein count and signature uniqueness over 30 seeds ===")
counts = []
uniq_fracs = []
for seed in range(30):
    g = GRN(Genome.random(4096, random.Random(seed)))
    ps = g.get_p_proteins()
    counts.append(len(ps))
    sigs = [p.signature for p in ps]
    uniq_fracs.append(len(set(sigs)) / len(sigs) if sigs else 1.0)
counts_sorted = sorted(counts)
print(f"  n_p counts: {counts_sorted}")
print(
    f"  n_p min/median/max: {min(counts)}/{counts_sorted[len(counts) // 2]}/{max(counts)}"
)
print(
    f"  mean distinct-signature fraction among P-proteins: "
    f"{sum(uniq_fracs) / len(uniq_fracs):.3f}  (1.0 = all unique)"
)

# (C) tendency degeneracy: tie fractions + distinct codon-order counts across states
print("\n=== (C) tendency ordering degeneracy (15 random states per genome) ===")
rng = random.Random(123)
for seed in [0, 3, 7, 12]:
    g = GRN(Genome.random(4096, random.Random(seed)))
    n_p = len(g.get_p_proteins())
    if n_p < 2:
        print(f"seed {seed}: n_p={n_p} (skip)")
        continue
    g.stabilize(max_iterations=10000)
    tf_snap = [p.concentration for p in g.tf_proteins]
    p_snap = [p.concentration for p in g.p_proteins]

    tie_fracs = []
    tend_orders = []
    conc_orders = []
    sample_changes = None
    for i in range(15):
        # restore the settled, pre-injection state (P-proteins are output-only)
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
        n_tied = sum(1 for _, c in changes if abs(c) < 1e-10)
        tie_fracs.append(n_tied / len(changes))
        tend_orders.append(
            tuple(s for s, _ in sorted(changes, key=lambda x: x[1], reverse=True))
        )
        conc_orders.append(
            tuple(
                p.signature
                for p in sorted(pj, key=lambda p: p.concentration, reverse=True)
            )
        )
        if i == 0:
            sample_changes = sorted([c for _, c in changes], reverse=True)

    print(f"\nseed {seed}: n_p={n_p}")
    print(
        f"  tie-frac (|change|<1e-10) per state: mean "
        f"{sum(tie_fracs) / len(tie_fracs):.2f}, max {max(tie_fracs):.2f}"
    )
    print(
        f"  distinct TENDENCY codon orders over 15 states:      {len(set(tend_orders))}/15"
    )
    print(
        f"  distinct CONCENTRATION codon orders over 15 states: {len(set(conc_orders))}/15"
    )
    if sample_changes:
        shown = ", ".join(f"{c:+.2e}" for c in sample_changes[:8])
        print(f"  sample sorted tendencies (state 0, top 8): {shown}")
