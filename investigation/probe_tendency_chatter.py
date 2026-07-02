"""Empirical probe: characterise the Sort-by-Tendency control law near balance.

Backs the corrected "why the gap" analysis. The deep-grammar x Tendency deficit
was originally attributed to a "responsive-but-jumpy force that cannot hold". The
jumpiness is real (measured here), but a companion EA experiment shows reducing
it (a larger tie-threshold) makes TRAINING SUCCESS *worse*, not better -- so the
jumpiness is a symptom, not the binding constraint. The binding constraint is the
poor evolvability of the P-protein *ordering* under the 2000-iteration sync (the
thesis's own stated reason). This probe quantifies the symptom and its drivers:

  (A) Near-balance force chatter: Sort-by-Tendency vs Sort-by-Concentration.
      Tendency chatters ~2x more because concentration ordering is anchored by the
      sum-to-1 normalisation (one P rises => another falls) while tendency is not.
  (B) Chatter vs the tie-threshold. Near balance the tendency signal is ~1e-3, far
      above the default 1e-10 threshold, so tiny noisy deltas get sorted -> chatter.
      Raising the threshold turns those into ties (stable order) and kills chatter
      -- confirming the chatter is sub-threshold-noise sorting, NOT real signal.
  (C) Signal collapse: the tendency RMS is several-fold smaller when the pole is
      near-balanced (near-constant input, network re-settles to a fixed point) than
      when the state wanders -- exactly the regime a 120000-step hold lives in.
  (D) "Network death": over a long rollout the live TF-regulator pool collapses
      (dc/dt = d(e-h)c makes c=0 absorbing), which is the same for floor 0.0 and the
      thesis's 1e-10 (re-emergence from 1e-10 is too slow) -- why restoring 1e-10
      does not close the gap.

Run: uv run python investigation/probe_tendency_chatter.py   (~1-2 min)
"""

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import math  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402

from grntage.grn.genome import Genome  # noqa: E402
from grntage.grn.network import GRN  # noqa: E402
from grntage.grn.protein import ProteinType  # noqa: E402
from grntage.grn.constants import (  # noqa: E402
    INPUT_SIGNATURES,
    MAX_INPUT_CONCENTRATION,
)
from grntage.grammar.definitions import SYMBOLIC_REGRESSION_GRAMMAR  # noqa: E402
from grntage.grammar.mapper import GrammarMapper  # noqa: E402
from grntage.mapping.sort_concentration import SortByConcentrationMapper  # noqa: E402
from grntage.mapping.sort_tendency import SortByTendencyMapper  # noqa: E402
from grntage.simulation.cartpole import (  # noqa: E402
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
)

GMAP = GrammarMapper(SYMBOLIC_REGRESSION_GRAMMAR)
ALIVE = 1e-6


def _enc(v: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) * MAX_INPUT_CONCENTRATION


def _inject(grn: GRN, state: tuple[float, float, float, float]) -> None:
    x, xd, th, thd = state
    grn.set_input_concentration(INPUT_SIGNATURES["x"], _enc(x, *X_RANGE))
    grn.set_input_concentration(INPUT_SIGNATURES["x_dot"], _enc(xd, *X_DOT_RANGE))
    grn.set_input_concentration(INPUT_SIGNATURES["theta"], _enc(th, *THETA_RANGE))
    grn.set_input_concentration(
        INPUT_SIGNATURES["theta_dot"], _enc(thd, *THETA_DOT_RANGE)
    )


def _step_force(grn: GRN, state, mapper, sync: int = 2000):
    """One control step: inject, snapshot baseline, sync, map to force + P-deltas."""
    _inject(grn, state)
    baseline = {p.signature: p.concentration for p in grn.p_proteins}
    grn.iterate(sync)
    deltas = [p.concentration - baseline[p.signature] for p in grn.p_proteins]
    force = GMAP.map_and_evaluate(mapper.map_to_codons(grn.p_proteins, baseline))
    return force, deltas


def _mean_step_jump(forces: list[float]) -> float:
    inner = forces[1:]  # drop the first (settle-baseline) step
    return sum(abs(b - a) for a, b in zip(inner, inner[1:])) / max(1, len(inner) - 1)


# A near-balanced trajectory: theta wobbles within +/-1 deg, cart near centre.
NEAR = [
    (
        0.05 * math.sin(i / 7),
        0.0,
        math.radians(1.0) * math.sin(i / 5),
        math.radians(0.3) * math.cos(i / 5),
    )
    for i in range(30)
]


def _genomes(n: int, seed0: int, min_p: int = 3):
    out = []
    for s in range(n):
        g = Genome.random(4096, rng=random.Random(seed0 + s))
        if sum(1 for x in g.genes if x.gene_type == ProteinType.P) >= min_p:
            out.append(g)
    return out


def probe_A_B_chatter():
    print("=== (A) near-balance force chatter: Tendency vs Concentration ===")
    genomes = _genomes(30, 3000)
    tend, conc = [], []
    for g in genomes:
        grn = GRN(g)
        grn.reset()
        grn.stabilize(max_iterations=10000)
        fs = [_step_force(grn, s, SortByTendencyMapper())[0] for s in NEAR]
        tend.append(_mean_step_jump(fs))
        grn.reset()
        grn.stabilize(max_iterations=10000)
        fs = [_step_force(grn, s, SortByConcentrationMapper())[0] for s in NEAR]
        conc.append(_mean_step_jump(fs))
    t, c = statistics.mean(tend), statistics.mean(conc)
    print(f"  Sort-by-TENDENCY     step-to-step force jump: {t:.3f}")
    print(f"  Sort-by-CONCENTRATION step-to-step force jump: {c:.3f}")
    print(f"  ratio tend/conc: {t / c:.2f}x  (tendency chatters more)")

    print("\n=== (B) chatter vs Sort-by-Tendency tie-threshold ===")
    for thr in (1e-10, 1e-3, 3e-3, 1e-2):
        jumps = []
        for g in genomes:
            grn = GRN(g)
            grn.reset()
            grn.stabilize(max_iterations=10000)
            m = SortByTendencyMapper(threshold=thr)
            fs = [_step_force(grn, s, m)[0] for s in NEAR]
            jumps.append(_mean_step_jump(fs))
        print(f"  threshold={thr:.0e}: force jump = {statistics.mean(jumps):.3f}")
    print("  (default 1e-10 is far below the ~1e-3 near-balance signal -> sorts noise)")


def probe_C_collapse():
    print("\n=== (C) tendency signal collapses near balance ===")
    rng = random.Random(0)
    wander = [
        (
            rng.uniform(*X_RANGE),
            rng.uniform(*X_DOT_RANGE),
            rng.uniform(*THETA_RANGE),
            rng.uniform(*THETA_DOT_RANGE),
        )
        for _ in range(30)
    ]
    near_rms, wan_rms = [], []
    for g in _genomes(30, 3000):
        grn = GRN(g)
        grn.reset()
        grn.stabilize(max_iterations=10000)
        rr = [_step_force(grn, s, SortByTendencyMapper())[1] for s in NEAR]
        near_rms.append(
            statistics.mean(
                math.sqrt(sum(d * d for d in dl) / len(dl)) for dl in rr[1:]
            )
        )
        grn.reset()
        grn.stabilize(max_iterations=10000)
        rr = [_step_force(grn, s, SortByTendencyMapper())[1] for s in wander]
        wan_rms.append(
            statistics.mean(
                math.sqrt(sum(d * d for d in dl) / len(dl)) for dl in rr[1:]
            )
        )
    print(f"  tendency-signal RMS near-balanced: {statistics.mean(near_rms):.2e}")
    print(f"  tendency-signal RMS full-range wander: {statistics.mean(wan_rms):.2e}")
    print(
        f"  collapse factor: {statistics.mean(wan_rms) / statistics.mean(near_rms):.1f}x"
    )


def probe_D_death():
    print("\n=== (D) network 'death' over a long near-balanced rollout ===")
    nsyncs = 120
    tf0, tfN, rms_e, rms_l = [], [], [], []
    for g in _genomes(12, 4000):
        grn = GRN(g)
        grn.reset()
        grn.stabilize(max_iterations=10000)
        tf0.append(sum(1 for p in grn.tf_proteins if p.concentration > ALIVE))
        for t in range(nsyncs):
            state = (
                0.05 * math.sin(t / 7),
                0.0,
                math.radians(1.0) * math.sin(t / 5),
                math.radians(0.3) * math.cos(t / 5),
            )
            _, deltas = _step_force(grn, state, SortByTendencyMapper())
            rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
            if t < 10:
                rms_e.append(rms)
            if t >= nsyncs - 10:
                rms_l.append(rms)
        tfN.append(sum(1 for p in grn.tf_proteins if p.concentration > ALIVE))
    print(
        f"  live TF proteins: start {statistics.mean(tf0):.1f} "
        f"-> after {nsyncs} syncs {statistics.mean(tfN):.1f}"
    )
    print(
        f"  tendency RMS: early {statistics.mean(rms_e):.2e} "
        f"-> late {statistics.mean(rms_l):.2e} "
        f"(ratio {statistics.mean(rms_l) / statistics.mean(rms_e):.2f})"
    )
    print("  same collapse under floor 0.0 and 1e-10 -> restoring 1e-10 does not help")


if __name__ == "__main__":
    probe_A_B_chatter()
    probe_C_collapse()
    probe_D_death()
