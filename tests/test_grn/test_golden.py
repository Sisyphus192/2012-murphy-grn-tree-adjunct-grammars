"""Golden-master safety net for the GRN dynamics (A.4 performance refactor).

These tests pin fixed-seed GRN concentration trajectories captured BEFORE the
A.4 numba refactor (JIT settle kernel + precomputed weight matrix), so any
change to the iteration math is caught. Capture/regenerate with:

    uv run pytest tests/test_grn/test_golden.py --update-golden

Tolerance is rtol=1e-9 / atol=1e-12 (the `golden` fixture default): the matvec
refactor preserves the summation order, but fastmath/reduction differences must
not be pinned to bit-equality. The GRN JIT runs with ``fastmath=False`` (D7) so
these trajectories are deterministic across runs.
"""

import random
from collections.abc import Callable

import numpy as np

from grntage.grn.constants import INPUT_SIGNATURES
from grntage.grn.genome import Genome
from grntage.grn.network import GRN

# Fixed network: seed 12345 @ 4096 bits -> the pinned 22-gene (15 TF / 7 P) GRN.
# This network does NOT reach steady state (stabilize runs the full 10000 cap),
# so it pins the no-convergence branch of the settle loop.
GOLDEN_SEED = 12345
GOLDEN_BITS = 4096

# A network that DOES converge before the cap, so the stabilize early-break path
# (stable_count / consecutive / returned iteration count) is pinned too.
CONVERGING_SEED = 7  # stabilize() converges at iteration 3386

# Cumulative iteration checkpoints for the settle trajectory.
SETTLE_CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def _golden_grn() -> GRN:
    return GRN(Genome.random(GOLDEN_BITS, random.Random(GOLDEN_SEED)))


def _snapshot(grn: GRN) -> list[float]:
    """Full concentration vector: TF proteins then P proteins, gene order."""
    return [p.concentration for p in grn.tf_proteins] + [
        p.concentration for p in grn.p_proteins
    ]


def test_golden_settle_trajectory(golden: Callable[..., None]) -> None:
    """Input-free iteration math, pinned at a range of iteration counts."""
    grn = _golden_grn()
    rows = []
    prev = 0
    for checkpoint in SETTLE_CHECKPOINTS:
        grn.iterate(checkpoint - prev)
        prev = checkpoint
        rows.append(_snapshot(grn))
    golden("grn_settle_trajectory", np.array(rows, dtype=np.float64))


def test_golden_input_trajectory(golden: Callable[..., None]) -> None:
    """Input-injection + iteration math, using the paper INPUT_SIGNATURES.

    Mirrors the control loop (controller.step): the four paper input signatures
    are held at an interior concentration (0.05, the centred-state encoding) and
    the GRN is run for repeated 2000-iteration control cycles.
    """
    grn = _golden_grn()
    rows = []
    for _ in range(4):
        for sig in INPUT_SIGNATURES.values():
            grn.set_input_concentration(sig, 0.05)
        grn.iterate(2000)
        rows.append(_snapshot(grn))
    golden("grn_input_trajectory", np.array(rows, dtype=np.float64))


def test_golden_stabilize(golden: Callable[..., None]) -> None:
    """stabilize() final state and its iteration count (no-convergence branch).

    The seed-12345 network never settles, so this pins the "run to the 10000
    cap" path.
    """
    grn = _golden_grn()
    iters = grn.stabilize()
    golden("grn_stabilize_final", np.array(_snapshot(grn), dtype=np.float64))
    golden("grn_stabilize_iters", np.array([iters], dtype=np.int64), exact=True)


def test_golden_stabilize_converging(golden: Callable[..., None]) -> None:
    """stabilize() on a network that converges before the cap.

    Exercises the early-break steady-state logic (stable_count / consecutive
    reset / returned iteration count) that the non-converging case above cannot
    reach -- the most behaviourally novel part of the A.4 JIT settle kernel.
    """
    grn = GRN(Genome.random(GOLDEN_BITS, random.Random(CONVERGING_SEED)))
    iters = grn.stabilize()
    assert iters < 10000, "converging-seed network must break before the cap"
    golden("grn_stabilize_conv_final", np.array(_snapshot(grn), dtype=np.float64))
    golden("grn_stabilize_conv_iters", np.array([iters], dtype=np.int64), exact=True)
