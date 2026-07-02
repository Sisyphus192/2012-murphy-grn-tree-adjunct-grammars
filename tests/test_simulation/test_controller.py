"""Tests for the GRN-cart-pole simulation controller (input encoding, paper sec. 4)."""

import math
import random

import pytest

from grntage.grammar.definitions import (
    DIRECT_MAPPING_GRAMMAR,
    SYMBOLIC_REGRESSION_GRAMMAR,
)
from grntage.grn.constants import (
    INPUT_SIGNATURES,
    MAX_INPUT_CONCENTRATION,
    TOTAL_INPUT_CONCENTRATION,
)
from grntage.grn.genome import Genome
from grntage.grn.network import GRN
from grntage.mapping.concentration import ConcentrationValueMapper
from grntage.mapping.sort_tendency import SortByTendencyMapper
from grntage.simulation.cartpole import CartPole, CartPoleState
from grntage.simulation.controller import (
    SimulationController,
    evaluate_best_protein_steps,
)


def _controller(state: CartPoleState) -> SimulationController:
    grn = GRN(Genome.random(4096, random.Random(0)))
    cartpole = CartPole(state=state)
    return SimulationController(
        grn, cartpole, ConcentrationValueMapper(), DIRECT_MAPPING_GRAMMAR
    )


def _injected(controller: SimulationController) -> dict[int, float]:
    controller._inject_state()
    return {p.signature: p.concentration for p in controller.grn.free_tf_proteins}


def test_inject_uses_paper_signatures() -> None:
    """Inputs use the paper's signatures (incl. x = all zeros), not local ones."""
    inputs = _injected(_controller(CartPoleState()))
    assert set(inputs) == set(INPUT_SIGNATURES.values())
    assert INPUT_SIGNATURES["x"] == 0x00000000


def test_inject_extreme_state_hits_caps() -> None:
    """At the range extremes each input == 0.1 and the total == 0.4 (paper sec. 4)."""
    state = CartPoleState(
        x=2.4, x_dot=1.0, theta=math.radians(12), theta_dot=math.radians(1.5)
    )
    inputs = _injected(_controller(state))
    for conc in inputs.values():
        assert conc == pytest.approx(MAX_INPUT_CONCENTRATION, abs=1e-9)
    assert sum(inputs.values()) == pytest.approx(TOTAL_INPUT_CONCENTRATION, abs=1e-9)


def test_inject_midpoint_is_half_max() -> None:
    """A centred state maps each variable to half of the per-input cap (0.05)."""
    inputs = _injected(_controller(CartPoleState()))  # all zeros = range midpoints
    for conc in inputs.values():
        assert conc == pytest.approx(MAX_INPUT_CONCENTRATION / 2, abs=1e-9)


def test_theta_dot_not_saturated_for_in_range_value() -> None:
    """An in-range theta_dot encodes strictly inside (0, 0.1) -- not clamped.

    Regression for the encoder/random-init/generalisation range mismatch that
    saturated the theta_dot input: a value at 60% of the canonical range must
    encode to an interior concentration, not the 0.0/0.1 caps.
    """
    from grntage.simulation.cartpole import THETA_DOT_RANGE

    theta_dot = 0.6 * THETA_DOT_RANGE[1]  # interior of the canonical range
    inputs = _injected(_controller(CartPoleState(theta_dot=theta_dot)))
    encoded = inputs[INPUT_SIGNATURES["theta_dot"]]
    assert 0.0 < encoded < MAX_INPUT_CONCENTRATION


def test_inject_never_exceeds_total_cap() -> None:
    """Out-of-range states are clamped so the total never exceeds 0.4."""
    state = CartPoleState(x=100.0, x_dot=-100.0, theta=100.0, theta_dot=100.0)
    inputs = _injected(_controller(state))
    assert sum(inputs.values()) <= TOTAL_INPUT_CONCENTRATION + 1e-9
    for conc in inputs.values():
        assert 0.0 <= conc <= MAX_INPUT_CONCENTRATION + 1e-9


def test_tendency_first_action_is_state_dependent() -> None:
    """The first control action must depend on the initial state (tendency path).

    Regression for the cold-start blindness: Sort-by-Tendency had no baseline on
    step 0, so every protein's change was 0, the sort was a no-op, and the codon
    order (hence the force) was identical for every initial state. That gave
    evolution zero gradient on the first action and made SymReg x Tendency
    unsolvable. The controller now snapshots the injection-time concentrations as
    the tendency baseline (paper sec. 3.1), so step 0 is state-dependent.
    """
    # SymReg consumes the full codon stream, so it is sensitive to the tendency
    # ordering (Discrete collapses to const(codon[0]) and would mask the fix).
    genome = Genome.random(4096, random.Random(2))  # many P-proteins -> real signal
    states = [
        CartPoleState(theta=-0.1),
        CartPoleState(theta=0.05),
        CartPoleState(theta=0.1),
        CartPoleState(x=2.0, x_dot=-0.8, theta=0.08, theta_dot=0.02),
    ]
    first_alphas = []
    for state in states:
        grn = GRN(genome)
        grn.stabilize(max_iterations=2000)
        controller = SimulationController(
            grn,
            CartPole(state=state),
            SortByTendencyMapper(),
            SYMBOLIC_REGRESSION_GRAMMAR,
            grn_iterations_per_step=2000,
        )
        alpha, _ = controller.step()
        first_alphas.append(alpha)

    assert len(set(first_alphas)) > 1, f"first action is state-blind: {first_alphas}"


def test_best_protein_selection_returns_max_over_pgenes() -> None:
    """Single-output (binary) methods use the BEST P-gene (paper 'Best Product' /
    'Best Single Output'; ref [10]: test all P-genes, use the most successful).

    evaluate_best_protein_steps must equal the max over per-protein single
    rollouts, and in particular be >= the old fixed-index-0 behaviour.
    """
    grammar = DIRECT_MAPPING_GRAMMAR
    genome = Genome.random(4096, random.Random(3))
    n_p = len(GRN(genome).get_p_proteins())
    assert n_p > 1  # need several P-proteins for the selection to be meaningful
    state = CartPoleState(theta=0.03)
    kw = dict(
        physics_dt=0.02,
        physics_substeps=1,
        grn_iterations_per_step=50,
        initial_grn_iterations=50,
        max_time_steps=200,
    )

    per_protein = []
    for idx in range(n_p):
        grn = GRN(genome)
        grn.reset()
        controller = SimulationController(
            grn,
            CartPole(state=state, dt=0.02, substeps=1),
            ConcentrationValueMapper(protein_index=idx),
            grammar,
            grn_iterations_per_step=50,
        )
        grn.stabilize(max_iterations=50)
        per_protein.append(controller.run(200))

    best = evaluate_best_protein_steps(
        GRN(genome), ConcentrationValueMapper(), grammar, state, **kw
    )
    assert best == max(per_protein)
    assert best >= per_protein[0]  # at least as good as the old index-0 default


def test_best_protein_single_rollout_for_multicodon() -> None:
    """Multi-codon (Sort) methods consume all P-proteins, so no selection: the
    helper runs a single rollout and returns a valid step count."""
    from grntage.mapping.sort_concentration import SortByConcentrationMapper

    grn = GRN(Genome.random(4096, random.Random(4)))
    steps = evaluate_best_protein_steps(
        grn,
        SortByConcentrationMapper(),
        SYMBOLIC_REGRESSION_GRAMMAR,
        CartPoleState(theta=0.03),
        physics_dt=0.02,
        physics_substeps=1,
        grn_iterations_per_step=50,
        initial_grn_iterations=50,
        max_time_steps=100,
    )
    assert 0 <= steps <= 100
