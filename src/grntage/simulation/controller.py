"""Simulation controller for GRN-controlled cart-pole.

Manages the interaction loop between the GRN, grammar mapper,
and cart-pole physics simulation.
"""

from grntage.grammar.definitions import Grammar
from grntage.grammar.mapper import GrammarMapper
from grntage.grn.constants import (
    INPUT_SIGNATURES,
    MAX_INPUT_CONCENTRATION,
    TOTAL_INPUT_CONCENTRATION,
)
from grntage.grn.network import GRN
from grntage.mapping.base import OutputMapper
from grntage.simulation.cartpole import (
    THETA_DOT_RANGE,
    THETA_RANGE,
    X_DOT_RANGE,
    X_RANGE,
    CartPole,
    CartPoleState,
)


class SimulationController:
    """Controller for GRN-cart-pole simulation.

    Implements the control loop from the paper (p. 383):
    "The GRN is then iterated 2000 times, corresponding to 0.2s of simulated
    time for the cart-pole model. After which, the P-proteins are mapped and
    the phenotype is evaluated resulting in a scaling co-efficient, α, for
    the force."

    Each step() call represents ONE CONTROL CYCLE:
    1. Inject cart-pole state into GRN as input proteins
    2. Run GRN dynamics for 2000 iterations
    3. Extract P-protein concentrations
    4. Map to control force α via grammar
    5. Apply constant force to cart-pole for 0.2s physics time

    Timing semantics:
    - One step() = 2000 GRN iterations + one physics update of 0.2s
    - 120,000 steps = 24,000 seconds = 6.67 hours of simulated physics time

    Attributes:
        grn: Gene regulatory network
        cartpole: Cart-pole physics simulation
        output_mapper: Maps P-proteins to codons
        grammar_mapper: Maps codons to control output
        grn_iterations_per_step: GRN iterations per control cycle (default 2000)
    """

    def __init__(
        self,
        grn: GRN,
        cartpole: CartPole,
        output_mapper: OutputMapper,
        grammar: Grammar,
        grn_iterations_per_step: int = 2000,
    ) -> None:
        """Initialize the controller.

        Args:
            grn: Gene regulatory network
            cartpole: Cart-pole simulation (should use dt=0.2 for paper compliance)
            output_mapper: P-protein to codon mapper
            grammar: Grammar for phenotype generation
            grn_iterations_per_step: GRN iterations per control cycle (default 2000).
                Per paper p. 383, 2000 iterations correspond to 0.2s physics time.
        """
        self.grn = grn
        self.cartpole = cartpole
        self.output_mapper = output_mapper
        self.grammar_mapper = GrammarMapper(grammar)
        self.grn_iterations_per_step = grn_iterations_per_step

    def step(self) -> tuple[float, bool]:
        """Execute one control cycle (2000 GRN iterations + 0.2s physics).

        Control cycle sequence:
        1. Inject cart-pole state into GRN as input protein concentrations
        2. Snapshot the injection-time P-protein concentrations (tendency baseline)
        3. Run GRN dynamics for grn_iterations_per_step (default 2000)
        4. Extract P-proteins and map to codons
        5. Map codons to control force α via grammar
        6. Apply constant force to cart-pole for dt seconds (default 0.2s)

        Each call represents 0.2s of simulated physics time (per paper p. 383).
        The fitness target of 120,000 steps = 24,000s = 6.67 hours physics time.

        Returns:
            Tuple of (force_alpha, is_valid) where:
            - force_alpha: Normalized force applied [-1.0, 1.0]
            - is_valid: Whether cart-pole state is still valid (within bounds)
        """
        # 1. Inject current state into GRN as input protein concentrations
        self._inject_state()

        # 2. Snapshot the injection-time P-protein concentrations as the tendency
        #    baseline BEFORE iterating. Paper sec. 3.1 defines concentration
        #    tendency as the signed change "from input-injection to the final
        #    iteration". For every step after the first this equals the previous
        #    cycle's final concentrations (injecting inputs only changes free
        #    TF-proteins, never the P-proteins), but on the FIRST step it gives a
        #    real, state-dependent baseline instead of None. Without it the first
        #    control action is identical for every initial state, so evolution
        #    gets no gradient on it -- fatal for the deep SymbolicRegression
        #    grammar (the Sort-by-Tendency x SymReg config could not be solved).
        baseline = self.output_mapper.get_previous_concentrations(
            self.grn.get_p_proteins()
        )

        # 3. Run GRN dynamics (single batch call).
        self.grn.iterate(self.grn_iterations_per_step)

        # 4. Extract P-proteins and map to codons against the injection baseline.
        p_proteins = self.grn.get_p_proteins()
        codons = self.output_mapper.map_to_codons(p_proteins, baseline)

        # 5. Map codons to control output using grammar.
        alpha = self.grammar_mapper.map_and_evaluate(codons)

        # 6. Apply force to cart-pole.
        is_valid = self.cartpole.step(alpha)

        return alpha, is_valid

    def _inject_state(self) -> None:
        """Inject the cart-pole state into the GRN as free TF-protein inputs.

        Each state variable is encoded into the concentration of its paper
        signature (``INPUT_SIGNATURES``), normalized into
        ``[0, MAX_INPUT_CONCENTRATION]`` (paper sec. 4: [0, 0.1]) so the four
        inputs take at most ``TOTAL_INPUT_CONCENTRATION`` (0.4) of total
        concentration. The four ranges are the canonical cart-pole state-variable
        ranges (cartpole.py), shared with random-init and the generalisation grid.
        """
        state = self.cartpole.state

        x = self._encode(state.x, X_RANGE[0], X_RANGE[1])
        x_dot = self._encode(state.x_dot, X_DOT_RANGE[0], X_DOT_RANGE[1])
        theta = self._encode(state.theta, THETA_RANGE[0], THETA_RANGE[1])
        theta_dot = self._encode(
            state.theta_dot, THETA_DOT_RANGE[0], THETA_DOT_RANGE[1]
        )

        assert x + x_dot + theta + theta_dot <= TOTAL_INPUT_CONCENTRATION + 1e-9

        self.grn.set_input_concentration(INPUT_SIGNATURES["x"], x)
        self.grn.set_input_concentration(INPUT_SIGNATURES["x_dot"], x_dot)
        self.grn.set_input_concentration(INPUT_SIGNATURES["theta"], theta)
        self.grn.set_input_concentration(INPUT_SIGNATURES["theta_dot"], theta_dot)

    @staticmethod
    def _encode(value: float, low: float, high: float) -> float:
        """Normalize ``value`` from [low, high] into [0, MAX_INPUT_CONCENTRATION]."""
        fraction = (value - low) / (high - low)
        fraction = max(0.0, min(1.0, fraction))
        return fraction * MAX_INPUT_CONCENTRATION

    def reset(self, state: CartPoleState | None = None) -> None:
        """Reset the simulation.

        Args:
            state: New cart-pole state (default: all zeros)
        """
        self.cartpole.reset(state)
        self.grn.reset()

    def run(self, max_steps: int) -> int:
        """Run simulation until failure or max steps reached.

        Each step is one control cycle (2000 GRN iterations + 0.2s physics).
        Total simulated time = max_steps * 0.2s.
        Example: max_steps=120000 → 24000s = 6.67 hours simulated time.

        Args:
            max_steps: Maximum number of control cycles to run

        Returns:
            Number of successful control cycles before failure (or max_steps)
        """
        for step in range(max_steps):
            _, is_valid = self.step()
            if not is_valid:
                return step
        return max_steps


def evaluate_best_protein_steps(
    grn: GRN,
    output_mapper: OutputMapper,
    grammar: Grammar,
    state: CartPoleState,
    *,
    physics_dt: float,
    physics_substeps: int,
    grn_iterations_per_step: int,
    initial_grn_iterations: int,
    max_time_steps: int,
) -> int:
    """Run the control rollout from ``state`` and return the steps survived.

    For the **single-output (binary) methods** (``output_mapper.is_binary()``),
    this implements the paper's "Best Product" / "Best Single Output" rule --
    ref [10] (Nicolau et al. 2010): *"all P-genes that are present in the genome
    are tested, and the most successful one is used"*. It tries each P-protein as
    the single output and returns the best survival (stopping early once one
    fully solves). For multi-codon methods (Sort-by-*), which already consume all
    P-proteins, it runs a single rollout.

    The initial (input-free) stabilization is **identical for every output-protein
    choice** -- P-proteins are output-only and never regulate, so the choice of
    which one drives the force does not affect the GRN's settling. So it is done
    ONCE and the stabilized state is restored before each candidate's rollout,
    avoiding an N-fold repeat of the 10000-iteration settle. This is exactly
    equivalent to reset+stabilize per candidate (pinned by
    test_best_protein_selection_returns_max_over_pgenes).
    """
    n_p = len(grn.get_p_proteins())
    if output_mapper.is_binary() and n_p > 1:
        candidates = list(range(n_p))
    else:
        candidates = [getattr(output_mapper, "protein_index", 0)]

    def _rollout(idx: int) -> int:
        if hasattr(output_mapper, "protein_index"):
            output_mapper.protein_index = idx
        output_mapper.reset()
        cartpole = CartPole(state=state, dt=physics_dt, substeps=physics_substeps)
        controller = SimulationController(
            grn=grn,
            cartpole=cartpole,
            output_mapper=output_mapper,
            grammar=grammar,
            grn_iterations_per_step=grn_iterations_per_step,
        )
        return controller.run(max_time_steps)

    # Stabilize once (input-free, choice-independent).
    grn.reset()
    grn.stabilize(max_iterations=initial_grn_iterations)
    if len(candidates) == 1:
        return _rollout(candidates[0])

    # Snapshot the settled, pre-injection state; restore it before each candidate.
    tf_snapshot = [p.concentration for p in grn.tf_proteins]
    p_snapshot = [p.concentration for p in grn.p_proteins]

    best_steps = 0
    for idx in candidates:
        for protein, conc in zip(grn.tf_proteins, tf_snapshot, strict=True):
            protein.concentration = conc
        for protein, conc in zip(grn.p_proteins, p_snapshot, strict=True):
            protein.concentration = conc
        grn.free_tf_proteins = []
        best_steps = max(best_steps, _rollout(idx))
        if best_steps >= max_time_steps:
            break  # already solves the full horizon; no need to try more proteins
    return best_steps
