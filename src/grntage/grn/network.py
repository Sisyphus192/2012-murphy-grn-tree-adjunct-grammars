"""Gene Regulatory Network dynamics.

Implements the GRN model from the paper including:
- Protein binding and regulation
- Expression rate calculations (Equations 1, 2, 3)
- Concentration normalization
- Steady-state detection

Performance optimized with Numba JIT compilation for the hot path.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from grntage.grn.constants import (
    BETA,
    DELTA,
    INITIAL_STABILIZATION_ITERATIONS,
    PER_STEP_ITERATIONS,
    STEADY_STATE_CONSECUTIVE,
    STEADY_STATE_THRESHOLD,
)
from grntage.grn.gene import Gene
from grntage.grn.genome import Genome
from grntage.grn.protein import Protein, ProteinType


# =============================================================================
# Numba JIT-compiled functions for hot path optimization
# =============================================================================


@njit(cache=True)
def _popcount32(x: np.int64) -> int:
    """Fast popcount for 32-bit integers using bit manipulation.

    Uses the parallel bit counting algorithm.
    Works with uint64 to avoid overflow issues with intermediate calculations.

    Args:
        x: Integer value (only lower 32 bits are counted)

    Returns:
        Number of set bits in the lower 32 bits
    """
    # Convert to unsigned 64-bit and mask to 32 bits
    val = np.uint64(x) & np.uint64(0xFFFFFFFF)
    val = val - ((val >> 1) & np.uint64(0x55555555))
    val = (val & np.uint64(0x33333333)) + ((val >> 2) & np.uint64(0x33333333))
    val = (val + (val >> 4)) & np.uint64(0x0F0F0F0F)
    result = (val * np.uint64(0x01010101)) >> 24
    return int(result & 0xFF)


@njit(cache=True)
def _compute_u_max_jit(
    protein_signatures: np.ndarray,
    enhancers: np.ndarray,
    inhibitors: np.ndarray,
) -> int:
    """Compute maximum complementary bits in the system.

    Args:
        protein_signatures: int64 array of protein signatures
        enhancers: int64 array of enhancer sites
        inhibitors: int64 array of inhibitor sites

    Returns:
        Maximum complementary bit count observed
    """
    u_max = 0
    n_proteins = len(protein_signatures)
    n_genes = len(enhancers)

    for g in range(n_genes):
        for p in range(n_proteins):
            sig = protein_signatures[p]
            # Enhancer
            xor_enh = sig ^ enhancers[g]
            u_enh = _popcount32(xor_enh)
            if u_enh > u_max:
                u_max = u_enh
            # Inhibitor
            xor_inh = sig ^ inhibitors[g]
            u_inh = _popcount32(xor_inh)
            if u_inh > u_max:
                u_max = u_inh

    return u_max


@njit(cache=True)
def _run_grn_jit(
    tf_signatures: np.ndarray,
    tf_concentrations: np.ndarray,
    tf_gene_indices: np.ndarray,
    p_signatures: np.ndarray,
    p_concentrations: np.ndarray,
    p_gene_indices: np.ndarray,
    free_tf_signatures: np.ndarray,
    free_tf_concentrations: np.ndarray,
    gene_indices: np.ndarray,
    gene_types: np.ndarray,
    enhancers: np.ndarray,
    inhibitors: np.ndarray,
    u_max: int,
    beta: float,
    delta: float,
    num_iterations: int,
    check_steady_state: bool,
    threshold: float,
    consecutive: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """JIT-compiled GRN settle loop (single kernel for iterate + stabilize).

    Runs the whole loop in nopython mode. The enhancing/inhibiting binding
    weights ``exp(beta * (u_j - u_max))`` depend only on the (constant) protein
    signatures, regulatory sites and ``u_max``, so they are precomputed ONCE into
    ``w_enh``/``w_inh`` and each per-iteration signal becomes a matrix-vector
    product. This is the dominant speed-up: the previous code recomputed every
    popcount and ``exp()`` on every iteration (e.g. 10000x during stabilize).

    Summation order is preserved -- ``sum_p(conc[p] * weight)`` then divide by N
    -- so under ``fastmath=False`` results are bit-identical to the pre-refactor
    inline loop (multiplication is commutative in IEEE-754 and ``exp`` of an
    identical argument is reproducible).

    When ``check_steady_state`` is set, the loop also tracks the maximum
    per-iteration concentration change and stops after ``consecutive`` iterations
    below ``threshold`` (the stabilize() criterion, evaluated in-kernel).

    Args:
        tf_signatures: TF protein signatures
        tf_concentrations: TF protein concentrations (copied internally)
        tf_gene_indices: Gene indices for TF proteins
        p_signatures: P protein signatures
        p_concentrations: P protein concentrations
        p_gene_indices: Gene indices for P proteins
        free_tf_signatures: Free TF (input) protein signatures
        free_tf_concentrations: Free TF (input) protein concentrations
        gene_indices: Array of gene indices
        gene_types: Array of gene types (0=TF, 1=P)
        enhancers: Enhancer sites per gene
        inhibitors: Inhibitor sites per gene
        u_max: Maximum complementary bits
        beta: Binding scaling factor
        delta: Expression rate scaling factor
        num_iterations: Maximum number of iterations to run
        check_steady_state: If True, stop early on steady state
        threshold: Steady-state max-change threshold
        consecutive: Consecutive sub-threshold iterations required to stop

    Returns:
        Tuple of (new_tf_concentrations, new_p_concentrations, iterations_run)
    """
    n_tf = len(tf_concentrations)
    n_p = len(p_concentrations)
    n_free = len(free_tf_concentrations)
    n_genes = len(gene_indices)
    n_all_tf = n_tf + n_free

    # Working copies (mutated in place across iterations).
    tf_conc = tf_concentrations.copy()
    p_conc = p_concentrations.copy()

    # Combined TF signatures (produced TF + free/input TF); constant across iters.
    all_tf_sigs = np.empty(n_all_tf, dtype=np.int64)
    for i in range(n_tf):
        all_tf_sigs[i] = tf_signatures[i]
    for i in range(n_free):
        all_tf_sigs[n_tf + i] = free_tf_signatures[i]

    # Gene-index -> local protein index lookups (-1 if no such protein).
    max_gene_idx = 0
    for i in range(n_genes):
        if gene_indices[i] > max_gene_idx:
            max_gene_idx = gene_indices[i]
    tf_lookup = np.full(max_gene_idx + 1, -1, dtype=np.int64)
    p_lookup = np.full(max_gene_idx + 1, -1, dtype=np.int64)
    for i in range(n_tf):
        idx = tf_gene_indices[i]
        if 0 <= idx <= max_gene_idx:
            tf_lookup[idx] = i
    for i in range(n_p):
        idx = p_gene_indices[i]
        if 0 <= idx <= max_gene_idx:
            p_lookup[idx] = i

    # Precompute the constant binding-weight matrices (Eq. 1 inner term):
    #   w_enh[g, p] = exp(beta * (popcount(sig_p ^ enhancer_g) - u_max))
    # They do not change between iterations, so the per-iteration signal is just
    # sum_p(conc[p] * w[g, p]) / n_all_tf.
    w_enh = np.zeros((n_genes, n_all_tf), dtype=np.float64)
    w_inh = np.zeros((n_genes, n_all_tf), dtype=np.float64)
    for g in range(n_genes):
        enh = enhancers[g]
        inh = inhibitors[g]
        for p in range(n_all_tf):
            sig = all_tf_sigs[p]
            u_enh = _popcount32(sig ^ enh)
            u_inh = _popcount32(sig ^ inh)
            w_enh[g, p] = np.exp(beta * (u_enh - u_max))
            w_inh[g, p] = np.exp(beta * (u_inh - u_max))

    # Hoisted per-iteration buffers (allocated once, reused every iteration).
    all_tf_concs = np.empty(n_all_tf, dtype=np.float64)
    new_tf_conc = np.empty(n_tf, dtype=np.float64)
    new_p_conc = np.empty(n_p, dtype=np.float64)
    for i in range(n_free):
        all_tf_concs[n_tf + i] = free_tf_concentrations[i]  # constant part

    # Free-TF total is constant (free concentrations never change in the loop).
    total_free = 0.0
    for i in range(n_free):
        total_free += free_tf_concentrations[i]

    stable_count = 0
    iterations_run = 0
    max_change = 0.0
    for _it in range(num_iterations):
        iterations_run = _it + 1

        # Refresh the produced-TF part of the combined concentration vector.
        for i in range(n_tf):
            all_tf_concs[i] = tf_conc[i]

        # Per gene: signals via matvec, then concentration update (Eq. 2/3).
        for g in range(n_genes):
            e_total = 0.0
            h_total = 0.0
            for p in range(n_all_tf):
                c = all_tf_concs[p]
                e_total += c * w_enh[g, p]
                h_total += c * w_inh[g, p]
            if n_all_tf > 0:
                e_i = e_total / n_all_tf
                h_i = h_total / n_all_tf
            else:
                e_i = 0.0
                h_i = 0.0

            gene_idx = gene_indices[g]
            if gene_types[g] == 0:  # TF gene -> rate carries the c_i factor (Eq. 2)
                local_idx = tf_lookup[gene_idx]
                if local_idx >= 0:
                    c_i = tf_conc[local_idx]
                    dc_dt = delta * (e_i - h_i) * c_i
                    new_tf_conc[local_idx] = max(0.0, c_i + dc_dt)
            else:  # P gene -> rate has no c_i factor (Eq. 3)
                local_idx = p_lookup[gene_idx]
                if local_idx >= 0:
                    c_i = p_conc[local_idx]
                    dc_dt = delta * (e_i - h_i)
                    new_p_conc[local_idx] = max(0.0, c_i + dc_dt)

        # Proteins without a corresponding gene keep their previous value.
        for i in range(n_tf):
            gene_idx = tf_gene_indices[i]
            if gene_idx < 0 or gene_idx > max_gene_idx or tf_lookup[gene_idx] != i:
                new_tf_conc[i] = tf_conc[i]
        for i in range(n_p):
            gene_idx = p_gene_indices[i]
            if gene_idx < 0 or gene_idx > max_gene_idx or p_lookup[gene_idx] != i:
                new_p_conc[i] = p_conc[i]

        # Normalize TF concentrations (produced TF + free TF sum to 1.0).
        total_produced = 0.0
        for i in range(n_tf):
            total_produced += new_tf_conc[i]
        if total_produced > 0.0:
            target = 1.0 - total_free
            if target > 0.0:
                scale = target / total_produced
                for i in range(n_tf):
                    new_tf_conc[i] *= scale
            else:
                for i in range(n_tf):
                    new_tf_conc[i] = 0.0

        # Normalize P concentrations (independently sum to 1.0).
        total_p = 0.0
        for i in range(n_p):
            total_p += new_p_conc[i]
        if total_p > 0.0:
            scale = 1.0 / total_p
            for i in range(n_p):
                new_p_conc[i] *= scale

        # Steady-state detection: max concentration change vs the previous state.
        if check_steady_state:
            max_change = 0.0
            for i in range(n_tf):
                d = new_tf_conc[i] - tf_conc[i]
                if d < 0.0:
                    d = -d
                if d > max_change:
                    max_change = d
            for i in range(n_p):
                d = new_p_conc[i] - p_conc[i]
                if d < 0.0:
                    d = -d
                if d > max_change:
                    max_change = d

        # Commit the new concentrations (reuse the working buffers next round).
        for i in range(n_tf):
            tf_conc[i] = new_tf_conc[i]
        for i in range(n_p):
            p_conc[i] = new_p_conc[i]

        if check_steady_state:
            if max_change < threshold:
                stable_count += 1
                if stable_count >= consecutive:
                    break
            else:
                stable_count = 0

    return tf_conc, p_conc, iterations_run


@dataclass
class GRNState:
    """Snapshot of GRN state for tracking changes."""

    tf_concentrations: list[float] = field(default_factory=list)
    p_concentrations: list[float] = field(default_factory=list)

    def max_change(self, other: "GRNState") -> float:
        """Calculate maximum concentration change from another state."""
        max_diff = 0.0
        for c1, c2 in zip(
            self.tf_concentrations, other.tf_concentrations, strict=False
        ):
            max_diff = max(max_diff, abs(c1 - c2))
        for c1, c2 in zip(self.p_concentrations, other.p_concentrations, strict=False):
            max_diff = max(max_diff, abs(c1 - c2))
        return max_diff


class GRN:
    """Gene Regulatory Network.

    Manages genes, proteins, and their dynamic interactions.
    """

    def __init__(self, genome: Genome) -> None:
        """Initialize GRN from a genome.

        Args:
            genome: Genome to extract genes from
        """
        self.genome = genome
        self.genes: list[Gene] = list(genome.genes)

        # Separate TF and P proteins
        self.tf_proteins: list[Protein] = []
        self.p_proteins: list[Protein] = []

        # Free TF-proteins (inputs)
        self.free_tf_proteins: list[Protein] = []

        # Initialize proteins from genes
        self._initialize_proteins()

        # Cache for u_max (maximum complementary bits observed)
        self._u_max: int = 0

    def _initialize_proteins(self) -> None:
        """Create proteins from genes with equal initial concentrations."""
        tf_genes = [g for g in self.genes if g.gene_type == ProteinType.TF]
        p_genes = [g for g in self.genes if g.gene_type == ProteinType.P]

        # Initialize TF-proteins with equal concentrations summing to 1.0
        if tf_genes:
            initial_tf_conc = 1.0 / len(tf_genes)
            for gene in tf_genes:
                protein = gene.create_protein(initial_tf_conc)
                self.tf_proteins.append(protein)

        # Initialize P-proteins with equal concentrations summing to 1.0
        if p_genes:
            initial_p_conc = 1.0 / len(p_genes)
            for gene in p_genes:
                protein = gene.create_protein(initial_p_conc)
                self.p_proteins.append(protein)

    def get_state(self) -> GRNState:
        """Get current state snapshot."""
        return GRNState(
            tf_concentrations=[p.concentration for p in self.tf_proteins],
            p_concentrations=[p.concentration for p in self.p_proteins],
        )

    def inject_inputs(self, input_proteins: list[Protein]) -> None:
        """Inject input proteins (free TF-proteins) into the system.

        Existing TF-protein concentrations are scaled to make room.

        Args:
            input_proteins: List of input proteins with concentrations
        """
        self.free_tf_proteins = input_proteins

        # Calculate total input concentration
        total_input = sum(p.concentration for p in input_proteins)

        # Scale existing TF-proteins
        if self.tf_proteins and total_input < 1.0:
            scale_factor = 1.0 - total_input
            for protein in self.tf_proteins:
                protein.concentration *= scale_factor

    def _compute_u_max(self) -> int:
        """Compute maximum complementary bits in the system."""
        u_max = 0
        all_tf = self.tf_proteins + self.free_tf_proteins

        for gene in self.genes:
            for protein in all_tf:
                u_enh = protein.count_complementary_bits(gene.enhancer)
                u_inh = protein.count_complementary_bits(gene.inhibitor)
                u_max = max(u_max, u_enh, u_inh)

        return u_max

    def _compute_signal(self, regulatory_site: int) -> float:
        """Compute enhancing or inhibiting signal for a regulatory site.

        Implements Equation 1 from the paper:
        e_i, h_i = (1/N) * sum(c_j * exp(beta * (u_j - u_max)))

        Args:
            regulatory_site: 32-bit regulatory site signature

        Returns:
            Signal strength
        """
        all_tf = self.tf_proteins + self.free_tf_proteins
        n = len(all_tf)
        if n == 0:
            return 0.0

        total = 0.0
        for protein in all_tf:
            u_j = protein.count_complementary_bits(regulatory_site)
            c_j = protein.concentration
            total += c_j * math.exp(BETA * (u_j - self._u_max))

        return total / n

    def _update_tf_protein(self, protein: Protein, gene: Gene) -> float:
        """Update TF-protein concentration.

        Implements Equation 2: dc_i/dt = delta * (e_i - h_i) * c_i

        Args:
            protein: TF-protein to update
            gene: Gene that produces this protein

        Returns:
            New concentration (before normalization)
        """
        e_i = self._compute_signal(gene.enhancer)
        h_i = self._compute_signal(gene.inhibitor)
        dc_dt = DELTA * (e_i - h_i) * protein.concentration
        return protein.concentration + dc_dt

    def _update_p_protein(self, protein: Protein, gene: Gene) -> float:
        """Update P-protein concentration.

        Implements Equation 3: dc_i/dt = delta * (e_i - h_i)
        Note: No multiplication by c_i for P-proteins.

        Args:
            protein: P-protein to update
            gene: Gene that produces this protein

        Returns:
            New concentration (before normalization)
        """
        e_i = self._compute_signal(gene.enhancer)
        h_i = self._compute_signal(gene.inhibitor)
        dc_dt = DELTA * (e_i - h_i)
        return protein.concentration + dc_dt

    def _normalize_concentrations(self) -> None:
        """Normalize protein concentrations.

        TF-proteins (produced) + free TF-proteins must sum to 1.0
        P-proteins must independently sum to 1.0
        """
        # Normalize TF-proteins (produced only, free stay fixed)
        total_free = sum(p.concentration for p in self.free_tf_proteins)
        total_produced = sum(p.concentration for p in self.tf_proteins)

        if total_produced > 0:
            target = 1.0 - total_free
            if target > 0:
                scale = target / total_produced
                for protein in self.tf_proteins:
                    protein.concentration *= scale
            else:
                # All concentration taken by inputs
                for protein in self.tf_proteins:
                    protein.concentration = 0.0

        # Normalize P-proteins
        total_p = sum(p.concentration for p in self.p_proteins)
        if total_p > 0:
            scale = 1.0 / total_p
            for protein in self.p_proteins:
                protein.concentration *= scale

    def _prepare_jit_arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Prepare NumPy arrays for JIT-compiled iteration.

        Returns:
            Tuple of arrays needed for _run_grn_jit
        """
        # TF proteins
        n_tf = len(self.tf_proteins)
        tf_signatures = np.empty(n_tf, dtype=np.int64)
        tf_concentrations = np.empty(n_tf, dtype=np.float64)
        tf_gene_indices = np.empty(n_tf, dtype=np.int64)
        for i, p in enumerate(self.tf_proteins):
            tf_signatures[i] = p.signature
            tf_concentrations[i] = p.concentration
            tf_gene_indices[i] = p.gene_index

        # P proteins
        n_p = len(self.p_proteins)
        p_signatures = np.empty(n_p, dtype=np.int64)
        p_concentrations = np.empty(n_p, dtype=np.float64)
        p_gene_indices = np.empty(n_p, dtype=np.int64)
        for i, p in enumerate(self.p_proteins):
            p_signatures[i] = p.signature
            p_concentrations[i] = p.concentration
            p_gene_indices[i] = p.gene_index

        # Free TF proteins (inputs)
        n_free = len(self.free_tf_proteins)
        free_tf_signatures = np.empty(n_free, dtype=np.int64)
        free_tf_concentrations = np.empty(n_free, dtype=np.float64)
        for i, p in enumerate(self.free_tf_proteins):
            free_tf_signatures[i] = p.signature
            free_tf_concentrations[i] = p.concentration

        # Genes
        n_genes = len(self.genes)
        gene_indices = np.empty(n_genes, dtype=np.int64)
        gene_types = np.empty(n_genes, dtype=np.int64)
        enhancers = np.empty(n_genes, dtype=np.int64)
        inhibitors = np.empty(n_genes, dtype=np.int64)
        for i, g in enumerate(self.genes):
            gene_indices[i] = g.index
            gene_types[i] = 0 if g.gene_type == ProteinType.TF else 1
            enhancers[i] = g.enhancer
            inhibitors[i] = g.inhibitor

        return (
            tf_signatures,
            tf_concentrations,
            tf_gene_indices,
            p_signatures,
            p_concentrations,
            p_gene_indices,
            free_tf_signatures,
            free_tf_concentrations,
            gene_indices,
            gene_types,
            enhancers,
            inhibitors,
        )

    def iterate(self, num_iterations: int = 1) -> None:
        """Run GRN for specified number of iterations (JIT-compiled).

        Args:
            num_iterations: Number of iterations to run
        """
        self._run_kernel(num_iterations, check_steady_state=False)

    def _run_kernel(
        self,
        num_iterations: int,
        *,
        check_steady_state: bool,
        threshold: float = STEADY_STATE_THRESHOLD,
        consecutive: int = STEADY_STATE_CONSECUTIVE,
    ) -> int:
        """Prepare arrays once, run the JIT settle kernel, apply results back.

        Shared by ``iterate()`` (fixed iteration count) and ``stabilize()`` (run
        until steady state). ``u_max`` depends only on the constant protein
        signatures and regulatory sites, so it is computed ONCE here rather than
        rebuilt per iteration (the old stabilize re-prepared arrays and recomputed
        u_max on every single iteration via ``iterate(1)``).

        Args:
            num_iterations: Maximum iterations to run
            check_steady_state: If True, stop early on steady state
            threshold: Steady-state max-change threshold
            consecutive: Consecutive sub-threshold iterations required to stop

        Returns:
            Number of iterations actually executed (0 if the GRN has no genes).
        """
        if not self.genes:
            return 0

        (
            tf_signatures,
            tf_concentrations,
            tf_gene_indices,
            p_signatures,
            p_concentrations,
            p_gene_indices,
            free_tf_signatures,
            free_tf_concentrations,
            gene_indices,
            gene_types,
            enhancers,
            inhibitors,
        ) = self._prepare_jit_arrays()

        # Compute u_max using JIT (constant for the whole settle).
        all_tf_sigs = np.concatenate([tf_signatures, free_tf_signatures])
        if len(all_tf_sigs) > 0:
            self._u_max = _compute_u_max_jit(all_tf_sigs, enhancers, inhibitors)
        else:
            self._u_max = 0

        new_tf_conc, new_p_conc, iterations_run = _run_grn_jit(
            tf_signatures,
            tf_concentrations,
            tf_gene_indices,
            p_signatures,
            p_concentrations,
            p_gene_indices,
            free_tf_signatures,
            free_tf_concentrations,
            gene_indices,
            gene_types,
            enhancers,
            inhibitors,
            self._u_max,
            BETA,
            DELTA,
            num_iterations,
            check_steady_state,
            threshold,
            consecutive,
        )

        # Apply results back to protein objects.
        for i, protein in enumerate(self.tf_proteins):
            protein.concentration = float(new_tf_conc[i])
        for i, protein in enumerate(self.p_proteins):
            protein.concentration = float(new_p_conc[i])

        return iterations_run

    def iterate_pure_python(self, num_iterations: int = 1) -> None:
        """Run GRN for specified number of iterations (pure Python version).

        This is the original non-JIT implementation, kept for testing and comparison.

        Args:
            num_iterations: Number of iterations to run
        """
        # Update u_max at start
        self._u_max = self._compute_u_max()

        # Build gene lookup for proteins
        tf_genes = {g.index: g for g in self.genes if g.gene_type == ProteinType.TF}
        p_genes = {g.index: g for g in self.genes if g.gene_type == ProteinType.P}

        for _ in range(num_iterations):
            # Calculate new concentrations for TF-proteins
            new_tf_concs = []
            for protein in self.tf_proteins:
                gene = tf_genes.get(protein.gene_index)
                if gene:
                    new_conc = self._update_tf_protein(protein, gene)
                    new_tf_concs.append(max(0.0, new_conc))
                else:
                    new_tf_concs.append(protein.concentration)

            # Calculate new concentrations for P-proteins
            new_p_concs = []
            for protein in self.p_proteins:
                gene = p_genes.get(protein.gene_index)
                if gene:
                    new_conc = self._update_p_protein(protein, gene)
                    new_p_concs.append(max(0.0, new_conc))
                else:
                    new_p_concs.append(protein.concentration)

            # Apply new concentrations
            for protein, conc in zip(self.tf_proteins, new_tf_concs, strict=True):
                protein.concentration = conc
            for protein, conc in zip(self.p_proteins, new_p_concs, strict=True):
                protein.concentration = conc

            # Normalize
            self._normalize_concentrations()

    def stabilize(
        self,
        max_iterations: int = INITIAL_STABILIZATION_ITERATIONS,
        threshold: float = STEADY_STATE_THRESHOLD,
        consecutive: int = STEADY_STATE_CONSECUTIVE,
    ) -> int:
        """Run GRN until steady state or max iterations (JIT-compiled).

        Steady state = ``consecutive`` successive iterations whose maximum
        concentration change is below ``threshold``. Both the iteration loop and
        the convergence check run inside the numba kernel, so the whole settle
        needs only one array preparation and one ``u_max`` computation (the
        previous version re-did both on every iteration via ``iterate(1)``).

        Args:
            max_iterations: Maximum iterations to run
            threshold: Maximum change threshold for steady state
            consecutive: Number of consecutive iterations below threshold

        Returns:
            Number of iterations actually run
        """
        return self._run_kernel(
            max_iterations,
            check_steady_state=True,
            threshold=threshold,
            consecutive=consecutive,
        )

    def run_with_input(
        self, input_proteins: list[Protein], iterations: int = PER_STEP_ITERATIONS
    ) -> None:
        """Inject inputs and run for specified iterations.

        Args:
            input_proteins: Input proteins to inject
            iterations: Number of iterations to run
        """
        self.inject_inputs(input_proteins)
        self.iterate(iterations)

    def get_p_protein_concentrations(self) -> list[tuple[Protein, float]]:
        """Get P-proteins sorted by concentration (descending).

        Returns:
            List of (protein, concentration) tuples sorted by concentration
        """
        return sorted(
            [(p, p.concentration) for p in self.p_proteins],
            key=lambda x: x[1],
            reverse=True,
        )

    def get_p_proteins(self) -> list[Protein]:
        """Get all P-proteins.

        Returns:
            List of P-proteins
        """
        return self.p_proteins

    def set_input_concentration(self, signature: int, concentration: float) -> None:
        """Set concentration of an input protein by signature.

        Creates or updates a free TF-protein with the given signature.

        Args:
            signature: 32-bit signature of the input protein
            concentration: Concentration value in [0, 1]
        """
        # Find existing input protein with this signature
        for protein in self.free_tf_proteins:
            if protein.signature == signature:
                protein.concentration = concentration
                return

        # Create new input protein
        input_protein = Protein(
            signature=signature,
            protein_type=ProteinType.TF,
            gene_index=-1,  # Input proteins don't come from genes
        )
        input_protein.concentration = concentration
        self.free_tf_proteins.append(input_protein)

    def step(self) -> None:
        """Execute one iteration of GRN dynamics.

        Convenience method for running a single iteration.
        """
        self.iterate(1)

    def reset(self) -> None:
        """Reset GRN to initial state.

        Re-initializes all proteins with equal concentrations.
        Clears free TF-proteins (inputs).
        """
        self.tf_proteins = []
        self.p_proteins = []
        self.free_tf_proteins = []
        self._u_max = 0
        self._initialize_proteins()
