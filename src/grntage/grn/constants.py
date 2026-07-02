"""Constants for the Gene Regulatory Network model.

Based on the paper "Differential Gene Expression with Tree-Adjunct Grammars"
Murphy et al., PPSN 2012
"""

# Gene structure sizes (in bits)
ENHANCER_BITS = 32
INHIBITOR_BITS = 32
PROMOTER_BITS = 32
GENE_INFO_BITS = 160  # 5 x 32-bit sections for majority vote
PROTEIN_BITS = 32

# Total gene size: 32 + 32 + 32 + 160 = 256 bits
GENE_TOTAL_BITS = ENHANCER_BITS + INHIBITOR_BITS + PROMOTER_BITS + GENE_INFO_BITS

# Promoter / gene-type signature (paper sec. 2.2).
#
# A 32-bit promoter is "XYZ" (24 arbitrary bits) followed by an 8-bit *type
# signature* that BOTH identifies a gene along the genome and sets its type:
#   - 0x00 (00000000) -> TF-gene  (produces a regulating TF-protein)
#   - 0xFF (11111111) -> P-gene   (produces an output-only P-protein)
# Any other 8-bit value is not a recognized promoter, so no gene starts there.
# There is no separate marker byte and no parity fallback: the paper identifies
# genes solely by this signature.
#
# Calibration knobs (see research/remediation_plan.md sec. 5):
#   D8 - which 8 bits are the "final" type signature. Under an MSB-first reading
#        of "XYZ<type>", the type is the low byte, so the default shift is 0.
#        Set to 24 to read the high byte instead.
#   D3 - overlap policy: scanning allows overlapping genes (see genome.py).
#   D4 - genome boundary: scanning is circular (see genome.py).
TYPE_SIGNATURE_SHIFT = 0  # D8: low 8 bits = the promoter's final type signature
TYPE_SIGNATURE_MASK = 0xFF
TF_GENE_SIGNATURE = 0x00  # type signature 00000000 -> TF-gene
P_GENE_SIGNATURE = 0xFF  # type signature 11111111 -> P-gene

# Scaling factors (from paper, both set to 1.0)
BETA = 1.0  # Binding scaling factor for enhancing/inhibiting signals
DELTA = 1.0  # Expression rate scaling factor

# GRN iteration parameters
INITIAL_STABILIZATION_ITERATIONS = 10000
PER_STEP_ITERATIONS = 2000
STEADY_STATE_THRESHOLD = 1e-6
STEADY_STATE_CONSECUTIVE = 100

# Input variable signatures (32-bit patterns for cart-pole state encoding)
INPUT_SIGNATURES = {
    "x": 0x00000000,  # Position: all zeros
    "x_dot": 0xFFFF0000,  # Velocity: upper 16 bits set
    "theta": 0xFFFFFFFF,  # Angle: all ones
    "theta_dot": 0x0000FFFF,  # Angular velocity: lower 16 bits set
}

# Maximum input concentration (each variable normalized to [0, 0.1])
MAX_INPUT_CONCENTRATION = 0.1
TOTAL_INPUT_CONCENTRATION = 0.4  # 4 variables x 0.1 each
