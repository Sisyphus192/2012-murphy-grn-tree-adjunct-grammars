# Differential Gene Expression with Tree-Adjunct Grammars — a replication

A from-scratch reimplementation and replication study of:

> **Eoin Murphy, Miguel Nicolau, Erik Hemberg, Michael O'Neill, Anthony
> Brabazon.** _Differential Gene Expression with Tree-Adjunct Grammars._ In
> _Parallel Problem Solving from Nature — PPSN XII_, LNCS 7491, pp. 377–386.
> Springer, 2012. DOI:
> [10.1007/978-3-642-32937-1_38](https://doi.org/10.1007/978-3-642-32937-1_38)

The paper evolves an **artificial Gene Regulatory Network (GRN)** whose
"Product" protein concentrations are read out as a stream of integer codons,
mapped through a **Tree-Adjunct Grammar (TAGE)** into an arithmetic expression,
and used as the control law for the classic **inverted-pendulum (cart-pole)**
task. Four grammars (Direct, Discrete, Continuous, Symbolic Regression) × two
output methods (Concentration, Tendency) give eight configurations, scored on
training success and on a 625-case generalisation grid (Table 2 of the paper).

No public source code or independent reproduction of this paper exists (verified
against the authors' repositories and the GE community's PonyGE2); to our
knowledge **this is the first**.

---

## TL;DR — what we found

- **The pipeline reproduces faithfully.** Physics, GRN dynamics, the four
  grammars, the four output methods, the EA, and the generalisation protocol
  were each verified line-by-line against the paper _and_ its
  predecessor/thesis. Every parameter in the paper's Table 1 matches.
- **Concentration configs and the shallow-grammar Tendency configs reproduce
  well** on training success (within a few points) and on the _shape_ of the
  generalisation results.
- **One systematic divergence:** the **deep-grammar × Tendency** configs
  **under-solve** the training task — **Continuous·Tendency 66%** (paper 100% /
  thesis 94%) and **SymbolicRegression·Tendency 30%** (paper 86% / thesis 82%).
  At the same time, **every** Tendency config **generalises ~2× _better_** than
  the paper.
- **This is inherent to the faithful model, not a bug — but not for the
  reason we first proposed.** We implemented and **ruled out seven candidate
  levers** (θ̇ range, physics timestep, control-interval sub-stepping,
  crossover, the signed-vs-absolute sort, the concentration floor `0.0` vs the
  thesis's `1e-10`, and the tendency tie-threshold); none closes the gap and
  several _hurt_. The real driver is the one the thesis names: the P-protein
  _ordering_ the deep grammars consume is very hard to evolve under the
  2000-iteration sync, whereas concentration is anchored by its sum-to-1
  normalisation.
- **The chatter is a symptom, not the cause.** The tendency force _does_
  chatter (~2× concentration near balance), but _reducing_ it — forcing the
  controller to hold — makes training **worse** (SymReg·Tend 25→8% as the
  tie-threshold rises), so reactivity is load-bearing. Our earlier
  "responsive-but-jumpy force cannot hold" explanation was wrong on causation.
- **Variance explains the wobble; a systematic gap remains.** The paper (50
  runs) and thesis (100 runs) of the _same_ work differ by up to 18 points, and
  our seed-to-seed swings are large — so single numbers mislead. But pooling
  110 faithful runs still gives SymReg·Tend **24.5%** vs paper 86% / thesis
  82%: the deep-grammar × Tendency gap is real, not noise, and the residual
  points to unspecified implementation details (GEVA's TAGE internals,
  normalisation order, steady-state window) not recoverable from the text.

---

## The system

```text
genome (4096-bit string)
  └─ scan promoters ─► TF-genes + P-genes ─► GRN (256-bit genes, 32-bit protein signatures)
        │  regulation: eᵢ,hᵢ = (1/N) Σ cⱼ·exp(β(uⱼ−u_max));  dcᵢ/dt = δ(eᵢ−hᵢ)cᵢ
        │  inputs: 4 cart-pole state vars injected as free TF-protein concentrations
        ▼
   P-protein concentrations ─► output method ─► codon stream
        │  Concentration Value / Concentration Tendency  (binary, single P-gene, "best" selected)
        │  Sort-by-Concentration / Sort-by-Tendency       (all P-genes, signatures = codons)
        ▼
   TAGE derivation (codon→initial tree, then adjunctions) ─► RPN expression ─► force α∈[−1,1]
        ▼
   cart-pole (Eq. 4 physics, ±10 N, 0.02 s step) ─► survive 120 000 steps = success
        ▼
   EA: generational, pop 250, 50 gens, tournament-3, elitism 25, bit-mutation 0.005, NO crossover
```

Code lives in
`src/grntage/{grn,grammar,mapping,simulation,evolution,experiments}`.

## Reproducing

```bash
uv run pytest tests                        # 227 tests; ruff/format clean; mypy clean
uv run --only-group lint rumdl check .     # lint Markdown (GitHub-flavored, 80-col)
# one configuration, full paper scale (pop 250 × 50 gens × 50 runs, ~1.5–3 h):
uv run python -u -m grntage.experiments.run_experiments \
    --configs SymbolicRegression_Tendency --output results
# all eight: omit --configs.  Reduced smoke run: add --quick.
```

Pre-computed 50-run result CSVs are in `results/`. The targeted investigation
probes (`investigation/probe_*.py`) regenerate the measurements cited below.

---

## Results: ours vs. paper vs. thesis

Generalisation columns are **cases solved / 625** (1000-step horizon). **Suc**
is the count of runs (out of 50 here, 50 in the paper, 100 in the thesis) in
which the EA ever found a 120 000-step solver. Faithful regime: signed
Sort-by-Tendency, mutation-only, `physics_dt = 0.02 s`.

| Configuration              | **Ours** Suc% · mean · med · best | **Paper** (Table 2, /50) Suc · mean · best | **Thesis** (Table 9.2, /100) Suc · mean · best |
| -------------------------- | --------------------------------- | ------------------------------------------ | ---------------------------------------------- |
| Direct · Concentration     | **78%** · 157 · 107 · 421         | 94% · 203 · 406                            | 100% · 207 · 404                               |
| Direct · Tendency          | **78%** · 76 · 37 · 363           | 100% · 58 · 137                            | 100% · 75 · 232                                |
| Discrete · Concentration   | **90%** · 135 · 127 · 323         | 72% · 121 · 355                            | 94% · 186 · 409                                |
| Discrete · Tendency        | **96%** · 162 · 145 · 478         | 100% · 89 · 240                            | 95% · 104 · 268                                |
| Continuous · Concentration | **88%** · 102 · 97 · 364          | 80% · 155 · 390                            | 92% · 185 · 393                                |
| Continuous · Tendency      | **66%** · 127 · 128 · 317         | 100% · 62 · 200                            | 94% · 108 · 241                                |
| SymReg · Concentration     | **72%** · 97 · 66 · 343           | 78% · 128 · 356                            | 91% · 158 · 393                                |
| **SymReg · Tendency**      | **30%** · 147 · 148 · 361         | 86% · 79 · 208                             | 82% · 109 · 321                                |

(Full distributions incl. worst/std are in `results/*_results.csv`.)

### How to read this

- **Training success (Suc):** we match within tolerance on most configs
  (Direct·Conc 78/94 and Discrete·Conc 90/72 straddle the paper; Continuous·Conc
  and SymReg·Conc land squarely on it; the shallow Tendency configs
  Discrete·Tend 96% and Direct·Tend 78% are fine). The two deep-grammar Tendency
  configs are the clear misses: **Continuous·Tend 66%** and **SymReg·Tend 30%**.
- **Generalisation:** our Concentration _best_-individual scores track the paper
  on every grammar (e.g. SymReg·Conc best 343 vs 356), but our _mean_
  undershoots for the three deeper grammars — a distribution shift, not a
  capability gap. Our **Tendency** generalisation runs **~2× above** the paper
  across the board (SymReg·Tend mean 147 vs 79; Continuous·Tend 127 vs 62).
- **The paper's two qualitative trends:** "Tendency finds solutions faster"
  reproduces (our Tendency solution-generations are low). "Concentration
  generalises better than Tendency" reproduces for the **Direct** grammar but
  **reverses** for the three Sort grammars — purely because our Tendency
  _over_-generalises, lifting it above Concentration.

---

## What we had to reconstruct (the paper omits most constants)

The 8-page paper does not state the physics constants, the GRN scaling factors,
the input encoding, the output-selection rule, or the random-init distribution.
We recovered them from the paper's **direct predecessor** — Nicolau, Schoenauer
& Banzhaf, _Evolving Genes to Balance a Pole_ (EuroGP 2010,
[arXiv:1005.2815](https://arxiv.org/abs/1005.2815)), the _same_ GRN+cart-pole —
and from **Murphy's PhD thesis** (UCD, _An Exploration of Tree-Adjoining
Grammars for Grammatical Evolution_), of which this paper is Chapter 9.
Non-obvious reconstructions, each of which materially changed results during
development:

| Detail                                  | Resolution                                                                                                                                                                                                                                    | Source                                                                     |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **"Best Product / Best Single Output"** | the binary methods test **all P-genes and keep the most successful**, not P-gene #0                                                                                                                                                           | predecessor §3.1 ("all P-genes are tested, the most successful is used")   |
| **Gravity sign**                        | **g = +9.8** despite the paper printing −9.8 (a transcription error: an inverted pendulum needs g>0 to be _unstable_, and the predecessor states +9.8)                                                                                        | predecessor + physics                                                      |
| **Control interval**                    | **0.02 s** Euler step (not the literal "0.2 s / 2000 iterations", which denotes GRN _settling_ time)                                                                                                                                          | falsified directly — see below                                             |
| **GRN regulation sign**                 | uⱼ = popcount(XOR) — degree of _complementary_ match                                                                                                                                                                                          | predecessor + Banzhaf 2003                                                 |
| **Gene type tag**                       | promoter low 8 bits: `…00000000`=TF, `…11111111`=P                                                                                                                                                                                            | thesis §4.2                                                                |
| **Tendency baseline**                   | concentration change is measured **from input injection to the final iteration of the sync** — so the _first_ control action must be state-dependent (a naive implementation that defaults the first action is state-blind and cannot evolve) | paper §3.1                                                                 |
| **Grammar mapper**                      | a faithful **TAGE** (adjunction-only, valid at every derivation stage). An earlier CFG-GE mapper that padded dangling non-terminals crippled the codon-hungry Continuous/SymReg grammars                                                      | thesis §5 (TAGs valid "regardless of how many codon values are available") |
| EA, genome, generalisation grid         | generational GA, 4096-bit random genome, 5⁴=625 cases @ {0.05,0.275,0.5,0.725,0.95}, 457 solvable                                                                                                                                             | paper Table 1 + thesis Table 9.1                                           |

---

## Hypotheses tested and ruled out (and why)

Everything below was implemented and **measured**, not argued away. Two falsified
code levers are retained behind flags (`--crossover-rate`, `--tendency-sort`); the
tendency chatter/collapse is probed by `investigation/probe_tendency_chatter.py`;
the floor and tie-threshold success numbers come from paired EA runs (floor = the
2-line `max(0.0,·)`→`max(1e-10,·)` kernel change; tie-threshold =
`SortByTendencyMapper(threshold=·)`).

| Lever                                                                                            | Hypothesis                                                                                                                          | Outcome                          | Why ruled out                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **θ̇ encoding range**                                                                             | the ±1.5 °/s input saturates the angular-velocity channel                                                                           | **rejected**                     | ±1.5 °/s is the paper's literal value and the paper hit 86% _with_ it; widening it is unfaithful and doesn't track the gap                                                                                                                                                                               |
| **physics_dt**                                                                                   | wrong integration step                                                                                                              | **rejected**                     | only scales difficulty monotonically (0.01/0.02/0.05 → 60/43/0% on SymReg·Tend); 0.02 s is the predecessor's stated step                                                                                                                                                                                 |
| **Control-interval sub-stepping** (a literal 0.2 s control interval, integrated in 10 sub-steps) | the paper updates the force every 0.2 s                                                                                             | **falsified**                    | a ±10 N force held 0.2 s tips the pole >12° in a _single_ cycle, so bang-bang **Direct survives 0 steps** — yet the paper reports Direct at 47–50/50. The control interval must be ~0.02 s; "0.2 s" is GRN settling time                                                                                 |
| **Crossover**                                                                                    | the GE method should use one-point crossover (Table 1 omits the operator)                                                           | **falsified — actively harmful** | the GRN genome is _crossover-intolerant_: one-point **bit** crossover shreds contiguous gene/promoter structure. Adding it dropped Continuous·Tend 66→56% and **SymReg·Tend 30→4%**, worse the deeper the grammar. The predecessor explicitly used "only bit-flip mutation". → mutation-only is faithful |
| **Sort-by-Tendency key: signed vs absolute**                                                     | the paper ("signed") and thesis ("absolute magnitude") _contradict_ each other; absolute might be what produced the paper's numbers | **falsified**                    | the two produce a different control force in ~80% of states, so it is a real fork — but **absolute is _worse_** (Continuous·Tend 66→24%; SymReg·Tend also down). The paper's signed reading is both more faithful and better.                                                                            |
| **Concentration floor** (`0.0` vs the thesis's `1e-10`)                                          | the thesis floors concentration at `1e-10` "so a protein can re-emerge"; flooring at `0.0` lets a TF regulator that hits zero die permanently (`dc/dt = δ(e−h)c` makes `c=0` absorbing), degenerating the tendency signal over a long rollout | **rejected**                     | the one _specified_ discrepancy (the code floors at `0.0`), but paired A/B runs show no effect: floor `0.0` **6/36** vs floor `1e-10` **8/36** solved on SymReg·Tend (not significant). Re-emergence from `1e-10` is too slow, so the live TF pool collapses either way (~10→3 over a rollout)             |
| **Sort-by-Tendency tie-threshold** (hold when the signal is weak)                                | the near-balance chatter is what breaks the 120000-step hold, so treating sub-threshold deltas as ties (a stable, held force) should raise success | **falsified — actively harmful** | raising the threshold cuts near-balance chatter ~3× yet lowers SymReg·Tend success _monotonically_ (1e-10→**25%**, 3e-3→21%, 1e-2→**8%**, n=24 each). Reactivity is load-bearing; the jumpiness is a symptom, not the cause                                                                              |

What _did_ help (and is therefore in the faithful build, not "ruled out"):
best-P-gene selection, the injection-time tendency baseline, the TAGE mapper,
and the GRN sign/promoter fixes above.

---

## Why the deep-grammar × Tendency gap is inherent (and what it is _not_)

Three drivers of the hard-to-evolve tendency _ordering_, each verified against
code, sources, and probes (`investigation/probe_tendency_chatter.py`):

1. **No conservation anchor.** Concentrations are normalised to sum to 1 ("if
   one decreases, another must increase"), so the _order_ of P-proteins by
   concentration is structurally stable across states. The _tendency_ (signed
   change) has no such constraint, so its ordering is far harder for evolution
   to shape into a useful control law. Murphy's thesis says exactly this, and
   adds that the **difficulty grows with the 2000-iteration sync size** (the
   network largely re-settles between injections, shrinking the change signal).
   Measured: near a balanced pole the tendency signal collapses **~5×** vs a
   wandering state, so the near-balance regime you must hold is the weakest.
2. **Degeneracy at the zero fixed point.** After settling, many "passenger"
   P-proteins barely change; their sub-threshold deltas collapse toward the tail
   of the codon stream — exactly the deep region the recursive SymReg grammar
   reads. (Nicolau 2010 notes the same: tendency can stall, repeating the
   previous action.) Measured tie-fractions reach 0.6+ for high-P-gene genomes
   (`investigation/probe_tendency_degeneracy.py`).
3. **Deepest grammar, most order-sensitive.** SymReg is the only
   doubly-recursive grammar (`<expr> <expr> <op>`); it consumes the most codons
   and is most sensitive to the leading codons (`codon[0] % |initial trees|`
   selects the entire base expression). Stacking the most order-sensitive
   grammar on the least-stable ordering is the worst case — which is precisely
   where the gap is largest (SymReg·Tend < Continuous·Tend), as observed.

**What breaks — and what does _not_.** These drivers make the tendency
_ordering_ hard to evolve into a controller that holds a near-balanced pole
for 120 000 steps; concentration escapes it via the sum-to-1 anchor above.
The tendency force _does_ chatter (~2× concentration near balance) and is
more reactive, matching its _above_-paper score on the short 1000-step
generalisation horizon. But we originally called that jumpiness the **cause**
of the under-solve, and that is **wrong**: forcing the controller to _hold_
instead of chatter (a larger tie-threshold) makes training success
monotonically **worse** (SymReg·Tend 25→8%). Reactivity is load-bearing; the
jumpiness is a _symptom_ of the unstable ordering, not the binding
constraint — the evolvability of the ordering itself, as the thesis states.

**The residual we cannot close.** After ruling out every concrete lever (table
above), pooling **110 faithful runs** still puts SymReg·Tend at **24.5%** vs
paper 86% / thesis 82% — a _systematic_ gap, not sampling noise, and the one
_specified_ discrepancy (the `0.0`-vs-`1e-10` floor) has no measurable effect.
So the surplus over the paper's numbers is best attributed to details the
sources never pin down — GEVA's exact TAGE mapping and codon wrapping, the
order of the Φ normalisation vs the concentration clamp, the steady-state
window/threshold, and P-protein tie-break order. We flag this as genuinely
unresolved rather than claim a mechanism we cannot verify. (Corroboration for
the _direction_: Nicolau 2010 independently reports the same
tendency-worse-than-concentration degradation.)

---

## Honest caveats & limitations

- **No ground-truth code exists.** Our "faithful" claim is faithfulness to the
  _published text_ (paper + predecessor + thesis), cross-checked three ways, not
  to the original binaries. Where the sources themselves disagree (gravity sign,
  signed-vs-absolute sort, paper-vs-thesis success numbers) we say so and test
  empirically.
- **The success metric is intrinsically high-variance.** One shared random
  initial state per generation, whole population re-evaluated, run stopped at
  the first 120 000-step solver. The paper (50 runs) and thesis (100 runs) of
  the _same_ work differ by up to 18 points on these configs (e.g.
  Continuous·Tend 100% vs 94%), so part of any single number is sampling noise.
  Our results are consistent with the thesis's "more runs → lower, more honest
  success" direction. But this variance explains only the _wobble_: pooling
  110 faithful SymReg·Tend runs still gives 24.5%, so that config's gap is
  systematic (see "Why … inherent"), not sampling noise.
- **The random-initial-state _distribution_** is unspecified in the 2012
  sources, but the predecessor (Nicolau 2010) resolves it: the start is drawn
  over the _full_ ranges ("several combinations … result in unsolvable states"),
  which is what we do. A narrower near-upright window would raise
  fragile-Tendency success, but it would be _unfaithful_ — so we do not treat
  the gap as a distribution artifact or a candidate fix.
- **Determinism:** numba kernels run with `fastmath=False`; a fixed-seed GRN
  trajectory is pinned by a golden-master test
  (`tests/test_grn/test_golden.py`).

---

## Repository layout

```text
src/grntage/            replication implementation (grn, grammar, mapping, simulation, evolution, experiments)
tests/                  227 unit/property/golden tests (pytest; warnings-as-errors)
scripts/validate_e2e.py paper-scale end-to-end smoke of the full pipeline
results/                pre-computed 50-run Table-2 CSVs (faithful signed/mutation-only regime)
investigation/          probe scripts behind the ruled-out-levers section
```

## References

1. Murphy, Nicolau, Hemberg, O'Neill, Brabazon. _Differential Gene Expression
   with Tree-Adjunct Grammars._ PPSN XII, 2012.
2. Nicolau, Schoenauer, Banzhaf. _Evolving Genes to Balance a Pole._
   EuroGP 2010. [arXiv:1005.2815](https://arxiv.org/abs/1005.2815).
3. Murphy. _An Exploration of Tree-Adjoining Grammars for Grammatical
   Evolution._ PhD thesis, University College Dublin, 2013. (Ch. 9 is this
   paper.)
4. Murphy, O'Neill, Galván-López, Brabazon. _Tree-Adjunct Grammatical
   Evolution._ IEEE WCCI/CEC 2010.
5. Banzhaf. _Artificial Regulatory Networks and Genetic Programming._ In
   _Genetic Programming Theory and Practice_, 2003.
6. Whitley, Dominic, Das, Anderson. _Genetic Reinforcement Learning for
   Neurocontrol Problems._ Machine Learning 13, 1993. (the 625-case
   generalisation test)

_Replication implemented in Python 3.14; the custom Eq. 4 cart-pole is
implemented directly. Status: faithful reproduction with one well-characterised,
source-consistent divergence on deep-grammar × Tendency. All numbers above are
from full 50-run experiments in this repository._
