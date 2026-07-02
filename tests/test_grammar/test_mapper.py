"""Tests for the grammar mapper."""

import pytest

from grntage.grammar.definitions import (
    CONTINUOUS_DIGITS_GRAMMAR,
    DIRECT_MAPPING_GRAMMAR,
    DISCRETE_DIGITS_GRAMMAR,
    SYMBOLIC_REGRESSION_GRAMMAR,
)
from grntage.grammar.mapper import GrammarMapper


class TestGrammarMapper:
    """Tests for GrammarMapper class."""

    def test_direct_mapping_first_choice(self) -> None:
        """Test direct mapping with first production choice (negative)."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        # Codon 0 selects first production: 0.0 1.0 - = -1.0
        tokens = mapper.map_codons([0])
        assert tokens == ["0.0", "1.0", "-"]

    def test_direct_mapping_second_choice(self) -> None:
        """Test direct mapping with second production choice (positive)."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        # Codon 1 selects second production: 1.0
        tokens = mapper.map_codons([1])
        assert tokens == ["1.0"]

    def test_direct_mapping_evaluate(self) -> None:
        """Test direct mapping evaluation produces -1.0 and +1.0."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        # First choice: 0.0 1.0 - = 0.0 - 1.0 = -1.0
        result = mapper.map_and_evaluate([0])
        assert result == pytest.approx(-1.0)

        # Second choice: 1.0
        result = mapper.map_and_evaluate([1])
        assert result == pytest.approx(1.0)

    def test_discrete_digits_one_codon_selects_initial_tree(self) -> None:
        """Discrete (TAGE): one codon selects an initial tree '0.0 <const> <op>'.

        Under TAGE the non-recursive discrete grammar has no auxiliary trees, so
        the whole phenotype is a single initial tree chosen by codon % 22. The
        operand order '0.0 <const> <op>' makes '+' yield +const and '-' yield
        0.0 - const = -const (faithful polarity).
        """
        mapper = GrammarMapper(DISCRETE_DIGITS_GRAMMAR)
        assert mapper.map_codons([10]) == ["0.0", "0.5", "+"]
        assert mapper.map_and_evaluate([10]) == pytest.approx(0.5)
        assert mapper.map_codons([11]) == ["0.0", "0.5", "-"]
        assert mapper.map_and_evaluate([11]) == pytest.approx(-0.5)

    def test_discrete_digits_range(self) -> None:
        """Discrete's 22 initial trees reach the full [-1.0, 1.0] grid (step 0.1)."""
        mapper = GrammarMapper(DISCRETE_DIGITS_GRAMMAR)
        outputs = set()
        for codon in range(22):  # all initial trees (11 consts x 2 ops)
            value = mapper.map_and_evaluate([codon])
            assert -1.0 <= value <= 1.0
            toks = mapper.map_codons([codon])
            assert len(toks) == 3 and toks[0] == "0.0" and toks[2] in {"+", "-"}
            outputs.add(round(value, 1))
        assert outputs == {round(-1.0 + 0.1 * i, 1) for i in range(21)}

    def test_continuous_single_digit_initial_tree(self) -> None:
        """Continuous (TAGE): one codon gives a single-digit constant '0.0 0.d <op>'."""
        mapper = GrammarMapper(CONTINUOUS_DIGITS_GRAMMAR)
        assert mapper.map_codons([10]) == ["0.0", "0.5", "+"]
        assert mapper.map_and_evaluate([10]) == pytest.approx(0.5)
        for codon in range(20):  # all initial trees (10 digits x 2 ops)
            toks = mapper.map_codons([codon])
            assert len(toks) == 3 and toks[0] == "0.0" and toks[2] in {"+", "-"}
            assert len(toks[1]) == 3  # '0.' + exactly one digit

    def test_continuous_adjunction_grows_constant(self) -> None:
        """Adjoining <digits> auxiliary trees prepends digits -> a longer constant.

        Each adjunction (two codons) adds one digit to the constant, so the
        numeric token lengthens by one digit per adjunction (e.g. 0.3 -> 0.31 ->
        0.731) while the expression stays a valid '0.0 0.ddd <op>'.
        """
        mapper = GrammarMapper(CONTINUOUS_DIGITS_GRAMMAR)

        def const_digits(codons: list[int]) -> int:
            return len(mapper.map_codons(codons)[1]) - 2  # digits after '0.'

        assert const_digits([2]) == 1  # initial tree only
        assert const_digits([2, 0, 3]) == 2  # one adjunction
        assert const_digits([2, 0, 3, 0, 7]) == 3  # two adjunctions

    def test_symbolic_regression_initial_tree(self) -> None:
        """SymReg (TAGE): one codon selects a complete 'a b <op>' RPN expression."""
        mapper = GrammarMapper(SYMBOLIC_REGRESSION_GRAMMAR)
        for codon in [0, 5, 100, 399]:  # spans the 400 initial trees
            toks = mapper.map_codons([codon])
            assert len(toks) == 3 and toks[2] in {"+", "-", "*", "/"}
            assert -1.0 <= mapper.map_and_evaluate([codon]) <= 1.0

    def test_symbolic_regression_adjunction_grows_expression(self) -> None:
        """Each SymReg adjunction (two codons) adds one operand + one operator.

        The phenotype stays a complete, valid RPN expression at every stage
        (the always-valid TAGE property), growing by two tokens per adjunction.
        """
        mapper = GrammarMapper(SYMBOLIC_REGRESSION_GRAMMAR)
        lengths = [len(mapper.map_codons([3] + [1, 2] * k)) for k in range(4)]
        assert lengths == [3, 5, 7, 9]
        for k in range(6):
            assert -1.0 <= mapper.map_and_evaluate([3] + [1, 2] * k) <= 1.0

    def test_symbolic_regression_clamping(self) -> None:
        """Test symbolic regression results are clamped."""
        mapper = GrammarMapper(SYMBOLIC_REGRESSION_GRAMMAR)
        # Any result should be clamped to [-1.0, 1.0]
        for i in range(20):
            result = mapper.map_and_evaluate([i, i + 1, i + 2, i + 3])
            assert -1.0 <= result <= 1.0

    def test_tage_always_valid_any_length(self) -> None:
        """TAGE's defining property: any codon string maps to a valid phenotype.

        Empty, single, odd-length, huge-valued and long codon strings must all
        map to a non-empty token list that evaluates within [-1.0, 1.0] -- the
        property the GRN front-end relies on (it emits a variable codon count).
        """
        codon_sets = [[], [0], [7], [999999], [1, 2], [3, 1, 2], list(range(50))]
        for grammar in (
            DIRECT_MAPPING_GRAMMAR,
            DISCRETE_DIGITS_GRAMMAR,
            CONTINUOUS_DIGITS_GRAMMAR,
            SYMBOLIC_REGRESSION_GRAMMAR,
        ):
            mapper = GrammarMapper(grammar)
            for codons in codon_sets:
                tokens = mapper.map_codons(codons)
                assert len(tokens) >= 1
                assert -1.0 <= mapper.map_and_evaluate(codons) <= 1.0

    def test_empty_codons(self) -> None:
        """Test mapping with empty codons produces valid output."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        tokens = mapper.map_codons([])
        # Should produce minimal valid phenotype
        assert len(tokens) > 0

    def test_phenotype_string(self) -> None:
        """Test phenotype string generation."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        phenotype = mapper.phenotype_string([0])
        assert isinstance(phenotype, str)
        assert len(phenotype) > 0

    def test_long_recursive_codon_stream_terminates(self) -> None:
        """A long codon stream over a recursive grammar terminates with a valid
        phenotype (TAGE derivation is bounded by the codon count, not depth).

        100 codons drive ~49 <digits> adjunctions, building a long but valid
        constant -- it must complete (not hang) and stay evaluable in [-1, 1].
        """
        mapper = GrammarMapper(CONTINUOUS_DIGITS_GRAMMAR)
        codons = [0] * 100
        tokens = mapper.map_codons(codons)
        assert len(tokens) > 0
        assert -1.0 <= mapper.map_and_evaluate(codons) <= 1.0

    def test_codon_modulo_wrapping(self) -> None:
        """Test that large codons wrap correctly."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        # Large codon should wrap: 100 % 2 = 0
        tokens1 = mapper.map_codons([100])
        tokens2 = mapper.map_codons([0])
        assert tokens1 == tokens2

        # 101 % 2 = 1
        tokens3 = mapper.map_codons([101])
        tokens4 = mapper.map_codons([1])
        assert tokens3 == tokens4


class TestEndToEndForce:
    """Pin the full codon -> force (alpha) mapping, incl. interacting polarities."""

    def test_direct_grammar_force_signs(self) -> None:
        """Direct grammar: codon 0 -> -1.0 (bang-bang reverse), codon 1 -> +1.0."""
        mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        assert mapper.map_and_evaluate([0]) == pytest.approx(-1.0)
        assert mapper.map_and_evaluate([1]) == pytest.approx(1.0)

    def test_tendency_then_direct_grammar(self) -> None:
        """End-to-end: a rising P-protein -> tendency codon 0 -> direct -1.0.

        Pins the interaction of the paper's tendency sign (positive change -> 0)
        with the direct grammar's production order (codon 0 -> -1.0). If either the
        tendency sign or the grammar operand order were flipped, the control
        polarity here would be wrong.
        """
        from grntage.grn.protein import Protein, ProteinType
        from grntage.mapping import ConcentrationTendencyMapper

        grammar_mapper = GrammarMapper(DIRECT_MAPPING_GRAMMAR)
        tendency = ConcentrationTendencyMapper(protein_index=0)
        protein = Protein(0xABCDEF00, ProteinType.P)

        protein.concentration = 0.4  # rose from 0.1 -> positive change
        rising = tendency.map_to_codons([protein], {protein.signature: 0.1})
        assert rising == [0]
        assert grammar_mapper.map_and_evaluate(rising) == pytest.approx(-1.0)

        tendency.reset()
        protein.concentration = 0.1  # fell from 0.4 -> negative change
        falling = tendency.map_to_codons([protein], {protein.signature: 0.4})
        assert falling == [1]
        assert grammar_mapper.map_and_evaluate(falling) == pytest.approx(1.0)
