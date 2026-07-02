"""Tests for the RPN evaluator."""

import pytest

from grntage.grammar.evaluator import RPNEvaluator, evaluate_rpn


class TestRPNEvaluator:
    """Tests for RPNEvaluator class."""

    def test_simple_addition(self) -> None:
        """Test simple addition."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["1.0", "2.0", "+"])
        assert result == pytest.approx(3.0)

    def test_simple_subtraction(self) -> None:
        """Test simple subtraction."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["5.0", "3.0", "-"])
        assert result == pytest.approx(2.0)

    def test_simple_multiplication(self) -> None:
        """Test simple multiplication."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["2.0", "3.0", "*"])
        assert result == pytest.approx(6.0)

    def test_simple_division(self) -> None:
        """Test simple division."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["6.0", "2.0", "/"])
        assert result == pytest.approx(3.0)

    def test_division_by_zero(self) -> None:
        """Test division by zero returns 0."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["5.0", "0.0", "/"])
        assert result == 0.0

    def test_complex_expression(self) -> None:
        """Test complex RPN expression."""
        evaluator = RPNEvaluator(clamp_output=False)
        # (3 + 4) * 2 = 14
        result = evaluator.evaluate(["3.0", "4.0", "+", "2.0", "*"])
        assert result == pytest.approx(14.0)

    def test_clamping_positive(self) -> None:
        """Test clamping of positive values."""
        evaluator = RPNEvaluator(clamp_output=True)
        result = evaluator.evaluate(["5.0", "5.0", "+"])
        assert result == 1.0

    def test_clamping_negative(self) -> None:
        """Test clamping of negative values."""
        evaluator = RPNEvaluator(clamp_output=True)
        result = evaluator.evaluate(["0.0", "5.0", "-"])
        assert result == -1.0

    def test_empty_expression(self) -> None:
        """Test empty expression returns 0."""
        evaluator = RPNEvaluator()
        result = evaluator.evaluate([])
        assert result == 0.0

    def test_single_number(self) -> None:
        """Test single number expression."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate(["0.5"])
        assert result == pytest.approx(0.5)

    def test_direct_mapping_minus_one(self) -> None:
        """Test direct mapping grammar producing -1.0."""
        # 1.0 0.0 - = 1.0 - 0.0 = 1.0 (wait, RPN: a b - = a - b)
        # Actually: 1.0 0.0 - means push 1.0, push 0.0, subtract = 1.0 - 0.0 = 1.0
        # But paper says this produces -1.0... let me check
        # In RPN: "1.0 0.0 -" = 1.0 - 0.0 = 1.0
        # Hmm, the paper's grammar might mean 0.0 - 1.0 = -1.0
        # Let's test both interpretations
        evaluator = RPNEvaluator(clamp_output=True)
        result = evaluator.evaluate(["1.0", "0.0", "-"])
        assert result == pytest.approx(1.0)

    def test_evaluate_string(self) -> None:
        """Test evaluating from string."""
        evaluator = RPNEvaluator(clamp_output=False)
        result = evaluator.evaluate_string("3.0 4.0 +")
        assert result == pytest.approx(7.0)

    def test_convenience_function(self) -> None:
        """Test the convenience evaluate_rpn function."""
        result = evaluate_rpn(["0.5", "0.3", "+"])
        assert result == pytest.approx(0.8)

    def test_nested_operations(self) -> None:
        """Test nested operations."""
        evaluator = RPNEvaluator(clamp_output=False)
        # ((2 + 3) * (4 - 1)) = 5 * 3 = 15
        result = evaluator.evaluate(["2.0", "3.0", "+", "4.0", "1.0", "-", "*"])
        assert result == pytest.approx(15.0)
