"""Reverse Polish Notation (RPN) expression evaluator.

Evaluates expressions produced by the grammar system.
All grammars produce RPN expressions for consistent evaluation.
"""

from typing import Callable


class RPNEvaluator:
    """Evaluates Reverse Polish Notation expressions.

    Supports basic arithmetic operations: +, -, *, /
    Results are clamped to [-1.0, 1.0] for control output.
    """

    # Operator functions
    OPERATORS: dict[str, Callable[[float, float], float]] = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b != 0 else 0.0,  # Safe division
    }

    def __init__(self, clamp_output: bool = True) -> None:
        """Initialize the evaluator.

        Args:
            clamp_output: Whether to clamp results to [-1.0, 1.0]
        """
        self.clamp_output = clamp_output

    def evaluate(self, tokens: list[str]) -> float:
        """Evaluate an RPN expression.

        Args:
            tokens: List of tokens (numbers and operators)

        Returns:
            Evaluated result, optionally clamped to [-1.0, 1.0]

        Raises:
            ValueError: If expression is invalid
        """
        if not tokens:
            return 0.0

        stack: list[float] = []

        for token in tokens:
            if token in self.OPERATORS:
                if len(stack) < 2:
                    # Not enough operands, return what we have
                    if stack:
                        result = stack[-1]
                    else:
                        result = 0.0
                    return self._clamp(result)

                b = stack.pop()
                a = stack.pop()
                result = self.OPERATORS[token](a, b)
                stack.append(result)
            else:
                # Try to parse as number
                try:
                    value = float(token)
                    stack.append(value)
                except ValueError:
                    # Skip invalid tokens
                    continue

        if not stack:
            return 0.0

        result = stack[-1]
        return self._clamp(result)

    def _clamp(self, value: float) -> float:
        """Clamp value to [-1.0, 1.0] if enabled.

        Args:
            value: Value to clamp

        Returns:
            Clamped value
        """
        if not self.clamp_output:
            return value
        return max(-1.0, min(1.0, value))

    def evaluate_string(self, expression: str) -> float:
        """Evaluate an RPN expression from a string.

        Args:
            expression: Space-separated RPN expression

        Returns:
            Evaluated result
        """
        tokens = expression.split()
        return self.evaluate(tokens)


# Default evaluator instance
default_evaluator = RPNEvaluator(clamp_output=True)


def evaluate_rpn(tokens: list[str]) -> float:
    """Convenience function to evaluate RPN expression.

    Args:
        tokens: List of tokens

    Returns:
        Evaluated result clamped to [-1.0, 1.0]
    """
    return default_evaluator.evaluate(tokens)
