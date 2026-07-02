"""Grammar mapper for codon-to-phenotype mapping.

Maps integer codons to phenotype expressions using Tree-Adjunct Grammatical
Evolution (TAGE): the context-free grammar is transformed into a tree-adjoining
grammar and codons drive initial-tree selection + adjunction (see
:mod:`grntage.grammar.tage`). The decisive property the paper relies on is
that a valid phenotype exists at every stage of derivation, so any codon string
-- of any length, as produced by the GRN -- maps to a runnable expression.
"""

from grntage.grammar.definitions import Grammar
from grntage.grammar.evaluator import RPNEvaluator
from grntage.grammar.tage import build_tag


class GrammarMapper:
    """Maps integer codons to phenotype expressions using a TAGE-derived TAG.

    Attributes:
        grammar: The context-free grammar (transformed to a TAG internally)
        evaluator: RPN evaluator for expression evaluation
        max_depth: Retained for API compatibility; TAGE derivation is bounded by
            the codon count, not a recursion depth.
    """

    def __init__(
        self,
        grammar: Grammar,
        evaluator: RPNEvaluator | None = None,
        max_depth: int = 50,
    ) -> None:
        """Initialize the mapper.

        Args:
            grammar: Grammar to use for derivation
            evaluator: RPN evaluator (creates default if None)
            max_depth: Unused by TAGE; kept for API compatibility
        """
        self.grammar = grammar
        self.evaluator = evaluator or RPNEvaluator(clamp_output=True)
        self.max_depth = max_depth
        self._tag = build_tag(grammar)

    def map_codons(self, codons: list[int]) -> list[str]:
        """Map codons to a phenotype expression (list of terminal tokens).

        Args:
            codons: List of integer codons

        Returns:
            List of terminal tokens forming the RPN expression
        """
        tokens = self._tag.derive(codons)
        return self._merge_number_tokens(tokens)

    @staticmethod
    def _merge_number_tokens(tokens: list[str]) -> list[str]:
        """Concatenate a '0.' prefix and following digit tokens into one number.

        The continuous and symbolic-regression grammars build a constant as the
        terminal '0.' followed by single-digit terminals; merge them into a single
        numeric token (e.g. ['0.', '5', '6'] -> ['0.56']) so the RPN evaluator
        reads one number rather than several.
        """
        merged: list[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            if token == "0.":
                number = "0."
                i += 1
                while i < n and tokens[i].isdigit():
                    number += tokens[i]
                    i += 1
                merged.append(number if number != "0." else "0.0")
                continue
            merged.append(token)
            i += 1
        return merged

    def map_and_evaluate(self, codons: list[int]) -> float:
        """Map codons to phenotype and evaluate.

        Args:
            codons: List of integer codons

        Returns:
            Evaluated phenotype value (clamped to [-1.0, 1.0])
        """
        tokens = self.map_codons(codons)
        return self.evaluator.evaluate(tokens)

    def phenotype_string(self, codons: list[int]) -> str:
        """Get phenotype as a string.

        Args:
            codons: List of integer codons

        Returns:
            Space-separated token string
        """
        tokens = self.map_codons(codons)
        return " ".join(tokens)
