"""Grammar definitions for the TAGE system.

Four grammars are defined as per the paper:
1. Direct mapping (binary): Produces -1.0 or 1.0
2. Discrete digits: Values in [-1.0, 1.0] with 0.1 resolution
3. Continuous digits: Values in (-0.9, 0.9) with arbitrary precision
4. Symbolic regression: Arbitrary expressions, clamped to [-1.0, 1.0]

All grammars use Reverse Polish Notation (RPN) for expression representation.
"""

from enum import Enum, auto
from typing import NamedTuple


class GrammarType(Enum):
    """Types of grammars available."""

    DIRECT_MAPPING = auto()
    DISCRETE_DIGITS = auto()
    CONTINUOUS_DIGITS = auto()
    SYMBOLIC_REGRESSION = auto()


class Production(NamedTuple):
    """A grammar production rule."""

    lhs: str  # Left-hand side (non-terminal)
    rhs: tuple[str, ...]  # Right-hand side (sequence of symbols)


class Grammar:
    """A context-free grammar for phenotype generation.

    Attributes:
        name: Grammar identifier
        grammar_type: Type of grammar
        start_symbol: Starting non-terminal
        productions: Dict mapping non-terminals to list of productions
        terminals: Set of terminal symbols
    """

    def __init__(
        self,
        name: str,
        grammar_type: GrammarType,
        start_symbol: str,
        productions: dict[str, list[tuple[str, ...]]],
    ) -> None:
        """Initialize a grammar.

        Args:
            name: Grammar name
            grammar_type: Type of grammar
            start_symbol: Starting non-terminal symbol
            productions: Dict mapping non-terminals to list of RHS alternatives
        """
        self.name = name
        self.grammar_type = grammar_type
        self.start_symbol = start_symbol
        self.productions = productions

        # Extract terminals (symbols that don't appear as LHS)
        all_symbols: set[str] = set()
        for rhs_list in productions.values():
            for rhs in rhs_list:
                all_symbols.update(rhs)
        self.terminals = all_symbols - set(productions.keys())

    def get_productions(self, non_terminal: str) -> list[tuple[str, ...]]:
        """Get all productions for a non-terminal.

        Args:
            non_terminal: The non-terminal symbol

        Returns:
            List of RHS alternatives
        """
        return self.productions.get(non_terminal, [])

    def num_choices(self, non_terminal: str) -> int:
        """Get number of production choices for a non-terminal.

        Args:
            non_terminal: The non-terminal symbol

        Returns:
            Number of alternative productions
        """
        return len(self.productions.get(non_terminal, []))

    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is a terminal.

        Args:
            symbol: Symbol to check

        Returns:
            True if terminal, False if non-terminal
        """
        return symbol in self.terminals

    def __repr__(self) -> str:
        return f"Grammar({self.name}, type={self.grammar_type.name})"


# Grammar 1: Direct Mapping (Binary)
# <power> ::= 0.0 1.0 - | 1.0
# Produces: -1.0 or 1.0 (bang-bang control)
# Note: RPN "0.0 1.0 -" evaluates as 0.0 - 1.0 = -1.0
DIRECT_MAPPING_GRAMMAR = Grammar(
    name="direct_mapping",
    grammar_type=GrammarType.DIRECT_MAPPING,
    start_symbol="<power>",
    productions={
        "<power>": [
            ("0.0", "1.0", "-"),  # RPN: 0.0 - 1.0 = -1.0
            ("1.0",),  # Produces 1.0
        ],
    },
)

# Grammar 2: Discrete Digits (paper Fig. 4b)
#   <power> ::= <const> 0.0 <op>
#   <op>    ::= + | -
#   <const> ::= 0.0 | 0.1 | ... | 1.0
# Per-rule production counts match the paper (<power>:1, <op>:2, <const>:11), so
# the codon->production mapping matches. Operand order is "0.0 <const> <op>" (not
# the paper's "<const> 0.0 <op>") so the standard RPN evaluator (a OP b) yields
# +const for "+" and 0.0 - const = -const for "-". Range: [-1.0, 1.0], 0.1 step.
DISCRETE_DIGITS_GRAMMAR = Grammar(
    name="discrete_digits",
    grammar_type=GrammarType.DISCRETE_DIGITS,
    start_symbol="<power>",
    productions={
        "<power>": [("0.0", "<const>", "<op>")],
        "<op>": [("+",), ("-",)],
        "<const>": [
            ("0.0",),
            ("0.1",),
            ("0.2",),
            ("0.3",),
            ("0.4",),
            ("0.5",),
            ("0.6",),
            ("0.7",),
            ("0.8",),
            ("0.9",),
            ("1.0",),
        ],
    },
)

# Grammar 3: Continuous Digits (paper Fig. 4c)
#   <power>  ::= <const> 0.0 <op>
#   <op>     ::= + | -
#   <const>  ::= 0.<digits>
#   <digits> ::= <digit><digits> | <digit>
#   <digit>  ::= 0 | 1 | ... | 9
# Same operand-order note as the discrete grammar. The "0." + digit terminals are
# concatenated into one numeric token by the mapper (e.g. 0. 5 6 -> 0.56).
CONTINUOUS_DIGITS_GRAMMAR = Grammar(
    name="continuous_digits",
    grammar_type=GrammarType.CONTINUOUS_DIGITS,
    start_symbol="<power>",
    productions={
        "<power>": [("0.0", "<const>", "<op>")],
        "<op>": [("+",), ("-",)],
        "<const>": [("0.", "<digits>")],
        "<digits>": [("<digit>", "<digits>"), ("<digit>",)],
        "<digit>": [
            ("0",),
            ("1",),
            ("2",),
            ("3",),
            ("4",),
            ("5",),
            ("6",),
            ("7",),
            ("8",),
            ("9",),
        ],
    },
)

# Grammar 4: Symbolic Regression (paper Fig. 4d)
#   <power> ::= <expr> <expr> <op>
#   <expr>  ::= <expr> <expr> <op> | <const>
#   <op>    ::= + | - | * | /
#   <const> ::= 0.<digit>
#   <digit> ::= 0 | 1 | ... | 9
# Produces arbitrary RPN expressions, clamped to [-1.0, 1.0]. The recursive
# <expr> production is listed FIRST, matching the paper; the mapper's depth bound
# (and running out of codons) forces a minimal terminating derivation, so a valid
# phenotype is always produced even though the recursive case is index 0.
SYMBOLIC_REGRESSION_GRAMMAR = Grammar(
    name="symbolic_regression",
    grammar_type=GrammarType.SYMBOLIC_REGRESSION,
    start_symbol="<power>",
    productions={
        "<power>": [("<expr>", "<expr>", "<op>")],
        "<expr>": [("<expr>", "<expr>", "<op>"), ("<const>",)],  # recursive first
        "<op>": [("+",), ("-",), ("*",), ("/",)],
        "<const>": [("0.", "<digit>")],
        "<digit>": [
            ("0",),
            ("1",),
            ("2",),
            ("3",),
            ("4",),
            ("5",),
            ("6",),
            ("7",),
            ("8",),
            ("9",),
        ],
    },
)

# All grammars for easy access
ALL_GRAMMARS = {
    GrammarType.DIRECT_MAPPING: DIRECT_MAPPING_GRAMMAR,
    GrammarType.DISCRETE_DIGITS: DISCRETE_DIGITS_GRAMMAR,
    GrammarType.CONTINUOUS_DIGITS: CONTINUOUS_DIGITS_GRAMMAR,
    GrammarType.SYMBOLIC_REGRESSION: SYMBOLIC_REGRESSION_GRAMMAR,
}
