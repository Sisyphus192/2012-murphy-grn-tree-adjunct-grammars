"""Base class for output mapping methods.

Defines the interface for converting P-protein states to codon values.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto

from grntage.grn.protein import Protein


class OutputMethod(Enum):
    """Available output mapping methods."""

    CONCENTRATION_VALUE = auto()  # Binary: threshold at 0.5
    CONCENTRATION_TENDENCY = auto()  # Binary: track change direction
    SORT_BY_CONCENTRATION = auto()  # Multi-codon: sort by concentration
    SORT_BY_TENDENCY = auto()  # Multi-codon: sort by change magnitude


class OutputMapper(ABC):
    """Abstract base class for output mapping methods.

    Converts P-protein concentrations to integer codons for grammar mapping.
    """

    @property
    @abstractmethod
    def method(self) -> OutputMethod:
        """Get the output method type."""
        ...

    @abstractmethod
    def map_to_codons(
        self,
        proteins: list[Protein],
        previous_concentrations: dict[int, float] | None = None,
    ) -> list[int]:
        """Map P-proteins to codon values.

        Args:
            proteins: List of P-proteins with current concentrations
            previous_concentrations: Dict mapping signature to previous concentration
                                    (required for tendency-based methods)

        Returns:
            List of integer codon values
        """
        ...

    @abstractmethod
    def is_binary(self) -> bool:
        """Check if this mapper produces binary (single codon) output.

        Returns:
            True if binary output, False if multi-codon
        """
        ...

    def get_previous_concentrations(self, proteins: list[Protein]) -> dict[int, float]:
        """Get current concentrations as a snapshot for future tendency calculation.

        Args:
            proteins: List of P-proteins

        Returns:
            Dict mapping signature to concentration
        """
        return {p.signature: p.concentration for p in proteins}

    def reset(self) -> None:
        """Reset mapper state.

        Override in subclasses that maintain state between evaluations.
        Called before each individual evaluation.
        """
        pass  # Default no-op for stateless mappers
