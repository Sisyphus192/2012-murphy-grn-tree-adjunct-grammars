"""Pytest configuration and fixtures."""

import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from grntage.grn.genome import Genome

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --update-golden flag for regenerating golden-master files."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden-master .npy files (refused under $CI).",
    )


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Compare a numpy array against a stored golden master, or regenerate it.

    Returns a callable ``check(name, array, *, rtol=1e-9, atol=1e-12, exact=False)``.
    With ``--update-golden`` it (re)writes ``tests/golden/<name>.npy`` instead of
    asserting; that flag is refused under ``$CI`` so golden files are only ever
    regenerated and reviewed locally. The default tolerance is honest about
    reduction-order changes from the numba refactor (rtol 1e-9, not 1e-12).
    """
    update = bool(request.config.getoption("--update-golden"))
    if update and os.environ.get("CI"):
        pytest.fail("--update-golden is refused under $CI; regenerate goldens locally.")

    def _check(
        name: str,
        array: Any,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-12,
        exact: bool = False,
    ) -> None:
        path = GOLDEN_DIR / f"{name}.npy"
        arr = np.asarray(array)
        if update:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
            return
        if not path.exists():
            pytest.fail(
                f"Golden {path} missing; regenerate with: "
                f"uv run pytest <test> --update-golden"
            )
        expected = np.load(path)
        assert arr.shape == expected.shape, (
            f"golden {name}: shape {arr.shape} != stored {expected.shape}"
        )
        if exact:
            np.testing.assert_array_equal(arr, expected)
        else:
            np.testing.assert_allclose(arr, expected, rtol=rtol, atol=atol)

    return _check


@pytest.fixture
def rng() -> random.Random:
    """Seeded random number generator for reproducibility."""
    return random.Random(42)


@pytest.fixture
def random_genome(rng: random.Random) -> Genome:
    """Create a random genome for testing."""
    return Genome.random(4096, rng)


@pytest.fixture
def small_genome(rng: random.Random) -> Genome:
    """Create a smaller genome for faster tests."""
    return Genome.random(1024, rng)
