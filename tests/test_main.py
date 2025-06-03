"""Tests for main module."""

import pytest
import numpy as np
from imageable.main import add_numbers, process_array, Calculator


def test_add_numbers() -> None:
    """Test add_numbers function."""
    assert add_numbers(2.0, 3.0) == 5.0
    assert add_numbers(-1.0, 1.0) == 0.0


def test_process_array() -> None:
    """Test process_array function."""
    result = process_array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)


class TestCalculator:
    """Test Calculator class."""

    def test_divide(self) -> None:
        """Test divide method."""
        calc = Calculator(precision=2)
        assert calc.divide(10.0, 3.0) == 3.33

    def test_divide_by_zero(self) -> None:
        """Test divide by zero raises ValueError."""
        calc = Calculator()
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10.0, 0.0)
