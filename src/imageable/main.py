from typing import Any

import numpy as np


def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.

    Parameters
    ----------
    a : float
        First number to add.
    b : float
        Second number to add.

    Returns
    -------
    float
        Sum of a and b.

    Examples
    --------
    >>> add_numbers(2.0, 3.0)
    5.0
    """
    return a + b


def process_array(arr: list[float], multiplier: float | None = None) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    Process an array of numbers.

    Parameters
    ----------
    arr : List[float]
        Input array of numbers.
    multiplier : Optional[float], default=None
        Optional multiplier to apply to each element.

    Returns
    -------
    np.ndarray
        Processed array.
    """
    np_arr = np.array(arr)
    if multiplier is not None:
        np_arr = np_arr * multiplier
    return np_arr


class Calculator:
    """A simple calculator class."""

    def __init__(self, precision: int = 2) -> None:
        """
        Initialize the calculator.

        Parameters
        ----------
        precision : int, default=2
            Number of decimal places for rounding results.
        """
        self.precision = precision

    def divide(self, a: float, b: float) -> float:
        """
        Divide two numbers.

        Parameters
        ----------
        a : float
            Dividend.
        b : float
            Divisor.

        Returns
        -------
        float
            Result of a divided by b.

        Raises
        ------
        ValueError
            If b is zero.
        """
        if b == 0:
            error = "Cannot divide by zero"
            raise ValueError(error)
        return round(a / b, self.precision)
