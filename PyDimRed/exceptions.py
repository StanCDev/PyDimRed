"""
This module defines new Error(s) used in the library. They are:

- DimensionError: should be raised when dimension constraints are not respected in a method

This module also contains utility functions to raise these errors if a boolean condition is not satisfied.
"""

class DimensionError(Exception):
    """
    Exception raised for errors in data dimensions.

    Attributes:
    -----------
        message (str): explanation of the error

        X1 : Shape of first array

        X2 : Shape of second array
    """

    def __init__(self, message: str, X1=None, X2=None) -> None:
        self.message = message
        self.X1 = X1
        self.X2 = X2
        super().__init__(self.message)

    def __str__(self) -> str:
        error_details = self.message
        if self.X1 is not None and self.X2 is not None:
            error_details += f" (Array 1 : {self.X1}, Array 2 : {self.X2})"
        return error_details


def checkCondition(cond: bool, message: str = ""):
    """
    Check if a boolean condition is true. If not raise a Value Error

    Args:
    -----
        cond (bool): condition to check

        message (str): error message to print

    Returns:
    --------
        None
    """
    if not cond:
        raise ValueError(message)


def checkDimensionCondition(cond: bool, message: str = "", X1=None, X2=None):
    """
    Check if a boolean condition related to array dimensions is true. If not raise a Dimension Error

    Args:
    -----
        cond (bool): condition to check

        message (str): error message to print

        X1: shape of first array

        X2: shape of second array

    Returns:
    --------
        None
    """
    if not cond:
        raise DimensionError(message, X1, X2)
