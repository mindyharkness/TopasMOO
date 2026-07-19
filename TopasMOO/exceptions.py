"""
Custom exception classes for TopasMOO.
"""


class TopasMOOError(Exception):
    """Base exception for all TopasMOO errors."""


class TopasExecutionError(TopasMOOError):
    """Raised when a TOPAS simulation fails to execute."""


class InvalidParameterError(TopasMOOError):
    """Raised when optimization parameters are invalid or inconsistent."""


class ObjectiveFunctionError(TopasMOOError):
    """Raised when the objective function returns unexpected results."""


class MalformedOutputError(TopasMOOError):
    """Raised when TOPAS output files are missing or malformed."""
