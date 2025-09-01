"""Models for function approximation"""

from .base import Model
from .polynomial import Polynomial, PolynomialModel

__all__ = ["Model", "Polynomial", "PolynomialModel"]
