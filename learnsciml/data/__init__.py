"""
Data generation and manipulation utilities for Scientific Machine Learning
"""

from .generators import (
    generate_1d,
    generate_2d,
    generate_physics_informed,
    generate_sparse,
)
from .loaders import BatchLoader, DataLoader1D, DataLoader2D
from .noise import (
    add_gaussian_noise,
    add_multiplicative_noise,
    add_outliers,
    add_uniform_noise,
)

__all__ = [
    "generate_1d",
    "generate_2d",
    "generate_sparse",
    "generate_physics_informed",
    "add_gaussian_noise",
    "add_uniform_noise",
    "add_outliers",
    "add_multiplicative_noise",
    "DataLoader1D",
    "DataLoader2D",
    "BatchLoader",
]
