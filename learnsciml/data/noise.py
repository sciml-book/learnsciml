"""
Noise generation utilities for data augmentation
"""

from typing import Optional

import numpy as np


def add_gaussian_noise(
    data: np.ndarray,
    std: float = 0.01,
    relative: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add Gaussian noise to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    std : float
        Standard deviation of noise
    relative : bool
        If True, std is relative to data magnitude
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Noisy data
    """
    if seed is not None:
        np.random.seed(seed)

    noise_std = std * np.abs(data) if relative else std

    noise = np.random.randn(*data.shape) * noise_std
    return data + noise


def add_uniform_noise(
    data: np.ndarray,
    amplitude: float = 0.01,
    relative: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add uniform noise to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    amplitude : float
        Maximum noise amplitude
    relative : bool
        If True, amplitude is relative to data magnitude
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Noisy data
    """
    if seed is not None:
        np.random.seed(seed)

    noise_amp = amplitude * np.abs(data) if relative else amplitude

    noise = (2 * np.random.rand(*data.shape) - 1) * noise_amp
    return data + noise


def add_outliers(
    data: np.ndarray,
    fraction: float = 0.05,
    magnitude: float = 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add outliers to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    fraction : float
        Fraction of points to make outliers
    magnitude : float
        Outlier magnitude (in standard deviations)
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Data with outliers
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_data = data.copy()
    n_outliers = int(fraction * data.size)

    # Random indices for outliers
    outlier_idx = np.random.choice(data.size, n_outliers, replace=False)

    # Unravel for multidimensional arrays
    outlier_coords = np.unravel_index(outlier_idx, data.shape)

    # Add large deviations
    data_std = np.std(data)
    outlier_values = magnitude * data_std * (2 * np.random.rand(n_outliers) - 1)

    noisy_data[outlier_coords] += outlier_values

    return noisy_data


def add_multiplicative_noise(
    data: np.ndarray, factor: float = 0.1, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add multiplicative noise to data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    factor : float
        Noise factor (fraction of signal)
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Noisy data
    """
    if seed is not None:
        np.random.seed(seed)

    noise = 1 + factor * (2 * np.random.rand(*data.shape) - 1)
    return data * noise
