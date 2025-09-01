"""
Data generation utilities for various scientific computing scenarios
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np


def generate_1d(
    func: Callable,
    n_points: int = 50,
    x_range: Tuple[float, float] = (0, 1),
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 1D data from a function.

    Parameters
    ----------
    func : callable
        Function to sample
    n_points : int
        Number of points
    x_range : tuple
        (min, max) range for x
    noise : float
        Noise standard deviation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    x, y : np.ndarray
        Input and output data
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = func(x)

    if noise > 0:
        y = y + noise * np.random.randn(n_points)

    return x, y


def generate_2d(
    func: Callable,
    n_points: Union[int, Tuple[int, int]] = (20, 20),
    x_range: Tuple[float, float] = (0, 1),
    y_range: Tuple[float, float] = (0, 1),
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D grid data from a function.

    Parameters
    ----------
    func : callable
        Function f(x, y) to sample
    n_points : int or tuple
        Number of points in each dimension
    x_range : tuple
        (min, max) range for x
    y_range : tuple
        (min, max) range for y
    noise : float
        Noise standard deviation
    seed : int, optional
        Random seed

    Returns
    -------
    X, Y, Z : np.ndarray
        Meshgrid arrays
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(n_points, int):
        n_points = (n_points, n_points)

    x = np.linspace(x_range[0], x_range[1], n_points[0])
    y = np.linspace(y_range[0], y_range[1], n_points[1])
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    if noise > 0:
        Z = Z + noise * np.random.randn(*Z.shape)

    return X, Y, Z


def generate_sparse(
    func: Callable,
    n_points: int = 20,
    domain: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (0, 1),
    sampling: str = "uniform",
    noise: float = 0.0,
    seed: Optional[int] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate sparse data for testing interpolation and approximation.

    Parameters
    ----------
    func : callable
        Function to sample
    n_points : int
        Number of sparse points
    domain : tuple
        Domain specification (1D or 2D)
    sampling : str
        'uniform', 'random', 'chebyshev', or 'halton'
    noise : float
        Noise level
    seed : int, optional
        Random seed

    Returns
    -------
    Points and values
    """
    if seed is not None:
        np.random.seed(seed)

    # Handle 1D case
    if not isinstance(domain[0], tuple):
        if sampling == "uniform":
            x = np.linspace(domain[0], domain[1], n_points)
        elif sampling == "random":
            x = np.sort(np.random.uniform(domain[0], domain[1], n_points))
        elif sampling == "chebyshev":
            # Chebyshev points for better interpolation
            i = np.arange(n_points)
            x = 0.5 * (domain[0] + domain[1]) + 0.5 * (domain[1] - domain[0]) * np.cos(
                (2 * i + 1) * np.pi / (2 * n_points)
            )
            x = np.sort(x)
        else:
            msg = f"Unknown sampling method: {sampling}"
            raise ValueError(msg)

        y = func(x)
        if noise > 0:
            y = y + noise * np.random.randn(n_points)
        return x, y

    # Handle 2D case
    x_range, y_range = domain
    if sampling == "uniform":
        n_per_dim = int(np.sqrt(n_points))
        x = np.linspace(x_range[0], x_range[1], n_per_dim)
        y = np.linspace(y_range[0], y_range[1], n_per_dim)
        X, Y = np.meshgrid(x, y)
        X, Y = X.flatten(), Y.flatten()
    elif sampling == "random":
        X = np.random.uniform(x_range[0], x_range[1], n_points)
        Y = np.random.uniform(y_range[0], y_range[1], n_points)
    elif sampling == "halton":
        # Quasi-random Halton sequence for better coverage
        from .sequences import halton_sequence

        points = halton_sequence(n_points, 2)
        X = x_range[0] + (x_range[1] - x_range[0]) * points[:, 0]
        Y = y_range[0] + (y_range[1] - y_range[0]) * points[:, 1]
    else:
        msg = f"Unknown sampling method: {sampling}"
        raise ValueError(msg)

    Z = func(X, Y)
    if noise > 0:
        Z = Z + noise * np.random.randn(len(Z))
    return X, Y, Z


def generate_physics_informed(
    pde_type: str,
    n_boundary: int = 50,
    n_collocation: int = 100,
    domain: Union[
        Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]
    ] = (0, 1),
    seed: Optional[int] = None,
) -> dict:
    """
    Generate training data for physics-informed neural networks.

    Parameters
    ----------
    pde_type : str
        Type of PDE ('poisson', 'heat', 'wave', etc.)
    n_boundary : int
        Number of boundary points
    n_collocation : int
        Number of collocation points
    domain : tuple
        Domain specification
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with boundary and collocation points
    """
    if seed is not None:
        np.random.seed(seed)

    data = {}

    # Handle 1D case
    if not isinstance(domain[0], tuple):
        # Boundary points
        data["x_boundary"] = np.array([domain[0], domain[1]])

        # Collocation points (interior)
        data["x_collocation"] = np.random.uniform(domain[0], domain[1], n_collocation)

        # Example boundary conditions
        if pde_type == "poisson":
            data["u_boundary"] = np.array([0.0, 0.0])  # Dirichlet BC

    # Handle 2D case
    else:
        x_range, y_range = domain

        # Boundary points (along edges)
        n_per_edge = n_boundary // 4

        # Bottom edge
        x_bottom = np.linspace(x_range[0], x_range[1], n_per_edge)
        y_bottom = np.full(n_per_edge, y_range[0])

        # Top edge
        x_top = np.linspace(x_range[0], x_range[1], n_per_edge)
        y_top = np.full(n_per_edge, y_range[1])

        # Left edge
        x_left = np.full(n_per_edge, x_range[0])
        y_left = np.linspace(y_range[0], y_range[1], n_per_edge)

        # Right edge
        x_right = np.full(n_per_edge, x_range[1])
        y_right = np.linspace(y_range[0], y_range[1], n_per_edge)

        data["x_boundary"] = np.concatenate([x_bottom, x_top, x_left, x_right])
        data["y_boundary"] = np.concatenate([y_bottom, y_top, y_left, y_right])

        # Collocation points (interior)
        data["x_collocation"] = np.random.uniform(x_range[0], x_range[1], n_collocation)
        data["y_collocation"] = np.random.uniform(y_range[0], y_range[1], n_collocation)

        # Example boundary conditions
        if pde_type == "poisson":
            data["u_boundary"] = np.zeros(len(data["x_boundary"]))

    return data
