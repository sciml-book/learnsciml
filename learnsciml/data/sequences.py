"""
Special sequences for quasi-random sampling
"""

import numpy as np


def halton_sequence(n: int, dim: int) -> np.ndarray:
    """
    Generate Halton sequence for quasi-random sampling.

    Parameters
    ----------
    n : int
        Number of points
    dim : int
        Dimension

    Returns
    -------
    np.ndarray
        Halton sequence points
    """

    def halton_single(index, base):
        result = 0
        f = 1.0
        while index > 0:
            f = f / base
            result = result + f * (index % base)
            index = index // base
        return result

    # Use first dim primes as bases
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    points = np.zeros((n, dim))
    for i in range(n):
        for d in range(dim):
            points[i, d] = halton_single(i + 1, primes[d])

    return points
