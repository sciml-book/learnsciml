"""
Data loading and batching utilities
"""

from typing import Iterator, Optional, Tuple

import numpy as np


class DataLoader1D:
    """
    Simple data loader for 1D datasets.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None
    ) -> None:
        """
        Initialize 1D data loader.

        Parameters
        ----------
        x : np.ndarray
            Input data
        y : np.ndarray
            Target data
        batch_size : int, optional
            Batch size for iteration
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size or len(x)
        self.n_samples = len(x)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield self.x[batch_idx], self.y[batch_idx]

    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a random batch of data."""
        size = batch_size or self.batch_size
        idx = np.random.choice(self.n_samples, size, replace=False)
        return self.x[idx], self.y[idx]


class DataLoader2D:
    """
    Data loader for 2D datasets.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize 2D data loader.

        Parameters
        ----------
        X, Y : np.ndarray
            Meshgrid arrays
        Z : np.ndarray
            Target values
        batch_size : int, optional
            Batch size for iteration
        """
        self.X_flat = X.flatten()
        self.Y_flat = Y.flatten()
        self.Z_flat = Z.flatten()
        self.shape = X.shape
        self.batch_size = batch_size or len(self.X_flat)
        self.n_samples = len(self.X_flat)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield (
                self.X_flat[batch_idx],
                self.Y_flat[batch_idx],
                self.Z_flat[batch_idx],
            )

    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a random batch of data."""
        size = batch_size or self.batch_size
        idx = np.random.choice(self.n_samples, size, replace=False)
        return self.X_flat[idx], self.Y_flat[idx], self.Z_flat[idx]


class BatchLoader:
    """
    Generic batch loader with support for train/validation split.
    """

    def __init__(
        self,
        *arrays: np.ndarray,
        batch_size: int = 32,
        validation_split: float = 0.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize batch loader.

        Parameters
        ----------
        *arrays : np.ndarray
            Data arrays to batch together
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data for validation
        shuffle : bool
            Whether to shuffle data
        seed : int, optional
            Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.arrays = arrays
        self.n_samples = len(arrays[0])
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Split indices
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        n_val = int(validation_split * self.n_samples)
        self.train_idx = indices[n_val:]
        self.val_idx = indices[:n_val] if n_val > 0 else None

    def get_train_batches(self) -> Iterator[Tuple[np.ndarray, ...]]:
        """Get training batches."""
        indices = self.train_idx.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = min(start + self.batch_size, len(indices))
            batch_idx = indices[start:end]
            yield tuple(arr[batch_idx] for arr in self.arrays)

    def get_validation_data(self) -> Tuple[np.ndarray, ...]:
        """Get validation data."""
        if self.val_idx is None:
            return None
        return tuple(arr[self.val_idx] for arr in self.arrays)
