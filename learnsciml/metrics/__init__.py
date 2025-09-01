"""
Error metrics for model evaluation
"""

import numpy as np


def mse(y_true, y_pred):
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def max_error(y_true, y_pred):
    """Maximum absolute error."""
    return np.max(np.abs(y_true - y_pred))


__all__ = ["mse", "rmse", "mae", "max_error"]
