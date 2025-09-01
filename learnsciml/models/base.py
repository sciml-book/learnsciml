"""Base model class"""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Abstract base class for all models"""

    def __init__(self) -> None:
        self.is_fitted = False

    @abstractmethod
    def train(self, x, y):
        """Train model on data"""

    @abstractmethod
    def predict(self, x):
        """Make predictions"""

    def score(self, x, y, metric="mse"):
        """
        Evaluate model performance

        Parameters
        ----------
        x : ndarray
            Input data
        y : ndarray
            True values
        metric : str
            Metric to compute ('mse', 'rmse', 'mae', 'max')

        Returns
        -------
        float
            Score value
        """
        if not self.is_fitted:
            msg = "Model not fitted"
            raise ValueError(msg)

        y_pred = self.predict(x)

        if metric == "mse":
            return np.mean((y - y_pred) ** 2)
        if metric == "rmse":
            return np.sqrt(np.mean((y - y_pred) ** 2))
        if metric == "mae":
            return np.mean(np.abs(y - y_pred))
        if metric == "max":
            return np.max(np.abs(y - y_pred))
        msg = f"Unknown metric: {metric}"
        raise ValueError(msg)
