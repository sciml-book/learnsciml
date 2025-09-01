"""Polynomial model"""

from numpy.polynomial import Polynomial as NPPolynomial

from .base import Model


class Polynomial(Model):
    """
    Polynomial regression model

    Parameters
    ----------
    degree : int
        Polynomial degree
    """

    def __init__(self, degree=5) -> None:
        super().__init__()
        self.degree = degree
        self.poly = None

    def train(self, x, y):
        """
        Train polynomial model using least squares

        Parameters
        ----------
        x : ndarray
            Input data
        y : ndarray
            Target values

        Returns
        -------
        self
        """
        self.poly = NPPolynomial.fit(x, y, self.degree)
        self.is_fitted = True
        return self

    def predict(self, x):
        """
        Predict values

        Parameters
        ----------
        x : ndarray
            Input data

        Returns
        -------
        ndarray
            Predictions
        """
        if not self.is_fitted:
            msg = "Model not fitted"
            raise ValueError(msg)
        return self.poly(x)

    @property
    def coefficients(self):
        """Get polynomial coefficients"""
        if not self.is_fitted:
            return None
        return self.poly.coef


# Keep PolynomialModel as alias for backward compatibility
PolynomialModel = Polynomial
