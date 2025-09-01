"""Basic tests for learnsciml package"""

import numpy as np

import learnsciml as sciml


def test_import():
    """Test that package can be imported"""
    assert sciml is not None


def test_data_generation():
    """Test data generation"""

    def sin_func(x):
        return np.sin(x)

    x, y = sciml.data.generate_1d(sin_func, n_points=10)
    assert len(x) == 10
    assert len(y) == 10


def test_polynomial_model():
    """Test polynomial model fitting"""
    x = np.linspace(0, 1, 10)
    y = x**2

    model = sciml.models.Polynomial(degree=2)
    model.train(x, y)
    y_pred = model.predict(x)

    # Should fit perfectly for polynomial data
    mse = sciml.metrics.mse(y, y_pred)
    assert mse < 1e-10


def test_metrics():
    """Test metric functions"""
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.1, 2.1, 2.9])

    mse = sciml.metrics.mse(y_true, y_pred)
    assert mse > 0

    rmse = sciml.metrics.rmse(y_true, y_pred)
    assert rmse == np.sqrt(mse)
