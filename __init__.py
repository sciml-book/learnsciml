"""
learnsciml: Clean, modular toolkit for Scientific Machine Learning education

A pedagogical library for scientific machine learning that prioritizes
clarity and understanding over performance.

Usage:
    import learnsciml as sciml

    # Generate data
    x, y = sciml.data.generate_1d(lambda x: np.sin(x), n_points=100)

    # Train model
    model = sciml.models.Polynomial(degree=5)
    model.fit(x, y)

    # Visualize
    sciml.viz.plot_1d(x, y, model.predict(x))
"""

__version__ = "0.3.0"

# Import submodules directly to avoid circular import
from .learnsciml import data, metrics, models, utils, viz

__all__ = ["data", "viz", "models", "metrics", "utils", "__version__"]
