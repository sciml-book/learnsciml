# learnsciml

A clean, modular toolkit for Scientific Machine Learning education.

## Installation

For development (editable installation):
```bash
pip install -e .
```

With optional dependencies:
```bash
# For development tools
pip install -e ".[dev]"

# For PyTorch support
pip install -e ".[torch]"

# Everything
pip install -e ".[full]"
```

## Usage

```python
import learnsciml as sciml
import numpy as np

# Generate data
x, y = sciml.data.generate_1d(lambda x: np.sin(2*np.pi*x), n_points=50, noise=0.1)

# Create and train model
model = sciml.models.Polynomial(degree=5)
model.fit(x, y)

# Make predictions
y_pred = model.predict(x)

# Visualize
fig = sciml.viz.plot_1d(x, y, y_pred)

# Compute metrics
metrics = sciml.utils.compute_metrics(y, y_pred)
print(f"RMSE: {metrics['rmse']:.4f}")
```

## Structure

```
learnsciml/
├── data/          # Data generation and loading
│   ├── generators.py   # Function sampling, physics-informed data
│   ├── noise.py        # Noise models
│   └── loaders.py      # Batch loading utilities
├── viz/           # Visualization
│   ├── plotting.py     # Core plotting functions
│   └── analysis.py     # Diagnostic plots
├── models/        # Model implementations
│   ├── base.py         # Base classes
│   ├── polynomial.py   # Classical methods
│   └── neural.py       # Neural networks
└── utils/         # Utilities
    ├── metrics.py      # Error metrics
    └── helpers.py      # Helper functions
```