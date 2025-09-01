"""
Helper utilities
"""

import numpy as np


def set_seed(seed) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


__all__ = ["set_seed"]
