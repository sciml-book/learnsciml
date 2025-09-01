"""
Generic plotting utilities
"""

import matplotlib.pyplot as plt
import numpy as np


def plot(x, y, y_pred=None, title=None, xlabel="x", ylabel="y", ax=None, **kwargs):
    """
    Generic plot function for 1D data.

    Parameters
    ----------
    x, y : arrays
        Data points
    y_pred : array, optional
        Predictions to overlay
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    ax : matplotlib axis, optional
        Axis to plot on
    **kwargs : dict
        Additional plot kwargs
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data
    ax.scatter(x, y, alpha=0.6, s=30, label="Data", zorder=5)

    # Plot prediction if provided
    if y_pred is not None:
        ax.plot(x, y_pred, "r-", linewidth=2, label="Model", **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def compare(x, y_true, models_dict, title=None, xlabel="x", ylabel="y", ax=None):
    """
    Compare multiple models on same data.

    Parameters
    ----------
    x : array
        Input points
    y_true : array
        True values
    models_dict : dict
        {name: y_pred} for each model
    title : str, optional
        Plot title
    xlabel, ylabel : str
        Axis labels
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot true data
    ax.scatter(x, y_true, alpha=0.6, s=30, color="black", label="Data", zorder=5)

    # Plot each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))
    for (name, y_pred), color in zip(models_dict.items(), colors):
        ax.plot(x, y_pred, linewidth=2, label=name, color=color, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_model_comparison(x_train, y_train, x_test, models, y_true=None, figsize=None):
    """
    Compare multiple models side by side.
    
    Parameters
    ----------
    x_train, y_train : arrays
        Training data points to show
    x_test : array
        Test points for smooth prediction curves
    models : dict
        {label: model} or {label: y_pred} dictionary
    y_true : array, optional
        True function values at x_test
    figsize : tuple, optional
        Figure size (default: auto)
    """
    n_models = len(models)
    if figsize is None:
        figsize = (4 * min(n_models, 4), 4 * ((n_models - 1) // 4 + 1))
    
    if n_models <= 4:
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
    else:
        nrows = (n_models - 1) // 4 + 1
        ncols = min(4, n_models)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, (label, model_or_pred) in enumerate(models.items()):
        ax = axes[idx]
        
        # Training data
        ax.scatter(x_train, y_train, alpha=0.6, s=30, label='Data')
        
        # True function if provided
        if y_true is not None:
            ax.plot(x_test, y_true, 'k--', alpha=0.5, label='True')
        
        # Model prediction
        if hasattr(model_or_pred, 'predict'):
            y_pred = model_or_pred.predict(x_test)
        else:
            y_pred = model_or_pred
        ax.plot(x_test, y_pred, 'r-', linewidth=2, label='Model')
        
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_metrics(x_values, y_values, xlabel='Parameter', ylabel='Metric', 
                 title=None, log_y=False, mark_optimal=True):
    """
    Generic metric plotting function.
    
    Parameters
    ----------
    x_values : array
        X-axis values (e.g., epochs, parameters, degrees)
    y_values : array
        Y-axis values (e.g., loss, error, accuracy)
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title
    log_y : bool
        Use log scale for y-axis
    mark_optimal : bool
        Mark the optimal (minimum) point
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if log_y:
        ax.set_yscale('log')
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    if mark_optimal:
        optimal_idx = np.argmin(y_values)
        ax.scatter(x_values[optimal_idx], y_values[optimal_idx], 
                   color='red', s=100, zorder=5, 
                   label=f'Optimal: {x_values[optimal_idx]}')
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_loss(losses, val_losses=None, ax=None):
    """
    Plot training (and validation) loss.

    Parameters
    ----------
    losses : array
        Training losses
    val_losses : array, optional
        Validation losses
    ax : matplotlib axis, optional
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, "b-", linewidth=2, label="Train")

    if val_losses is not None:
        ax.plot(epochs, val_losses, "r--", linewidth=2, label="Validation")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Wrapper for plt.subplots with better defaults.
    """
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Make axes always iterable
    if nrows * ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()

    return fig, axes


__all__ = [
    "plot", 
    "compare", 
    "plot_loss", 
    "subplots",
    "plot_model_comparison",
    "plot_metrics"
]
