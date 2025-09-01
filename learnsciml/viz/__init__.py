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


__all__ = ["plot", "compare", "plot_loss", "subplots"]
