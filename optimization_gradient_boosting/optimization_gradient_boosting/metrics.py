import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def variance_reduction(y: np.array, y_left: np.array, y_right: np.array) -> float:
    """Calculate de variance reduction given the y of the parent and children"""
    ntotal = len(y)
    weight_left = len(y_left) / ntotal
    weight_right = len(y_right) / ntotal
    return np.var(y) - (np.var(y_left) * weight_left + np.var(y_right) * weight_right)

def mse(y_true: np.array, y_pred: np.array) -> float:
    """Compute the mean squared error."""
    return np.mean(np.square(np.subtract(y_true, y_pred)))