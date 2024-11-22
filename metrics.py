import numpy as np


def r2_metric(y_true, y_pred, weights=None):
    """Calculate weighted R2 score"""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # If weights is None, use uniform weights
    if weights is None:
        weights = np.ones_like(y_true)
    else:
        weights = weights.ravel()

    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true ** 2))
    r2_score = 1 - (numerator / denominator)
    return 'r2', r2_score, True