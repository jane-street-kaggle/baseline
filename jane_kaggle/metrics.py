import numpy as np

def calculate_r2(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * (y_true ** 2))
    r2_score = 1 - (numerator / denominator)
    return r2_score