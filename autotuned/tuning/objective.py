import numpy as np


def anomaly_objective(scores, expected_rate):
    threshold = np.percentile(scores, 100 * (1 - expected_rate))
    detected_rate = (scores >= threshold).mean()
    return abs(detected_rate - expected_rate)
