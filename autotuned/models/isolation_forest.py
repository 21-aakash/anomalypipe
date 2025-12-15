import numpy as np
from sklearn.ensemble import IsolationForest
from .base import BaseAnomalyModel


class IsolationForestModel(BaseAnomalyModel):
    """
    Isolation Forest anomaly detection model.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.model = IsolationForest(
            n_estimators=int(params.get("n_estimators", 100)),
            max_samples=params.get("max_samples", "auto"),
            contamination="auto",   # we do NOT leak anomaly_percentage here
            random_state=params.get("random_state", None),
        )

    def fit(self, series: np.ndarray):
        X = series.reshape(-1, 1)
        self.model.fit(X)

    def score(self, series: np.ndarray) -> np.ndarray:
        X = series.reshape(-1, 1)

        # sklearn IF: higher score = more normal
        scores = -self.model.score_samples(X)

        return scores

    @classmethod
    def search_space(cls):
        return {
            "n_estimators": (50, 300),
            "random_state": (0, 10000),
        }
