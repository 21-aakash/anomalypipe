import numpy as np
from .base import BaseAnomalyModel


class ZScoreModel(BaseAnomalyModel):
    def fit(self, series: np.ndarray):
        self.mean_ = np.mean(series)
        self.std_ = np.std(series) + 1e-8

    def score(self, series: np.ndarray) -> np.ndarray:
        z = np.abs((series - self.mean_) / self.std_)
        return z

    @classmethod
    def search_space(cls):
        return {
            "z_threshold": (1.5, 5.0),
        }
