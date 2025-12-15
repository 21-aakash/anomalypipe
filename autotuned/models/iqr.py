import numpy as np
from .base import BaseAnomalyModel


class IQRModel(BaseAnomalyModel):
    def fit(self, series: np.ndarray):
        q1 = np.percentile(series, 25)
        q3 = np.percentile(series, 75)
        self.iqr_ = q3 - q1
        self.lower_ = q1 - self.params.get("factor", 1.5) * self.iqr_
        self.upper_ = q3 + self.params.get("factor", 1.5) * self.iqr_

    def score(self, series: np.ndarray) -> np.ndarray:
        return np.maximum(series - self.upper_, self.lower_ - series)

    @classmethod
    def search_space(cls):
        return {
            "factor": (0.5, 3.0),
        }
