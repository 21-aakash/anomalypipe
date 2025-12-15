from abc import ABC, abstractmethod
import numpy as np


class BaseAnomalyModel(ABC):
    """
    Base contract for all anomaly models.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, series: np.ndarray):
        pass

    @abstractmethod
    def score(self, series: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores (higher = more anomalous).
        """
        pass

    def predict(self, series: np.ndarray) -> np.ndarray:
        return self.score(series)

    @classmethod
    @abstractmethod
    def search_space(cls) -> dict:
        """
        Defines hyperparameter search space.
        """
        pass
