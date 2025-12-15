import os
import numpy as np
from autotuned.core.model_registry import get_model_class
from autotuned.validation.splitter import time_series_splits
from autotuned.tuning.optimizer import optimize
from autotuned.core.context_manager import ContextManager
from autotuned.persistence.serializer import save_model, load_model


class AutoTunedModel:
    def __init__(
        self,
        tuned_model: str,
        anomaly_percentage: float,
        n_splits: int = 3,
        train_val_ratio: int = 3,
        n_trials: int = 128,
        timeout: float = 10.0,
        storage_dir: str = "models_store",
    ):
        self.model_name = tuned_model
        self.anomaly_percentage = anomaly_percentage
        self.n_splits = n_splits
        self.train_val_ratio = train_val_ratio
        self.n_trials = n_trials
        self.timeout = timeout
        self.storage_dir = storage_dir

        # labelset -> trained model
        self.models_ = {}

    def _labelset_to_filename(self, labelset):
        if labelset is None:
            return "default.pkl"
        return f"{hash(labelset)}.pkl"

    def _fit_single_series(self, series: np.ndarray):
        model_cls = get_model_class(self.model_name)

        splits = time_series_splits(
            series,
            self.n_splits,
            self.train_val_ratio,
        )

        best_params = optimize(
            model_cls,
            splits,
            self.anomaly_percentage,
            self.n_trials,
            self.timeout,
        )

        model = model_cls(**best_params)
        model.fit(series)
        return model

    def fit(self, data):
        # Single series
        if isinstance(data, np.ndarray):
            model = self._fit_single_series(data)
            self.models_[None] = model
            return

        grouped = ContextManager.group_by_labelset(data)

        for labelset, series_list in grouped.items():
            merged_series = np.concatenate(series_list)
            self.models_[labelset] = self._fit_single_series(merged_series)

    def predict(self, data):
        if not self.models_:
            raise RuntimeError("Model not fitted or loaded")

        if isinstance(data, np.ndarray):
            return self.models_[None].predict(data)

        results = []
        for item in data:
            labelset = frozenset(item["labelset"].items())
            model = self.models_.get(labelset)
            if model is None:
                raise ValueError(f"No model for labelset {dict(labelset)}")
            results.append(model.predict(item["values"]))
        return results

    def save(self):
        """
        Save all trained models to disk.
        """
        for labelset, model in self.models_.items():
            filename = self._labelset_to_filename(labelset)
            path = os.path.join(self.storage_dir, filename)
            save_model(model, path)

    def load(self):
        """
        Load models from disk into memory.
        """
        self.models_.clear()

        if not os.path.exists(self.storage_dir):
            raise FileNotFoundError("Storage directory not found")

        for file in os.listdir(self.storage_dir):
            path = os.path.join(self.storage_dir, file)
            model = load_model(path)

            if file == "default.pkl":
                self.models_[None] = model
            else:
                # hashed labelset (opaque, but consistent)
                self.models_[file] = model
