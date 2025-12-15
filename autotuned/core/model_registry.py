from autotuned.models.zscore import ZScoreModel
from autotuned.models.iqr import IQRModel
from autotuned.models.isolation_forest import IsolationForestModel


MODEL_REGISTRY = {
    "zscore": ZScoreModel,
    "iqr": IQRModel,
     "isolation_forest": IsolationForestModel,
}


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]
