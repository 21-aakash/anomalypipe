import os
import pickle


def save_model(model, path: str):
    """
    Persist a trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """
    Load a persisted model from disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
