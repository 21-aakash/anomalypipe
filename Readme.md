# AutoTuned – Unsupervised Anomaly Detection with Auto Hyperparameter Tuning

AutoTuned is a Python framework for unsupervised anomaly detection that
automatically tunes hyperparameters of built-in models using time-series-aware
validation and an expected anomaly rate.

## Why AutoTuned?

Hyperparameter tuning for anomaly detection is hard:
- No labels
- Thresholds are data-dependent
- Time series require special validation

AutoTuned removes this burden.

## Core idea

During `fit()`:
- Data is split into time-based folds
- Candidate hyperparameters are evaluated
- The best configuration is selected
- One tuned model is trained per labelset

During `predict()`:
- The last tuned model is used for inference
- No re-tuning happens

## Minimal usage

```python
from autotuned import AutoTunedModel

model = AutoTunedModel(
    tuned_model="zscore",
    anomaly_percentage=0.01,
    n_trials=50,
    timeout=5,
)

model.fit(series)
scores = model.predict(series)


autotuned/
├── README.md
├── pyproject.toml
├── autotuned/
│   ├── __init__.py
│
│   ├── core/
│   │   ├── __init__.py
│   │   ├── autotuned_model.py     # Orchestrates tuning (main class)
│   │   ├── context_manager.py     # Labelset → model mapping
│   │   └── model_registry.py      # Built-in model lookup
│
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseAnomalyModel contract
│   │   ├── zscore.py              # Z-Score model
│   │   └── iqr.py                 # IQR model
│
│   ├── validation/
│   │   ├── __init__.py
│   │   └── splitter.py            # regular / leaky splits
│
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── optimizer.py           # tuning loop
│   │   ├── objective.py           # unsupervised objective
│   │   └── search_space.py        # model hyperparameter spaces
│
│   └── persistence/
│       ├── __init__.py
│       └── serializer.py          # save/load tuned models


