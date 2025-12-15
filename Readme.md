# AutoTuned

**Unsupervised Anomaly Detection with Automatic Hyperparameter Tuning**

AutoTuned is a Python framework for **unsupervised anomaly detection on time-series data**.
It automatically tunes model hyperparameters **without labels**, using time-aware validation and an expected anomaly rate.

The goal is to remove the manual trial-and-error usually required when deploying anomaly detection systems in real-world pipelines.

---

## Table of Contents

* [Motivation](#motivation)
* [Key Features](#key-features)
* [Core Concepts](#core-concepts)
* [How It Works](#how-it-works)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Supported Models](#supported-models)
* [Validation Strategy](#validation-strategy)
* [Tuning Objective](#tuning-objective)
* [Project Structure](#project-structure)
* [Extending AutoTuned](#extending-autotuned)
* [Persistence](#persistence)
* [Limitations](#limitations)
* [Use Cases](#use-cases)
* [Roadmap](#roadmap)
* [License](#license)

---

## Motivation

Hyperparameter tuning for anomaly detection is fundamentally difficult:

* No ground-truth labels
* Thresholds depend on data distribution
* Time-series data cannot be randomly shuffled
* Models often overfit when tuned incorrectly

AutoTuned addresses these issues by **formalizing tuning as an unsupervised optimization problem** constrained by:

* temporal validation
* an expected anomaly rate
* model stability across folds

---

## Key Features

* Fully **unsupervised** tuning
* **Time-series-aware** cross-validation
* Expected anomaly percentage as a tuning signal
* Modular model registry
* Pluggable optimization logic
* No re-tuning during inference
* Simple sklearn-like API

---

## Core Concepts

### 1. Tuned Once, Used Many Times

Hyperparameter optimization happens **only during `fit()`**.
The selected configuration is frozen and reused for inference.

### 2. Expected Anomaly Rate

Instead of labels, AutoTuned uses an **expected anomaly percentage** to guide threshold selection and scoring.

### 3. Context-Aware Models

Each labelset (or context) maps to **one tuned model instance**, avoiding cross-contamination of distributions.

---

## How It Works

### During `fit()`

1. Input series is split into **time-based folds**
2. Hyperparameter candidates are sampled
3. Each candidate is evaluated using an unsupervised objective
4. The best configuration is selected
5. A final model is trained on full data
6. The tuned model is stored in the context manager

### During `predict()`

1. The last tuned model for the context is loaded
2. Scores are generated
3. **No tuning or retraining occurs**

---

## Installation

```bash
pip install autotuned
```

Or from source:

```bash
git clone https://github.com/your-org/autotuned.git
cd autotuned
pip install -e .
```

---

## Quick Start

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
```

* `series` can be a NumPy array or pandas Series
* Output `scores` are continuous anomaly scores (higher = more anomalous)

---

## Supported Models

Currently implemented models:

| Model   | Description                            |
| ------- | -------------------------------------- |
| Z-Score | Mean/standard-deviation based detector |
| IQR     | Interquartile range based detector     |

Each model defines:

* its own hyperparameter search space
* a consistent scoring interface

---

## Validation Strategy

AutoTuned uses **time-based splitting**, not random shuffling.

Supported strategies:

* Regular forward-chaining splits
* Controlled leaky splits (optional)

This prevents future information from leaking into past evaluations.

---

## Tuning Objective

Because no labels exist, AutoTuned optimizes for:

* Stability of anomaly scores across folds
* Consistency with expected anomaly rate
* Penalization of degenerate thresholds
* Distributional smoothness

The objective is defined in `tuning/objective.py`.

---

## Project Structure

```text
autotuned/
├── README.md
├── pyproject.toml
├── autotuned/
│   ├── __init__.py
│
│   ├── core/
│   │   ├── autotuned_model.py     # Main orchestration class
│   │   ├── context_manager.py     # Labelset → model mapping
│   │   └── model_registry.py      # Built-in model lookup
│
│   ├── models/
│   │   ├── base.py                # BaseAnomalyModel contract
│   │   ├── zscore.py
│   │   └── iqr.py
│
│   ├── validation/
│   │   └── splitter.py            # Time-series splits
│
│   ├── tuning/
│   │   ├── optimizer.py           # Hyperparameter search loop
│   │   ├── objective.py           # Unsupervised objective
│   │   └── search_space.py        # Model parameter spaces
│
│   └── persistence/
│       └── serializer.py          # Save/load tuned models
```

---

## Extending AutoTuned

### Add a New Model

1. Subclass `BaseAnomalyModel`
2. Implement:

   * `fit()`
   * `score()`
3. Define a search space in `search_space.py`
4. Register the model in `model_registry.py`

---

## Persistence

AutoTuned supports serialization of tuned models:

* Save trained configurations
* Reload without re-tuning
* Enables offline training + online inference workflows

---

## Limitations

* No supervised evaluation metrics
* Assumes anomaly rarity
* Designed primarily for univariate time series
* Does not currently support deep learning models

---

## Use Cases

* Infrastructure metrics monitoring
* Log-derived time series
* Sensor data anomaly detection
* Financial or operational telemetry
* Pre-labeling for downstream supervised models

---

## Roadmap

* Multivariate time-series support
* Additional statistical models
* Neural anomaly detectors
* Online / streaming adaptation
* Visualization utilities

---

## License

MIT License
