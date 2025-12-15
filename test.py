import numpy as np
from autotuned import AutoTunedModel

# 1. Generate synthetic time series
np.random.seed(42)

normal = np.random.normal(0, 1, 1000)
anomalies = np.array([8, 9, 10, -9, -10])
series = np.concatenate([normal, anomalies])

# 2. Initialize model
model = AutoTunedModel(
    tuned_model="zscore",
    anomaly_percentage=0.01,
    n_trials=30,
    timeout=3,
)

# 3. Fit
model.fit(series)

# 4. Predict
scores = model.predict(series)

# 5. Inspect results
print("Top anomaly scores:")
top_idx = np.argsort(scores)[-10:]
print(top_idx)
print(series[top_idx])
