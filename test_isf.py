import numpy as np
from autotuned import AutoTunedModel

np.random.seed(0)

series = np.random.normal(0, 1, 2000)
series[300:305] = 7
series[1200:1205] = -8

model = AutoTunedModel(
    tuned_model="isolation_forest",
    anomaly_percentage=0.01,
    n_trials=20,
    timeout=5,
)

model.fit(series)
scores = model.predict(series)

print("Detected anomaly indices:")
print(np.argsort(scores)[-10:])
