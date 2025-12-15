import numpy as np
from autotuned import AutoTunedModel

series = np.random.normal(0, 1, 1000)
series[500] = 9

# Train & save
model = AutoTunedModel(
    tuned_model="zscore",
    anomaly_percentage=0.01,
)

model.fit(series)
model.save()

# Load & infer
loaded = AutoTunedModel(
    tuned_model="zscore",
    anomaly_percentage=0.01,
)

loaded.load()
scores = loaded.predict(series)

print("Anomaly index:", scores.argmax())
