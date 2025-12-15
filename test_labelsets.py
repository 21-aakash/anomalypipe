import numpy as np
from autotuned import AutoTunedModel

data = []

for host in ["A", "B"]:
    base = np.random.normal(0, 1, 1000)
    if host == "A":
        base[100] = 10
    else:
        base[800] = -9

    data.append({
        "labelset": {"host": host},
        "values": base
    })

model = AutoTunedModel(
    tuned_model="zscore",
    anomaly_percentage=0.01,
    n_trials=20,
    timeout=3,
)

model.fit(data)
results = model.predict(data)

for i, res in enumerate(results):
    print(f"Host {data[i]['labelset']['host']} anomaly index:",
          res.argmax())
