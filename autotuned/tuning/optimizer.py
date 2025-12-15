import time
import random
import numpy as np
from autotuned.tuning.objective import anomaly_objective


def optimize(model_cls, splits, expected_rate, n_trials, timeout):
    best_loss = float("inf")
    best_params = None
    start = time.time()

    space = model_cls.search_space()

    for _ in range(n_trials):
        if time.time() - start > timeout:
            break

        params = {
            k: random.uniform(v[0], v[1])
            for k, v in space.items()
        }

        losses = []

        for train, val in splits:
            model = model_cls(**params)
            model.fit(train)
            scores = model.score(val)
            losses.append(anomaly_objective(scores, expected_rate))

        loss = np.mean(losses)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    return best_params
