from __future__ import annotations

import numpy as np


class TemperatureScaler:
    def __init__(self):
        self.temperature_: float = 1.0

    def fit(self, logits, y):
        import torch
        import torch.nn.functional as F

        x = torch.tensor(np.asarray(logits, dtype="float32"))
        target = torch.tensor(np.asarray(y, dtype="float32"))
        log_temperature = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temperature).clamp(0.05, 20.0)
            loss = F.binary_cross_entropy_with_logits(x / temp, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature_ = float(torch.exp(log_temperature).detach().clamp(0.05, 20.0).item())
        return self

    def predict_proba(self, logits):
        logits = np.asarray(logits, dtype=float) / self.temperature_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

