from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import logit


class PlattCalibrator:
    def __init__(self, input_is_probability: bool = False):
        self.input_is_probability = input_is_probability
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    def _prepare(self, scores):
        scores = np.asarray(scores, dtype=float)
        if self.input_is_probability:
            scores = logit(scores)
        return scores.reshape(-1, 1)

    def fit(self, scores, y):
        self.model.fit(self._prepare(scores), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, scores):
        return self.model.predict_proba(self._prepare(scores))[:, 1]

