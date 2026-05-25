from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPS = 1e-7


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def calibration_intercept_slope(y_true, y_prob) -> tuple[float | None, float | None]:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return None, None
    x = logit(y_prob).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(x, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return None, None


def optimize_threshold(y_true, y_prob, metric: str = "f1") -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        if metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            score = sens + spec - 1.0
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else None,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ppv": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece_10bin": expected_calibration_error(y_true, y_prob, n_bins=10),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    out["calibration_intercept"] = intercept
    out["calibration_slope"] = slope
    return out


def bootstrap_auc_ci(y_true, y_prob, seed: int = 42, n_boot: int = 500) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"auc_ci_low": None, "auc_ci_high": None}
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return {"auc_ci_low": None, "auc_ci_high": None}
    return {
        "auc_ci_low": float(np.percentile(aucs, 2.5)),
        "auc_ci_high": float(np.percentile(aucs, 97.5)),
    }
