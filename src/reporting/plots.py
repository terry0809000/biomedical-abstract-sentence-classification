from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot_table(index="split", columns="label_name", values="n", fill_value=0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Sentences")
    ax.set_xlabel("Split")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_text_length_distribution(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for label, sub in df.groupby("label_name"):
        plt.hist(sub["word_count"], bins=60, alpha=0.55, label=label, density=True)
    plt.xlim(0, np.percentile(df["word_count"], 99))
    plt.xlabel("Word count")
    plt.ylabel("Density")
    plt.title("Sentence length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_roc(y_true, y_prob, out_path: Path, title: str) -> None:
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pr(y_true, y_prob, out_path: Path, title: str) -> None:
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion(y_true, y_prob, threshold: float, out_path: Path, title: str) -> None:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["other", "findings/conclusion"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_calibration(y_true, y_prob, out_path: Path, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Ideal")
    plt.plot(mean_pred, frac_pos, marker="o", color="#4C78A8", label="Observed")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

