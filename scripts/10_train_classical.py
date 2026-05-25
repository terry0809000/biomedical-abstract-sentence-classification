from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.platt import PlattCalibrator
from src.evaluation.metrics import binary_metrics, optimize_threshold
from src.features.text_vectorizers import build_word_char_tfidf, build_word_tfidf
from src.utils.paths import OUTPUTS_DIR, PROCESSED_DIR, RUNS_DIR, ensure_project_dirs
from src.utils.seed import set_seed


SEED = 42


def load_data():
    train = pd.read_csv(PROCESSED_DIR / "primary_train.csv")
    val = pd.read_csv(PROCESSED_DIR / "primary_validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "primary_test.csv")
    external = pd.read_csv(PROCESSED_DIR / "external_pubmed_rct20k_test.csv")
    return train, val, test, external


def write_run(
    model_name: str,
    config: dict,
    model,
    calibrator,
    threshold: float,
    frames: dict[str, pd.DataFrame],
    raw_scores: dict[str, np.ndarray],
    probs: dict[str, np.ndarray],
) -> None:
    run_dir = RUNS_DIR / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "seed.txt").write_text(str(config.get("seed", SEED)), encoding="utf-8")
    if model is not None:
        joblib.dump(model, run_dir / "model_artifact.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, run_dir / "calibration_artifact.joblib")

    lines = [f"model: {model_name}", f"threshold_from_validation: {threshold:.6f}"]
    for split, frame in frames.items():
        pred = frame[["dataset", "split", "abstract_id", "sentence_index", "source_label", "text", "label"]].copy()
        pred["raw_score"] = raw_scores[split]
        pred["probability"] = probs[split]
        pred["prediction"] = (pred["probability"] >= threshold).astype(int)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        if split in {"validation", "test"}:
            pred.to_csv(OUTPUTS_DIR / "07_predictions" / f"{model_name}_{split}.csv", index=False)
        metrics = binary_metrics(pred["label"].to_numpy(), pred["probability"].to_numpy(), threshold)
        (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        lines.append(f"{split}: roc_auc={metrics.get('roc_auc')} brier={metrics.get('brier_score')} ece={metrics.get('ece_10bin')}")
    (run_dir / "training_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    set_seed(SEED)
    train, val, test, external = load_data()
    frames = {"validation": val, "test": test, "external": external}
    y_train = train["label"].to_numpy()

    # Majority prevalence baseline.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train["text"], y_train)
    prior = float(y_train.mean())
    raw_scores = {split: np.full(len(df), prior) for split, df in frames.items()}
    probs = raw_scores.copy()
    threshold = 0.5
    write_run(
        "majority_baseline",
        {"model": "DummyClassifier(strategy=prior)", "seed": SEED, "calibration": "none"},
        dummy,
        None,
        threshold,
        frames,
        raw_scores,
        probs,
    )

    model_specs = [
        (
            "tfidf_logistic_regression",
            Pipeline(
                [
                    ("tfidf", build_word_char_tfidf()),
                    (
                        "clf",
                        LogisticRegression(
                            C=2.0,
                            solver="liblinear",
                            max_iter=1000,
                            class_weight=None,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "decision_function",
        ),
        (
            "tfidf_linear_svm",
            Pipeline(
                [
                    ("tfidf", build_word_char_tfidf()),
                    ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)),
                ]
            ),
            "decision_function",
        ),
    ]

    for model_name, pipeline, score_method in model_specs:
        pipeline.fit(train["text"], y_train)
        val_raw = getattr(pipeline, score_method)(val["text"])
        calibrator = PlattCalibrator(input_is_probability=False).fit(val_raw, val["label"].to_numpy())
        raw_scores = {}
        probs = {}
        for split, df in frames.items():
            raw = getattr(pipeline, score_method)(df["text"])
            raw_scores[split] = raw
            probs[split] = calibrator.predict_proba(raw)
        threshold = optimize_threshold(val["label"].to_numpy(), probs["validation"], metric="f1")
        config = {
            "model": model_name,
            "seed": SEED,
            "features": "TF-IDF word 1-2 grams plus character 3-5 grams",
            "calibration": "Platt sigmoid calibration fitted on validation scores",
            "threshold": "Optimized for validation F1 only",
        }
        write_run(model_name, config, pipeline, calibrator, threshold, frames, raw_scores, probs)
        print(f"completed {model_name}")


if __name__ == "__main__":
    main()

