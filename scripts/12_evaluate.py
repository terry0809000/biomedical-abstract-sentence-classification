from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import binary_metrics, bootstrap_auc_ci
from src.reporting.plots import save_calibration, save_confusion, save_pr, save_roc
from src.utils.paths import OUTPUTS_DIR, RUNS_DIR, ensure_project_dirs


def load_metrics() -> tuple[pd.DataFrame, dict]:
    rows = []
    all_metrics = {}
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "metrics_validation.json").exists()):
        model_metrics = {}
        for split in ["validation", "test", "external"]:
            path = run_dir / f"metrics_{split}.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            model_metrics[split] = m
            row = {"model": run_dir.name, "split": split, **m}
            rows.append(row)
        all_metrics[run_dir.name] = model_metrics
    return pd.DataFrame(rows), all_metrics


def main() -> None:
    ensure_project_dirs()
    metrics_df, all_metrics = load_metrics()
    if metrics_df.empty:
        raise RuntimeError("No run metrics found. Train models first.")
    (OUTPUTS_DIR / "04_metrics" / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    metrics_df.to_csv(OUTPUTS_DIR / "06_tables" / "main_results.csv", index=False)

    cal_cols = [
        "model",
        "split",
        "n",
        "prevalence",
        "brier_score",
        "ece_10bin",
        "calibration_intercept",
        "calibration_slope",
    ]
    metrics_df[[c for c in cal_cols if c in metrics_df.columns]].to_csv(
        OUTPUTS_DIR / "06_tables" / "calibration_results.csv", index=False
    )
    per_class = []
    for _, row in metrics_df.iterrows():
        per_class.append(
            {
                "model": row["model"],
                "split": row["split"],
                "class": "findings_conclusion",
                "precision": row.get("precision_ppv"),
                "recall": row.get("sensitivity_recall"),
                "f1": row.get("f1"),
            }
        )
        per_class.append(
            {
                "model": row["model"],
                "split": row["split"],
                "class": "other",
                "specificity": row.get("specificity"),
            }
        )
    pd.DataFrame(per_class).to_csv(OUTPUTS_DIR / "06_tables" / "per_class_results.csv", index=False)

    validation = metrics_df[metrics_df["split"] == "validation"].sort_values(["roc_auc", "brier_score"], ascending=[False, True])
    best_model = str(validation.iloc[0]["model"])
    best_dir = RUNS_DIR / best_model
    best_test_pred = pd.read_csv(best_dir / "predictions_test.csv")
    test_metrics = binary_metrics(best_test_pred["label"], best_test_pred["probability"], float(validation.iloc[0]["threshold"]))
    test_metrics.update(bootstrap_auc_ci(best_test_pred["label"], best_test_pred["probability"]))
    all_metrics[best_model]["test_auc_ci"] = test_metrics
    (OUTPUTS_DIR / "04_metrics" / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    save_roc(best_test_pred["label"], best_test_pred["probability"], OUTPUTS_DIR / "05_figures" / "roc_curve_best_model.png", f"ROC curve: {best_model}")
    save_pr(best_test_pred["label"], best_test_pred["probability"], OUTPUTS_DIR / "05_figures" / "pr_curve_best_model.png", f"Precision-recall curve: {best_model}")
    save_calibration(best_test_pred["label"], best_test_pred["probability"], OUTPUTS_DIR / "05_figures" / "calibration_curve_best_model.png", f"Calibration curve: {best_model}")
    save_confusion(
        best_test_pred["label"],
        best_test_pred["probability"],
        float(validation.iloc[0]["threshold"]),
        OUTPUTS_DIR / "05_figures" / "confusion_matrix_best_model.png",
        f"Confusion matrix: {best_model}",
    )

    test_auc = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["split"] == "test")]["roc_auc"].iloc[0]
    val_auc = validation.iloc[0]["roc_auc"]
    external_auc = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["split"] == "external")]["roc_auc"].iloc[0]
    if test_auc >= 0.70:
        report = [
            "# AUC Target Achieved",
            "",
            f"- Model name: {best_model}",
            f"- Validation AUC: {val_auc:.6f}",
            f"- Test AUC: {test_auc:.6f}",
            f"- External AUC: {external_auc:.6f}",
            f"- Test AUC 95% bootstrap CI: {test_metrics.get('auc_ci_low'):.6f} to {test_metrics.get('auc_ci_high'):.6f}",
            f"- Test Brier score: {metrics_df[(metrics_df['model'] == best_model) & (metrics_df['split'] == 'test')]['brier_score'].iloc[0]:.6f}",
            f"- Test ECE: {metrics_df[(metrics_df['model'] == best_model) & (metrics_df['split'] == 'test')]['ece_10bin'].iloc[0]:.6f}",
            "",
            "The AUC target is empirical and was met by the selected validation-best model without using the test or external set for training, model selection, calibration fitting, or threshold selection.",
        ]
        (OUTPUTS_DIR / "04_metrics" / "auc_target_achieved.md").write_text("\n".join(report), encoding="utf-8")
    else:
        report = [
            "# AUC Target Not Achieved",
            "",
            f"- Best validation model: {best_model}",
            f"- Best achieved validation AUC: {val_auc:.6f}",
            f"- Best achieved test AUC: {test_auc:.6f}",
            f"- Best achieved external AUC: {external_auc:.6f}",
            "- Attempted improvements: word and character TF-IDF, class-weighted linear SVM, recurrent neural networks, validation-only calibration, and validation-only threshold optimization.",
            "- Likely reasons: label shift between source corpora, limited CPU neural training budget, and section-heading-derived labels that may not fully match rhetorical content.",
        ]
        (OUTPUTS_DIR / "12_failure_reports" / "auc_target_not_achieved.md").write_text("\n".join(report), encoding="utf-8")
    print(f"best_model={best_model}")


if __name__ == "__main__":
    main()

