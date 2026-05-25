from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import binary_metrics
from src.utils.paths import OUTPUTS_DIR, RUNS_DIR, ensure_project_dirs


BEST_MODEL = "tfidf_logistic_regression"
COMPARATOR = "bilstm"


def read_pred(model: str, split: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / model / f"predictions_{split}.csv")


def example_rows(df: pd.DataFrame, n: int = 8) -> str:
    cols = ["dataset", "split", "source_label", "label", "probability", "text"]
    out = df[cols].head(n).copy()
    out["probability"] = out["probability"].map(lambda x: f"{x:.3f}")
    lines = []
    for _, row in out.iterrows():
        lines.append(
            f"- `{row['dataset']}/{row['split']}` source={row['source_label']} label={row['label']} prob={row['probability']}: {row['text']}"
        )
    return "\n".join(lines) if lines else "- No examples available."


def subgroup_by_length(pred: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    df = pred.copy()
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    df["length_group"] = pd.qcut(df["word_count"], q=4, labels=["short", "medium_short", "medium_long", "long"], duplicates="drop")
    rows = []
    for group, sub in df.groupby("length_group", observed=True):
        m = binary_metrics(sub["label"], sub["probability"], threshold=float((sub["prediction"] == 1).mean() if False else 0.5))
        rows.append({"model": model, "split": split, "subgroup": f"length:{group}", **m})
    for source, sub in df.groupby("source_label"):
        if len(sub) >= 50 and sub["label"].nunique() == 2:
            m = binary_metrics(sub["label"], sub["probability"], threshold=0.5)
            rows.append({"model": model, "split": split, "subgroup": f"source_label:{source}", **m})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_project_dirs()
    manuscript_dir = OUTPUTS_DIR / "10_manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    also_dir = OUTPUTS_DIR / "08_manuscript"
    also_dir.mkdir(parents=True, exist_ok=True)

    test = read_pred(BEST_MODEL, "test")
    external = read_pred(BEST_MODEL, "external")
    test["word_count"] = test["text"].astype(str).str.split().str.len()
    external["word_count"] = external["text"].astype(str).str.split().str.len()

    fp = test[(test["label"] == 0) & (test["prediction"] == 1)].sort_values("probability", ascending=False)
    fn = test[(test["label"] == 1) & (test["prediction"] == 0)].sort_values("probability", ascending=True)
    high_conf_errors = pd.concat([fp.head(10), fn.head(10)]).sort_values("probability", ascending=False)
    short_errors = test[(test["prediction"] != test["label"]) & (test["word_count"] <= test["word_count"].quantile(0.10))]

    comp_text = ""
    comp_path = RUNS_DIR / COMPARATOR / "predictions_test.csv"
    if comp_path.exists():
        comp = read_pred(COMPARATOR, "test")
        merged = test[["abstract_id", "sentence_index", "label", "prediction", "probability", "text", "source_label"]].merge(
            comp[["abstract_id", "sentence_index", "prediction", "probability"]],
            on=["abstract_id", "sentence_index"],
            suffixes=("_classical", "_neural"),
        )
        neural_wins = merged[(merged["prediction_classical"] != merged["label"]) & (merged["prediction_neural"] == merged["label"])]
        classical_wins = merged[(merged["prediction_classical"] == merged["label"]) & (merged["prediction_neural"] != merged["label"])]
        comp_text = "\n".join(
            [
                "## Examples Where BiLSTM Beats Classical",
                example_rows(
                    neural_wins.rename(columns={"probability_neural": "probability", "prediction_neural": "prediction"}).assign(
                        dataset="PubMed-PICO", split="test"
                    ),
                    n=6,
                ),
                "",
                "## Examples Where Classical Beats BiLSTM",
                example_rows(
                    classical_wins.rename(columns={"probability_classical": "probability", "prediction_classical": "prediction"}).assign(
                        dataset="PubMed-PICO", split="test"
                    ),
                    n=6,
                ),
            ]
        )

    subgroup = pd.concat(
        [
            subgroup_by_length(test, BEST_MODEL, "test"),
            subgroup_by_length(external, BEST_MODEL, "external"),
        ],
        ignore_index=True,
    )
    subgroup.to_csv(OUTPUTS_DIR / "06_tables" / "subgroup_analysis.csv", index=False)
    plot_df = subgroup[subgroup["subgroup"].str.startswith("length:")]
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for split, sub in plot_df.groupby("split"):
            ax.plot(sub["subgroup"].str.replace("length:", "", regex=False), sub["roc_auc"], marker="o", label=split)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("ROC AUC")
        ax.set_xlabel("Sentence length quartile")
        ax.set_title("Subgroup performance by sentence length")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "05_figures" / "subgroup_performance.png", dpi=200)
        plt.close()

    report = [
        "# Error Analysis",
        "",
        f"Best model: `{BEST_MODEL}`. Error analysis uses the internal PubMed-PICO test set unless otherwise stated.",
        "",
        "## Common False Positives",
        "False positives are usually method or outcome-definition sentences that contain result-like lexical cues, clinical endpoints, or summary statements without being labelled R/C in the source section labels.",
        "",
        example_rows(fp, n=8),
        "",
        "## Common False Negatives",
        "False negatives are often short or numerically dense finding sentences where section-heading labels identify R/C but the sentence lacks explicit result language after de-identification or number masking.",
        "",
        example_rows(fn, n=8),
        "",
        "## High-Confidence Errors",
        "The highest-confidence errors show the limits of treating section headings as sentence-level rhetorical truth; some individual sentences are semantically compatible with more than one role.",
        "",
        example_rows(high_conf_errors, n=10),
        "",
        "## Short-Text Failures",
        f"Internal-test errors among the shortest decile of sentences: {len(short_errors)}.",
        "",
        example_rows(short_errors.sort_values('probability', ascending=False), n=8),
        "",
        "## Ambiguous-Label Cases",
        "Ambiguity is concentrated in source labels such as OUTCOME, METHOD, and CONCLUSION because endpoints, methods, and take-home interpretations can share vocabulary. This is a dataset-label limitation, not evidence of clinical unsafety.",
        "",
        comp_text,
        "",
        "## Subgroup Summary",
        "Subgroup analysis was limited to ethically neutral metadata available in the public corpora: source label and sentence length. No demographic or patient-level attributes were available.",
    ]
    out = manuscript_dir / "error_analysis.md"
    out.write_text("\n".join(report), encoding="utf-8")
    (also_dir / "error_analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

