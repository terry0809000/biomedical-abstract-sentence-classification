from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parsers import class_distribution
from src.reporting.plots import save_class_distribution, save_text_length_distribution
from src.utils.paths import OUTPUTS_DIR, PROCESSED_DIR, ensure_project_dirs


def summarize_lengths(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "split", "label_name"])["word_count"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )


def main() -> None:
    ensure_project_dirs()
    primary = pd.read_csv(PROCESSED_DIR / "primary_all.csv")
    external = pd.read_csv(PROCESSED_DIR / "external_pubmed_rct20k_test.csv")
    all_df = pd.concat([primary, external], ignore_index=True)

    dist = class_distribution(all_df, ["dataset", "split"])
    dist.to_csv(OUTPUTS_DIR / "06_tables" / "class_distribution.csv", index=False)
    lengths = summarize_lengths(all_df)
    lengths.to_csv(OUTPUTS_DIR / "06_tables" / "text_length_summary.csv", index=False)

    save_class_distribution(dist, OUTPUTS_DIR / "05_figures" / "class_distribution.png")
    save_text_length_distribution(all_df, OUTPUTS_DIR / "05_figures" / "text_length_distribution.png")

    audit = [
        "# Data Audit",
        "",
        "## Task",
        "Binary sentence-level classification: positive if a sentence belongs to a results/findings or conclusion section, negative otherwise.",
        "",
        "## Leakage Controls",
        "- Official PubMed-PICO train/dev/test splits are preserved.",
        "- Vectorizers and neural vocabularies are fitted on the primary training split only.",
        "- The PubMed 20k RCT external test set is loaded separately and is not used for model selection, threshold selection, or calibration fitting.",
        "- Raw files are kept under `data/raw`; processed CSV files are written under `data/processed`.",
        "",
        "## Class Distribution",
        "```text",
        dist.to_string(index=False),
        "```",
        "",
        "## Text Length Summary",
        "```text",
        lengths.to_string(index=False),
        "```",
    ]
    out = OUTPUTS_DIR / "02_data" / "data_audit.md"
    out.write_text("\n".join(audit), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
