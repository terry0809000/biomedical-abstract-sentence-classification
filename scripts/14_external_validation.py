from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import bootstrap_auc_ci
from src.utils.paths import OUTPUTS_DIR, RUNS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    rows = []
    metrics = {}
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        path = run_dir / "metrics_external.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        pred_path = run_dir / "predictions_external.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            m.update(bootstrap_auc_ci(pred["label"], pred["probability"]))
        metrics[run_dir.name] = m
        rows.append({"model": run_dir.name, **m})
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "06_tables" / "external_validation_results.csv", index=False)
    (OUTPUTS_DIR / "04_metrics" / "external_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    section = [
        "# External Validation",
        "",
        "The external validation set is the official PubMed 20k RCT test split. It was not used for model fitting, hyperparameter selection, calibration fitting, or threshold selection.",
        "",
        "Positive labels were RESULTS and CONCLUSIONS; all other PubMed RCT section labels were mapped to the negative class. This is label-compatible with the primary PubMed-PICO task only for the rhetorical finding/conclusion construct and should not be interpreted as validation of PICO Population, Intervention, or Outcome extraction.",
        "",
        "```text",
        pd.DataFrame(rows).to_string(index=False) if rows else "No external validation rows were available.",
        "```",
    ]
    out_dir = OUTPUTS_DIR / "10_manuscript"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "external_validation_section.md").write_text("\n".join(section), encoding="utf-8")
    print(OUTPUTS_DIR / "04_metrics" / "external_validation_metrics.json")


if __name__ == "__main__":
    main()
