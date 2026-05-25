from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import OUTPUTS_DIR, RUNS_DIR, ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    rows = []
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        config_path = run_dir / "config.yaml"
        for split in ["validation", "test", "external"]:
            path = run_dir / f"metrics_{split}.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": run_dir.name,
                    "split": split,
                    "calibration_artifact": str(run_dir / "calibration_artifact.joblib")
                    if (run_dir / "calibration_artifact.joblib").exists()
                    else "",
                    "config": str(config_path) if config_path.exists() else "",
                    "brier_score": m.get("brier_score"),
                    "ece_10bin": m.get("ece_10bin"),
                    "calibration_intercept": m.get("calibration_intercept"),
                    "calibration_slope": m.get("calibration_slope"),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "06_tables" / "calibration_results.csv", index=False)
    print(OUTPUTS_DIR / "06_tables" / "calibration_results.csv")


if __name__ == "__main__":
    main()

