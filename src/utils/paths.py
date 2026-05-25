from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = ROOT / "runs"
ARTIFACTS_DIR = RUNS_DIR / "artifacts"
OUTPUTS_DIR = ROOT / "project_outputs"


def ensure_project_dirs() -> None:
    dirs = [
        OUTPUTS_DIR / "00_project_selection",
        OUTPUTS_DIR / "01_environment",
        OUTPUTS_DIR / "02_data",
        OUTPUTS_DIR / "03_experiments",
        OUTPUTS_DIR / "04_metrics",
        OUTPUTS_DIR / "05_figures",
        OUTPUTS_DIR / "06_tables",
        OUTPUTS_DIR / "07_predictions",
        OUTPUTS_DIR / "08_literature",
        OUTPUTS_DIR / "09_journal_selection",
        OUTPUTS_DIR / "10_manuscript",
        OUTPUTS_DIR / "11_latex_zip",
        OUTPUTS_DIR / "12_failure_reports",
        RAW_DIR,
        PROCESSED_DIR,
        ARTIFACTS_DIR,
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)

