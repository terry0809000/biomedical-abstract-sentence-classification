from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(script_name: str) -> None:
    script_path = ROOT / "scripts" / script_name
    print(f"\n=== {script_name} ===", flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible biomedical sentence-classification pipeline.")
    parser.add_argument(
        "--include-neural",
        action="store_true",
        help="Also train CPU neural models. This can take substantially longer than the classical baseline path.",
    )
    args = parser.parse_args()

    steps = [
        "01_download_data.py",
        "02_prepare_data.py",
        "03_data_audit.py",
        "10_train_classical.py",
    ]
    if args.include_neural:
        steps.append("11_train_neural.py")
    steps.extend(
        [
            "12_evaluate.py",
            "13_calibrate.py",
            "14_external_validation.py",
            "20_error_analysis.py",
        ]
    )

    for step in steps:
        run_step(step)


if __name__ == "__main__":
    main()
