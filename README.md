# Biomedical Abstract Sentence Classification

This repository contains cleaned reproducibility artifacts for a healthcare NLP study on sentence-level classification of biomedical abstracts. The task is to distinguish findings/conclusion sentences from other abstract sentences using CPU-feasible machine-learning models, with emphasis on transparent evaluation rather than deployment.

## Repository Scope

Included:

- `scripts/`: runnable experiment pipeline for downloading public datasets, preparing splits, training models, evaluating performance, calibration, external validation, and error/subgroup analysis.
- `src/`: reusable data parsing, feature extraction, model, calibration, evaluation, plotting, and utility modules.
- `results/figures/`: final diagnostic figures for class balance, sentence length, ROC, precision-recall, calibration, confusion matrix, and sentence-length subgroup performance.
- `results/tables/`: CSV versions of the locked result tables used for reporting model comparison, calibration, external validation, and subgroup diagnostics.
- `requirements.txt`: dependencies for the classical CPU pipeline.
- `requirements-neural.txt`: optional CPU PyTorch dependency for neural models.

Intentionally excluded:

- Word documents (`.docx`), compiled manuscripts (`.pdf`), and submission zip packages.
- Manuscript source files, cover letters, references, journal submission checklists, LaTeX folders, and paper-only folders.
- Downloaded raw/processed datasets, trained model artifacts, prediction dumps, run folders, and generated project-output folders.
- Unsafe biological protocol or operational guidance.

## Study Summary

The study evaluates lightweight classical and recurrent NLP models for identifying findings/conclusion sentences in biomedical abstracts. The primary corpus is PubMed-PICO with official train, validation, and test splits. PubMed 20k RCT is used as an external validation corpus with a compatible binary label mapping.

The reported workflow emphasizes:

- split-aware model development using official dataset partitions;
- preprocessing fitted only on training data;
- validation-based model selection;
- held-out internal test evaluation;
- external validation where available;
- calibration assessment using Brier score, expected calibration error, calibration slope/intercept, and reliability curves;
- sentence-length subgroup diagnostics;
- decision-threshold reporting through confusion-matrix and threshold-dependent metrics.

## Quick Start

Create an environment and install the classical CPU dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the core classical pipeline:

```bash
python scripts/run_core_pipeline.py
```

This command downloads public source data, prepares the binary task, trains the majority baseline plus TF-IDF logistic regression and linear SVM models, evaluates internal and external performance, writes calibration diagnostics, and produces subgroup/error-analysis outputs.

To also train the CPU neural models, install the optional PyTorch dependency and pass `--include-neural`:

```bash
python -m pip install -r requirements-neural.txt
python scripts/run_core_pipeline.py --include-neural
```

The neural path is intentionally optional because it can take substantially longer on CPU-only hardware.

## Pipeline Steps

The individual scripts can also be run in order:

```bash
python scripts/01_download_data.py
python scripts/02_prepare_data.py
python scripts/03_data_audit.py
python scripts/10_train_classical.py
python scripts/11_train_neural.py        # optional CPU neural models
python scripts/12_evaluate.py
python scripts/13_calibrate.py
python scripts/14_external_validation.py
python scripts/20_error_analysis.py
```

Generated files are written under `data/`, `runs/`, and `project_outputs/`; these are ignored by Git to keep the repository lightweight.

## Main Result Artifacts

The cleaned result tables include:

- `dataset_characteristics.csv`
- `model_comparison.csv`
- `calibration_metrics.csv`
- `external_validation.csv`
- `subgroup_performance.csv`

The figure artifacts include:

- `class_distribution.png`
- `text_length_distribution.png`
- `roc_curve_best_model.png`
- `pr_curve_best_model.png`
- `calibration_curve_best_model.png`
- `confusion_matrix_best_model.png`
- `subgroup_performance.png`

## Reproducibility Note

The repository now includes the runnable experimental code from the local CPU project package. Manuscript-generation and submission-packaging scripts were intentionally left out, because this repository is meant to expose the computational workflow rather than the paper files.

## Data Note

This project uses public biomedical NLP datasets. Source datasets should be obtained from their official providers rather than redistributed as substitutes for the original releases.

## License

Source code is released under the repository `LICENSE`. Manuscript or submission material is not included in this public computational package.

## Public Repository Release Layer

This repository has been prepared as a public, citable research-software
package rather than a notebook-only upload.

Key public-release files:

- `docs/CLINICAL_ALGORITHMICIAN_ANNOTATION.md`
- `docs/DEPLOYMENT_READINESS.md`
- `DATA_ACCESS.md`
- `.zenodo.json`
- `scripts/validate_repository.py`
- `.github/workflows/repository-publication.yml`
- `Dockerfile`

Validate the release package with:

```bash
python scripts/validate_repository.py
```
