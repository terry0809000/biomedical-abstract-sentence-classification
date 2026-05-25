# Biomedical Abstract Sentence Classification

This repository contains cleaned reproducibility artifacts for a healthcare NLP study on sentence-level classification of biomedical abstracts. The task is to distinguish findings/conclusion sentences from other abstract sentences using CPU-feasible machine-learning models, with emphasis on transparent evaluation rather than deployment.

## Repository Scope

Included:

- `results/figures/`: final diagnostic figures for class balance, sentence length, ROC, precision-recall, calibration, confusion matrix, and sentence-length subgroup performance.
- `results/tables/`: CSV versions of the locked result tables used for reporting model comparison, calibration, external validation, and subgroup diagnostics.

Intentionally excluded:

- Word documents (`.docx`), compiled manuscripts (`.pdf`), and submission zip packages.
- Manuscript source files, cover letters, references, journal submission checklists, and paper-only folders.
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

The supplied local folder did not contain runnable model-training source code. For that reason, this repository preserves the clean result artifacts only and avoids uploading manuscript or submission files. If the experiment scripts are later available, add them under an `analysis/` or `src/` directory with dependency instructions and a deterministic run command.

## Data Note

This project uses public biomedical NLP datasets. Source datasets should be obtained from their official providers rather than redistributed as substitutes for the original releases.

## License

No license has been declared in the supplied materials. Unless a license is added by the repository owner, all rights are reserved by default.
