<p align="center"><img src="assets/project-cover.png" width="100%" alt="Conceptual scientific artwork for Biomedical sentence classification" /></p>

# Biomedical abstract sentence classification: revised pipeline

**Clinical NLP · Research pipeline · Source-code release**

This project provides the author's existing analysis implementation for biomedical abstract sentence classification: revised pipeline. Scientific source files are preserved from the local research product. This upload organizes code and documentation; it does not represent new experiments, model retraining or an independently verified reproduction.

## Pipeline

1. **Acquire inputs.** Acquire the original PubMed RCT and external abstract corpus releases using the saved downloader. Raw texts, row-level predictions and checkpoints are excluded.
2. **Prepare the environment.** Use the project-specific dependency files listed below in an isolated environment. Preserve the versions and seeds recorded in the source release.
3. **Prepare data and run the model or analysis.** Follow the original execution notes and entry points below, preserving time order, participant grouping and fixed evaluation boundaries.
4. **Evaluate and summarize.** Use the included evaluation/analysis scripts. Write generated data, predictions, fitted objects and caches to a private working copy.

This is a source-only repository. Raw data, individual-level derivatives and fitted checkpoints are not bundled. Some saved-evidence reproduction commands in the original notes require artifacts from the original source release; obtain those independently where authorized.

## Environment

- [`revision_20260916/analysis/requirements.txt`](revision_20260916/analysis/requirements.txt)
- [`revision_20260916/environment_versions.json`](revision_20260916/environment_versions.json)

## Execution

See the [pipeline and command reference](PIPELINE.md) for entry points, required arguments and the source-specific input contract.

Run commands from the repository root unless the original notes specify another working directory. Use a fresh copy and configure local data paths before running.

```bash
python revision_20260916/analysis/run_reproduction.py fit
```

### Original execution notes

- [README.md](docs/original/revision_20260916/README.md)

### Scientific source inventory

<details>
<summary>Browse the preserved scientific implementation</summary>

- [`revision_20260916/analysis/check_statistics.py`](revision_20260916/analysis/check_statistics.py)
- [`revision_20260916/analysis/download_revision_data.py`](revision_20260916/analysis/download_revision_data.py)
- [`revision_20260916/analysis/evidence_io.py`](revision_20260916/analysis/evidence_io.py)
- [`revision_20260916/analysis/legacy_src/src/__init__.py`](revision_20260916/analysis/legacy_src/src/__init__.py)
- [`revision_20260916/analysis/legacy_src/src/data/__init__.py`](revision_20260916/analysis/legacy_src/src/data/__init__.py)
- [`revision_20260916/analysis/legacy_src/src/data/parsers.py`](revision_20260916/analysis/legacy_src/src/data/parsers.py)
- [`revision_20260916/analysis/legacy_src/src/evaluation/__init__.py`](revision_20260916/analysis/legacy_src/src/evaluation/__init__.py)
- [`revision_20260916/analysis/legacy_src/src/evaluation/metrics.py`](revision_20260916/analysis/legacy_src/src/evaluation/metrics.py)
- [`revision_20260916/analysis/legacy_src/src/features/__init__.py`](revision_20260916/analysis/legacy_src/src/features/__init__.py)
- [`revision_20260916/analysis/legacy_src/src/features/neural_tokenizer.py`](revision_20260916/analysis/legacy_src/src/features/neural_tokenizer.py)
- [`revision_20260916/analysis/make_figures.py`](revision_20260916/analysis/make_figures.py)
- [`revision_20260916/analysis/pin_source_versions.py`](revision_20260916/analysis/pin_source_versions.py)
- [`revision_20260916/analysis/predict_saved_model.py`](revision_20260916/analysis/predict_saved_model.py)
- [`revision_20260916/analysis/rebenchmark_inference.py`](revision_20260916/analysis/rebenchmark_inference.py)
- [`revision_20260916/analysis/recompute_run_metrics.py`](revision_20260916/analysis/recompute_run_metrics.py)
- [`revision_20260916/analysis/restore_prediction_archives.py`](revision_20260916/analysis/restore_prediction_archives.py)
- [`revision_20260916/analysis/revised_pipeline.py`](revision_20260916/analysis/revised_pipeline.py)
- [`revision_20260916/analysis/revised_statistics.py`](revision_20260916/analysis/revised_statistics.py)
- [`revision_20260916/analysis/run_reproduction.py`](revision_20260916/analysis/run_reproduction.py)

</details>

## Repository contents and rights

Only scientific code, dependencies, configuration and execution documentation are included. Manuscripts, submission documents, raw participant records, trained models, credentials, caches and software environments are excluded. Notebook cell outputs and execution counters are cleared for publication; code cells are retained.

Existing source and third-party license notices govern their respective files. No blanket license is invented for material whose original release did not specify one. Original dataset licenses and access requirements continue to apply.

Research software is not a clinical deployment claim. Negative findings and the source release's interpretation limits remain applicable. No tests or experiments were run during this upload, as requested by the author.
