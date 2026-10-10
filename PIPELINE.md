# Execution interfaces — Biomedical abstract sentence classification: revised pipeline

This guide describes the preserved source interfaces by reading the code. No research module was imported, no command was run, and no performance or portability validation is implied. Use the original project instructions for the scientific run order.

## Data → analysis → outputs

**Input contract:** Acquire the original PubMed RCT and external abstract corpus releases using the saved downloader. Raw texts, row-level predictions and checkpoints are excluded.

1. Acquire inputs from their original providers and configure the paths defined in the source.
2. Install the recorded dependencies in an isolated environment.
3. Preserve grouping, seeds, development/evaluation separation and any protocol/hash gates.
4. Run acquisition/extraction and preprocessing before model development. Run evaluation only after the selection/freeze required by the original implementation.
5. Keep raw data, row-level derivatives, predictions and fitted models outside the public repository.

## Project entry sequence

```powershell
python revision_20260916/analysis/run_reproduction.py fit
```

These commands are documented from the saved project interfaces; they were not executed for this release. Source-specific preconditions still apply.

## Script reference

### `revision_20260916/analysis/check_statistics.py`

[Open source](revision_20260916/analysis/check_statistics.py) · Supporting module

Validate weighted bootstrap calculations against explicit resampling.

### `revision_20260916/analysis/download_revision_data.py`

[Open source](revision_20260916/analysis/download_revision_data.py) · Supporting module

Download exact original public files and require their archived SHA-256 hashes.

### `revision_20260916/analysis/evidence_io.py`

[Open source](revision_20260916/analysis/evidence_io.py) · Supporting module

Read archived floating-point evidence without avoidable decimal conversion drift.

### `revision_20260916/analysis/make_figures.py`

[Open source](revision_20260916/analysis/make_figures.py) · Supporting module

All figures are deterministic plots of revised data, not generated illustrations.

### `revision_20260916/analysis/pin_source_versions.py`

[Open source](revision_20260916/analysis/pin_source_versions.py) · Supporting module

Verify retained inputs against immutable upstream Git blobs and record provenance. Release preparation helper; ordinary reproduction uses download_revision_data.py. No authentication token is used or needed.

### `revision_20260916/analysis/predict_saved_model.py`

[Open source](revision_20260916/analysis/predict_saved_model.py) · Executable entry point

Load a trained revision model and reproduce scores from a sentence CSV. Usage: python predict_saved_model.py ROOT RUN INPUT.csv OUTPUT.csv ROOT contains runs/ and shared_artifacts/. Neural and classical files are trusted author-generated archives; do not unpickle unrelated downloaded artifacts.

### `revision_20260916/analysis/rebenchmark_inference.py`

[Open source](revision_20260916/analysis/rebenchmark_inference.py) · Supporting module

Serial inference remeasurement after every fitting process has completed.

### `revision_20260916/analysis/recompute_run_metrics.py`

[Open source](revision_20260916/analysis/recompute_run_metrics.py) · Supporting module

Unify metrics on the exact archived probabilities, without refitting models. The historical helper clipped all probabilities to [1e-7, 1-1e-7]. The revision uses full saved precision for evaluation and clips only when taking logits for the descriptive calibration intercept/slope. This cannot change any model fit.

### `revision_20260916/analysis/restore_prediction_archives.py`

[Open source](revision_20260916/analysis/restore_prediction_archives.py) · Supporting module

Restore precise run CSVs from S5/S6/S7 Data for metric/figure reproduction.

### `revision_20260916/analysis/revised_pipeline.py`

[Open source](revision_20260916/analysis/revised_pipeline.py) · Executable entry point

Revision pipeline with disjoint selection/calibration/threshold documents. Scientific runtime: Python with numpy pandas scipy sklearn torch joblib psutil. All outputs are new files; the original study is never overwritten.

| Argument | Required | Default / allowed values |
|---|---|---|
| `stage` | No / positional | ; choices: ['prepare','classical','neural'] |

### `revision_20260916/analysis/revised_statistics.py`

[Open source](revision_20260916/analysis/revised_statistics.py) · Executable entry point

Statistics from fixed revision predictions; no model fitting or test tuning.

### `revision_20260916/analysis/run_reproduction.py`

[Open source](revision_20260916/analysis/run_reproduction.py) · Executable entry point

One entry point for refitting or replaying the published prediction evidence.

| Argument | Required | Default / allowed values |
|---|---|---|
| `mode` | No / positional | ; choices: ['fit', 'replay'] |
| `--archives` | No / positional | default: Path.cwd() |

## Reproduction limits

Absolute paths, frozen-file references and platform-specific assumptions are preserved. Configure a matching local workspace rather than bypassing scientific integrity guards. A repository containing existing scripts is not evidence that every data or environment prerequisite is satisfied. No original clinical results or completed-run artifacts are redistributed.
