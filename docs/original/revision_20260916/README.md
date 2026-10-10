# Biomedical abstract sentence classification — revision release

Author: Tianyi Yu. Version: 2026.09.16. Related manuscript: PONE-D-26-34455.

This package implements the revised analysis, including exclusions discovered during review. The July 2026 repository archive represents the earlier analysis and does not contain these revised experiments. No new public DOI is asserted by this package.

## Installation

The tested platform is Windows 11, Python 3.14.2, CPU PyTorch 2.12.0+cpu and four computational threads. Exact scientific dependencies are pinned in `analysis/requirements.txt`; see `environment_versions.json`. Other platforms are not certified by this release.

Extract the full reproducibility archive once; its `revision_20260916` folder is already runnable. If using only the journal supplements, first extract S4_Code.zip to obtain that folder. From the directory containing `revision_20260916`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r revision_20260916/analysis/requirements.txt --index-url https://pypi.org/simple
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe revision_20260916/analysis/check_statistics.py
```

All execution commands below may use `.venv\Scripts\python.exe` instead of `python`. The tested Windows PyPI wheel reports PyTorch `2.12.0+cpu`; CUDA is unavailable in that build. A separate clean virtual environment successfully installed all thirty pinned dependencies and passed `pip check` on this host. Input downloads require internet access; the analyses operate locally after download. Download URLs are pinned to upstream commits, the retained inputs were matched against Git blob identifiers, and SHA-256 hashes are checked before use; see `data/source_provenance.json`. Allow several GB of free disk space for uncompressed predictions, models and figures. The recorded host had about 31.5 GiB RAM; this is an observed environment, not a measured minimum.

## Two reproduction routes

1. **Replay saved predictions.** Put S5_Data.zip, S6_Data.zip and S7_Data.zip together, then run `python revision_20260916/analysis/run_reproduction.py replay --archives .`. This recreates numerical statistics and figures without claiming that model training was repeated. It uses original raw input files to reconstruct cohort metadata and length boundaries. Supplied coefficients are retained if trained weights are absent.
2. **Refit all models.** In a fresh extraction, run `python revision_20260916/analysis/run_reproduction.py fit`. This fits eight classical/control runs and twelve neural runs, measures inference serially, computes clustered uncertainty and recreates the figures. Fit duration depends on hardware. Do not use a prediction-replay extraction for this route: replay creates completion markers.

The individual commands and exact definitions are in `REPRODUCE.md` and `PROTOCOL_LOCK.md`. `COMPUTATIONAL_PROTOCOL.md` provides a step-by-step protocol suitable for manual entry into protocols.io if desired.

## Contents and provenance

- `analysis/`: executable pipeline, statistics, plotting, checkpoint inference, resampling verification and input downloader.
- `analysis/legacy_src/`: only the parser, tokenizer and metric compatibility modules used by this release. The revised recurrent implementation is in `revised_pipeline.py`; unused historical model implementations are excluded from this code package. Cluster-bootstrap intervals are computed by `revised_statistics.py`.
- `data/`: checksums, document assignments, exclusions and training-row selection; raw corpora are downloaded from their original public repositories.
- `runs/`: exact configurations, calibrators, epoch logs, metrics and observed costs.
- `tables/`: full-precision numerical evidence used in the manuscript and figures.
- S5–S7 Data: ordered sentence metadata and lossless score arrays for all twenty runs; row indices join arrays to sentence metadata.
- The full reproducibility archive already places the trained weights and preprocessing under `revision_20260916/runs/` and `revision_20260916/shared_artifacts/`. No manual merging is needed. See `analysis/predict_saved_model.py`. It also preserves the original S4–S7 ZIPs for correspondence with the journal supplements.

No private editorial correspondence or Word review history is included in public code materials. This is research software for source-derived sentence labels, with no clinical-use validation. Third-party corpus text remains subject to its original terms. Author-generated code is provided under the MIT license; see `LICENSE` and `CITATION.cff`.

## Expected checks

Release verification on the tested host completed the full prediction-replay command in a separately installed pinned virtual environment. All fifteen table CSVs matched the release byte for byte and all seven TIFF pixel arrays matched exactly. The lexical-coefficient table and recorded timing evidence were retained during replay; replay does not remeasure them or refit models. Separately reloaded logistic, feed-forward and BiLSTM models reproduced all 1,000 checked decisions in that clean environment. All twenty saved model runs had also passed checkpoint-reload decision checks during serial inference measurement in the analysis environment.

The revised internal test contains 31,254 sentences from 2,399 abstracts. The cross-corpus evaluation contains 28,168 sentences from 2,351 abstracts. Development has 1,200 model-selection, 600 calibration and 600 threshold-selection abstracts. The resampling verification must pass. The pipeline must produce twenty completed runs. Compare regenerated statistical tables against the supplied full-precision tables; small floating-point or stochastic differences across library/platform combinations require investigation, and fitting-time differences are expected. Replaying the retained predictions should preserve model decisions and reported rounded results.
