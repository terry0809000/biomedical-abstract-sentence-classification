# Deployment Readiness and Public Release Guide

## Status

Current status: RESEARCH_ONLY_TEXT_ANALYTICS; human review is required for evidence synthesis or clinical use.

This repository is deployable as a public research package: it has documented
inputs, reproducibility commands, validation checks, citation metadata, and a
no-raw-data policy. It is not cleared for clinical use.

## Quick Validation

```bash
python scripts/validate_repository.py
```

Optional container validation:

```bash
docker build -t biomedical-abstract-sentence-classification:validation .
docker run --rm biomedical-abstract-sentence-classification:validation
```

To install the full analysis dependencies inside the image, build with:

```bash
docker build --build-arg INSTALL_ANALYSIS_DEPS=true -t biomedical-abstract-sentence-classification:analysis .
```

## Main Reproducibility Entry Point

```bash
python scripts/run_core_pipeline.py
```

Credentialed or large datasets must be configured outside Git according to
`DATA_ACCESS.md`.

## Release Checklist

- Public GitHub visibility confirmed.
- `README.md`, `CITATION.cff`, `.zenodo.json`, and `DATA_ACCESS.md` present.
- `scripts/validate_repository.py` passes locally and in GitHub Actions.
- No raw credentialed data, model binaries, compressed archives, or local
  workstation paths are tracked.
- A GitHub release is created from the validated commit.
- Zenodo GitHub integration or Zenodo API deposit archives that release.

## Clinical Safety Boundary

Before any operational pilot, a site must perform local data-mapping QA,
prospective silent validation, calibration review, threshold selection,
subgroup safety review, workflow simulation, governance approval, monitoring
design, and rollback planning. Until those are complete, all outputs remain
retrospective research artifacts.
