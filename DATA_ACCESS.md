# Data Access

The pipeline downloads public NLP corpora from official providers. Downloaded datasets, trained models, prediction dumps, and run directories are excluded from Git.

## Repository Boundary

Public GitHub and Zenodo releases should include source code, notebooks,
aggregate tables, aggregate figures, documentation, manifests, and validation
metadata. They should not include raw restricted data, patient-level derived
extracts, private credentials, local caches, model binaries, or large
regenerated work directories.

## Reproducibility Pattern

1. Obtain the required datasets from their official providers.
2. Keep source archives and extracted row-level data outside this repository.
3. Configure local paths with environment variables or documented runtime
   settings.
4. Run the workflow entry point listed in `docs/DEPLOYMENT_READINESS.md`.
5. Commit only aggregate, de-identified, redistributable artifacts.
