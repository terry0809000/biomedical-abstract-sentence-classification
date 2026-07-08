# Senior Clinical Algorithmician Annotation

## Clinical Algorithmic Frame

Repository: [Biomedical abstract sentence classification reproducibility workflow](https://github.com/terry0809000/biomedical-abstract-sentence-classification)

Domain: biomedical abstract sentence-level evidence classification.

Intended research use: public-dataset NLP reproducibility, model comparison, calibration analysis, and transparent error analysis.

Explicit non-use: automated clinical evidence grading, guideline recommendations, systematic-review exclusion without human review, or patient-specific decision support.

Primary analytic signal: PubMed-PICO splits, PubMed 20k RCT external validation, TF-IDF/logistic and linear SVM baselines, calibration, and subgroup diagnostics.

Primary validation claim: CPU-feasible sentence classification pipeline with optional neural branch kept separate from the locked classical pipeline.

Deployment gate: RESEARCH_ONLY_TEXT_ANALYTICS; human review is required for evidence synthesis or clinical use.

## Algorithmic Interpretation

This repository should be read as a clinical algorithm research package, not
as a deployable medical device. The model outputs are risk estimates or text
labels conditioned on the data-generating process used in the study. They are
not causal effects, treatment rules, or substitutes for clinician judgement.

The clinically important question is not only whether discrimination is high.
A senior review must ask whether the cohort definition is stable, whether the
outcome is observable before downstream care changes it, whether features are
available at the stated decision time, whether missingness reflects illness or
documentation practice, and whether calibration survives source shift.

## Minimum Evidence Standard

- Cohort construction must be locked before model selection.
- Feature availability must be anchored to the stated prediction time.
- Patient, stay, site, or state-week clustering must be respected in splits
  and uncertainty estimates.
- AUROC must be interpreted with AUPRC, Brier score, calibration-in-the-large,
  calibration slope/intercept, expected/observed ratios, and threshold burden.
- External or temporal validation must be labelled according to the actual
  data source and split design.
- Recalibration must be distinguished from frozen external validation.
- Explanation methods are diagnostic descriptions of the fitted model, not
  biological or clinical causality.

## Operational Guardrails

A real-world implementation would require a local feature map, data-refresh
checks, locked thresholds, subgroup and drift monitoring, incident response,
audit logs, model versioning, clinical owner signoff, and a prospective
silent-trial period. The package is therefore intentionally conservative:
where evidence is retrospective, sparse, poorly calibrated, or source-shifted,
the correct algorithmic state is blocked or research-only.

## Reviewer Checklist

- Does the README identify the intended use and the non-use cases?
- Can a reviewer run repository validation without credentialed data?
- Are raw datasets and patient-level derivatives absent from Git?
- Are local workstation paths removed from public files?
- Is Zenodo metadata present for citable public release?
- Are deployment claims separated from retrospective modelling results?
