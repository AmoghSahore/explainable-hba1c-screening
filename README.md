# Explainable Screening for Elevated HbA1c

This repository contains the MCA 521-4 CIA 3 **ML for Social Good** project for Mission Health. The planned system uses NHANES August 2021-August 2023 data to screen for elevated HbA1c among respondents older than 18.

## Status

Phase 0 is complete: the repository structure and Git hygiene are in place. Data processing, modelling, explainability, ethics analysis, and the live demo will be implemented in later phases.

## Repository layout

- `data/raw/`: local, immutable NHANES XPT inputs; intentionally excluded from Git
- `data/processed/`: reproducible generated datasets; intentionally excluded from Git
- `docs/`: faculty brief and project documentation
- `notebooks/`: faculty-facing analysis notebook
- `src/`: reusable data, feature, training, evaluation, and explanation code
- `tests/`: automated validation and leakage checks
- `artifacts/`: generated model binaries and metrics; binary models are excluded from Git
- `reports/figures/`: final report-ready figures

## Data policy

The raw NHANES files are public data but are not committed to this repository. See `data/raw/README.md` for the required filenames and official sources. `SEQN` is used only to join source files and will not be used as a model feature.

