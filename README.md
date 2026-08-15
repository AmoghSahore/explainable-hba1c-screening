# Explainable Screening for Elevated HbA1c

This repository contains the MCA 521-4 CIA 3 **ML for Social Good** Mission Health project. It uses NHANES August 2021-August 2023 data to estimate whether an adult respondent has elevated HbA1c (`LBXGH >= 5.7%`) from demographic, anthropometric, blood-pressure, physical-activity, and smoking information.

The output is an educational screening aid for prioritizing confirmatory tests. It is not a diagnosis, prognosis, or treatment recommendation.

## Repository layout

- `data/raw/`: local, immutable NHANES XPT inputs; intentionally excluded from Git
- `data/processed/`: reproducible generated datasets; intentionally excluded from Git
- `docs/`: faculty brief and project documentation
- `notebooks/`: faculty-facing analysis notebook
- `src/`: reusable data, training, evaluation, and explanation code
- `tests/`: automated validation and leakage checks
- `artifacts/`: generated model binaries and metrics; binary models are excluded from Git
- `reports/figures/`: final report-ready figures

## Data policy

The raw NHANES files are public data but are not committed to this repository. See `data/raw/README.md` for the required filenames and official sources. `SEQN` is used only to join source files and will not be used as a model feature.

## Reproduce the project

Python 3.11–3.14 is supported. From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the six XPT files listed in `data/raw/README.md` under `data/raw/`, then run:

```powershell
.\.venv\Scripts\python.exe -m src.data_pipeline
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train
```

The training command performs the reproducible 70/15/15 split, five-fold tuning, validation-only threshold selection, untouched test evaluation, bootstrap uncertainty analysis, subgroup auditing, SHAP generation, and model packaging. A faster engineering smoke run is available as `python -m src.train --quick` but must not be used for final reported results.

Launch the live demo after training:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the notebook with:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks\ML_CIA3.ipynb
```

## Required models

- Logistic Regression baseline
- Random Forest bagging model
- XGBoost boosting model
- Heterogeneous stack of Logistic Regression, Random Forest, and XGBoost with an out-of-fold Logistic Regression meta-learner

All imputers, outlier clipping, scaling, and categorical encoding are fitted inside cross-validation pipelines. The raw target and merge identifier never enter the predictor matrix.

## Documentation

- `docs/DATA_DICTIONARY.md` — variables and derivations
- `docs/ETHICS.md` — fairness, privacy, uncertainty, error costs, oversight, and limitations
- `docs/ACKNOWLEDGEMENTS.md` — dataset and software citations
- `docs/VIDEO_SCRIPT.md` — exact three-minute presentation sequence
- `docs/RUBRIC_CHECKLIST.md` — faculty requirement traceability
- `reports/MODEL_CARD.md` — generated final model and test results

## Generated outputs

Processed data and fitted model binaries are reproducible and ignored by Git. Final result tables, model cards, and figures under `reports/` are intended for submission. Large video files should be submitted separately.

