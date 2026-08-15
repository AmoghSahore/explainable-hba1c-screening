"""Project-wide paths, schema definitions, and reproducibility settings."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / ".cache"

for directory in (
    PROCESSED_DATA_DIR,
    ARTIFACT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    CACHE_DIR / "matplotlib",
):
    directory.mkdir(parents=True, exist_ok=True)

# Keep runtime caches inside the ignored project directory.
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))

RANDOM_STATE = 42
HBA1C_THRESHOLD = 5.7
EXPECTED_COHORT_SIZE = 5_872
EXPECTED_POSITIVE_COUNT = 2_300

RAW_FILES = {
    "demo": "DEMO_L.xpt",
    "bmx": "BMX_L.xpt",
    "bpxo": "BPXO_L.xpt",
    "paq": "PAQ_L.xpt",
    "smq": "SMQ_L.xpt",
    "ghb": "GHB_L.xpt",
}

TARGET_COLUMN = "elevated_hba1c"

NUMERIC_FEATURES = [
    "age",
    "income_ratio",
    "bmi",
    "waist_cm",
    "mean_systolic_bp",
    "mean_diastolic_bp",
    "moderate_activity_min_week",
    "vigorous_activity_min_week",
    "sedentary_min_day",
]

CATEGORICAL_FEATURES = [
    "sex",
    "education",
    "smoking_history",
    "current_smoking",
]

CATEGORY_LEVELS = {
    "sex": ["Female", "Male"],
    "education": [
        "Less than 9th grade",
        "9-11th grade/no diploma",
        "High school/GED",
        "Some college/AA",
        "College graduate or above",
    ],
    "smoking_history": ["Never", "Ever"],
    "current_smoking": ["Not current", "Current"],
}

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DISPLAY_NAMES = {
    "age": "Age (years)",
    "income_ratio": "Family income-to-poverty ratio",
    "bmi": "BMI (kg/m²)",
    "waist_cm": "Waist circumference (cm)",
    "mean_systolic_bp": "Mean systolic BP (mmHg)",
    "mean_diastolic_bp": "Mean diastolic BP (mmHg)",
    "moderate_activity_min_week": "Moderate activity (min/week)",
    "vigorous_activity_min_week": "Vigorous activity (min/week)",
    "sedentary_min_day": "Sedentary time (min/day)",
    "sex": "Sex recorded by NHANES",
    "education": "Education",
    "smoking_history": "Smoking history",
    "current_smoking": "Current smoking",
}
