"""Build the adult NHANES analytic cohort from immutable XPT source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    EXPECTED_COHORT_SIZE,
    EXPECTED_POSITIVE_COUNT,
    FEATURE_COLUMNS,
    HBA1C_THRESHOLD,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_FILES,
    TARGET_COLUMN,
)


REQUIRED_COLUMNS = {
    "demo": ["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "INDFMPIR"],
    "bmx": ["SEQN", "BMXBMI", "BMXWAIST"],
    "bpxo": [
        "SEQN",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
    ],
    "paq": [
        "SEQN",
        "PAD790Q",
        "PAD790U",
        "PAD800",
        "PAD810Q",
        "PAD810U",
        "PAD820",
        "PAD680",
    ],
    "smq": ["SEQN", "SMQ020", "SMQ040"],
    "ghb": ["SEQN", "LBXGH"],
}

EDUCATION_LABELS = {
    1.0: "Less than 9th grade",
    2.0: "9-11th grade/no diploma",
    3.0: "High school/GED",
    4.0: "Some college/AA",
    5.0: "College graduate or above",
}


def read_source_tables(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read and validate the six required NHANES transport files."""
    tables: dict[str, pd.DataFrame] = {}
    for source, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required NHANES file: {path}")
        frame = pd.read_sas(path, format="xport", encoding="utf-8")
        missing = sorted(set(REQUIRED_COLUMNS[source]) - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")
        if frame["SEQN"].isna().any():
            raise ValueError(f"{filename} contains a missing SEQN")
        if frame["SEQN"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate SEQN values")
        tables[source] = frame.loc[:, REQUIRED_COLUMNS[source]].copy()
    return tables


def _clean_numeric(series: pd.Series, invalid_codes: tuple[float, ...] = ()) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").astype(float)
    # pandas' XPORT reader can expose SAS/IBM zero as ~5.4e-79. No NHANES
    # measurement in this project has a legitimate magnitude this small.
    cleaned = cleaned.mask(cleaned.abs().lt(1e-70), 0.0)
    if invalid_codes:
        cleaned = cleaned.mask(cleaned.isin(invalid_codes))
    return cleaned


def derive_weekly_activity(
    frequency: pd.Series,
    unit: pd.Series,
    minutes_each_time: pd.Series,
) -> pd.Series:
    """Convert NHANES leisure-time activity responses to minutes per week."""
    frequency = _clean_numeric(frequency, (7777.0, 9999.0))
    minutes = _clean_numeric(minutes_each_time, (7777.0, 9999.0))
    normalized_unit = unit.astype("string").str.strip().str.upper()
    factors = normalized_unit.map({"D": 7.0, "W": 1.0, "M": 12.0 / 52.0, "Y": 1.0 / 52.0})
    weekly = frequency * minutes * factors
    weekly = weekly.mask(frequency.eq(0), 0.0)
    # A week contains 10,080 minutes; higher reports are not physically possible.
    return weekly.mask((weekly < 0) | (weekly > 10_080))


def derive_smoking(smq020: pd.Series, smq040: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Respect the adult smoking questionnaire skip pattern."""
    ever_raw = _clean_numeric(smq020, (7.0, 9.0))
    current_raw = _clean_numeric(smq040, (7.0, 9.0))

    ever = pd.Series(pd.NA, index=ever_raw.index, dtype="string")
    ever.loc[ever_raw.eq(1)] = "Ever"
    ever.loc[ever_raw.eq(2)] = "Never"

    current = pd.Series(pd.NA, index=ever_raw.index, dtype="string")
    current.loc[ever_raw.eq(2)] = "Not current"
    current.loc[ever_raw.eq(1) & current_raw.isin([1.0, 2.0])] = "Current"
    current.loc[ever_raw.eq(1) & current_raw.eq(3)] = "Not current"
    return ever, current


def _mask_outside(series: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = _clean_numeric(series)
    return numeric.mask((numeric < lower) | (numeric > upper))


def build_analytic_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    *,
    enforce_expected_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge sources, apply age/target eligibility, and derive approved features."""
    tables = read_source_tables(raw_dir)
    source_rows = {source: int(len(frame)) for source, frame in tables.items()}

    cohort = tables["demo"].loc[tables["demo"]["RIDAGEYR"].gt(18)].copy()
    adults_before_exam_merge = int(len(cohort))
    for source in ("bmx", "bpxo", "paq", "smq", "ghb"):
        cohort = cohort.merge(
            tables[source],
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

    cohort["LBXGH"] = _clean_numeric(cohort["LBXGH"])
    cohort = cohort.loc[cohort["LBXGH"].notna()].copy()

    output = pd.DataFrame(index=cohort.index)
    output["age"] = _mask_outside(cohort["RIDAGEYR"], 19, 80)
    output["income_ratio"] = _mask_outside(cohort["INDFMPIR"], 0, 5)
    output["bmi"] = _mask_outside(cohort["BMXBMI"], 10, 100)
    output["waist_cm"] = _mask_outside(cohort["BMXWAIST"], 40, 250)

    systolic = cohort[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].apply(
        pd.to_numeric, errors="coerce"
    )
    diastolic = cohort[["BPXODI1", "BPXODI2", "BPXODI3"]].apply(
        pd.to_numeric, errors="coerce"
    )
    output["mean_systolic_bp"] = _mask_outside(systolic.mean(axis=1), 50, 300)
    output["mean_diastolic_bp"] = _mask_outside(diastolic.mean(axis=1), 20, 200)

    output["moderate_activity_min_week"] = derive_weekly_activity(
        cohort["PAD790Q"], cohort["PAD790U"], cohort["PAD800"]
    )
    output["vigorous_activity_min_week"] = derive_weekly_activity(
        cohort["PAD810Q"], cohort["PAD810U"], cohort["PAD820"]
    )
    output["sedentary_min_day"] = _clean_numeric(
        cohort["PAD680"], (7777.0, 9999.0)
    ).mask(lambda values: (values < 0) | (values > 1_440))

    output["sex"] = cohort["RIAGENDR"].map({1.0: "Male", 2.0: "Female"}).astype("string")
    education = _clean_numeric(cohort["DMDEDUC2"], (7.0, 9.0))
    output["education"] = education.map(EDUCATION_LABELS).astype("string")
    output["smoking_history"], output["current_smoking"] = derive_smoking(
        cohort["SMQ020"], cohort["SMQ040"]
    )
    output[TARGET_COLUMN] = cohort["LBXGH"].ge(HBA1C_THRESHOLD).astype(int)

    # scikit-learn expects ordinary object/NaN values rather than pandas.NA.
    for column in ("sex", "education", "smoking_history", "current_smoking"):
        output[column] = output[column].astype(object).where(output[column].notna(), np.nan)

    output = output.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)
    if output[FEATURE_COLUMNS].duplicated().all():
        raise ValueError("Feature construction produced an invalid duplicated cohort")

    positive_count = int(output[TARGET_COLUMN].sum())
    if enforce_expected_counts:
        if len(output) != EXPECTED_COHORT_SIZE:
            raise ValueError(
                f"Expected {EXPECTED_COHORT_SIZE:,} eligible rows, found {len(output):,}"
            )
        if positive_count != EXPECTED_POSITIVE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_POSITIVE_COUNT:,} positive targets, found {positive_count:,}"
            )

    missing = output[FEATURE_COLUMNS].isna().sum()
    audit: dict[str, Any] = {
        "source_rows": source_rows,
        "adult_demo_rows": adults_before_exam_merge,
        "analytic_rows": int(len(output)),
        "positive_target_rows": positive_count,
        "negative_target_rows": int(len(output) - positive_count),
        "positive_target_rate": float(output[TARGET_COLUMN].mean()),
        "minimum_age": int(output["age"].min()),
        "maximum_age_code": int(output["age"].max()),
        "age_19_rows": int(output["age"].eq(19).sum()),
        "age_19_missing_education_rows": int(
            (output["age"].eq(19) & output["education"].isna()).sum()
        ),
        "missing_by_feature": {column: int(value) for column, value in missing.items()},
        "complete_case_rows": int(output[FEATURE_COLUMNS].notna().all(axis=1).sum()),
        "target_definition": f"LBXGH >= {HBA1C_THRESHOLD}%",
        "age_filter": "RIDAGEYR > 18",
        "unit_of_analysis": "One NHANES examined adult respondent",
    }
    return output, audit


def save_analytic_dataset(
    output_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, audit = build_analytic_dataset()
    dataset_path = output_dir / "nhanes_hba1c_adults.csv"
    audit_path = output_dir / "data_audit.json"
    dataset.to_csv(dataset_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return dataset_path, audit_path, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    dataset, audit = build_analytic_dataset(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_dir / "nhanes_hba1c_adults.csv", index=False)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
