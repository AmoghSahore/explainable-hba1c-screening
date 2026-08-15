from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EXPECTED_COHORT_SIZE, EXPECTED_POSITIVE_COUNT, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_pipeline import build_analytic_dataset, derive_smoking, derive_weekly_activity


def test_analytic_cohort_matches_validated_release():
    dataset, audit = build_analytic_dataset()
    assert len(dataset) == EXPECTED_COHORT_SIZE
    assert int(dataset[TARGET_COLUMN].sum()) == EXPECTED_POSITIVE_COUNT
    assert dataset["age"].min() == 19
    assert list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert "SEQN" not in dataset.columns
    assert "LBXGH" not in dataset.columns
    assert audit["age_filter"] == "RIDAGEYR > 18"
    assert audit["age_19_rows"] == 105
    assert audit["age_19_missing_education_rows"] == 105
    assert audit["missing_by_feature"]["moderate_activity_min_week"] == 43
    assert audit["missing_by_feature"]["vigorous_activity_min_week"] == 36


def test_weekly_activity_conversion_and_special_codes():
    result = derive_weekly_activity(
        pd.Series([2, 3, 4, 52, 0, 7777, 9999], dtype=float),
        pd.Series(["D", "W", "M", "Y", "", "", ""]),
        pd.Series([30, 30, 30, 30, np.nan, 30, 30], dtype=float),
    )
    expected = [420.0, 90.0, 4 * 30 * 12 / 52, 30.0, 0.0, np.nan, np.nan]
    np.testing.assert_allclose(result.iloc[:5], expected[:5])
    assert result.iloc[5:].isna().all()


def test_smoking_skip_pattern_is_respected():
    ever, current = derive_smoking(
        pd.Series([1, 1, 1, 2, 7, 9], dtype=float),
        pd.Series([1, 2, 3, np.nan, np.nan, np.nan], dtype=float),
    )
    assert ever.tolist()[:4] == ["Ever", "Ever", "Ever", "Never"]
    assert current.tolist()[:4] == ["Current", "Current", "Not current", "Not current"]
    assert ever.iloc[4:].isna().all()
    assert current.iloc[4:].isna().all()
