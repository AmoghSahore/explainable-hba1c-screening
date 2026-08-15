"""Model-agnostic SHAP explanations for the complete deployed pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    DISPLAY_NAMES,
    FEATURE_COLUMNS,
    FIGURE_DIR,
    NUMERIC_FEATURES,
    REPORT_DIR,
)


def default_synthetic_record() -> pd.DataFrame:
    """Return a plausible fictional adult record for the required live demo."""
    return pd.DataFrame(
        [
            {
                "age": 58.0,
                "income_ratio": 1.8,
                "bmi": 34.0,
                "waist_cm": 112.0,
                "mean_systolic_bp": 145.0,
                "mean_diastolic_bp": 88.0,
                "moderate_activity_min_week": 30.0,
                "vigorous_activity_min_week": 0.0,
                "sedentary_min_day": 600.0,
                "sex": "Male",
                "education": "High school/GED",
                "smoking_history": "Ever",
                "current_smoking": "Current",
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def encode_explanation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent mixed raw inputs numerically for SHAP's tabular masker."""
    encoded = pd.DataFrame(index=frame.index)
    for feature in NUMERIC_FEATURES:
        encoded[feature] = pd.to_numeric(frame[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        mapping = {level: index for index, level in enumerate(CATEGORY_LEVELS[feature])}
        encoded[feature] = frame[feature].map(mapping).fillna(-1).astype(float)
    return encoded.loc[:, FEATURE_COLUMNS]


def decode_explanation_values(values: Any) -> pd.DataFrame:
    """Decode SHAP's numeric masking representation back to model inputs."""
    encoded = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    decoded = pd.DataFrame(index=encoded.index)
    for feature in NUMERIC_FEATURES:
        decoded[feature] = pd.to_numeric(encoded[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        levels = CATEGORY_LEVELS[feature]

        def decode(value: Any):
            if pd.isna(value) or float(value) < -0.5:
                return np.nan
            index = int(round(float(value)))
            return levels[index] if 0 <= index < len(levels) else np.nan

        decoded[feature] = encoded[feature].map(decode)
    return decoded.loc[:, FEATURE_COLUMNS]


def build_explainer(model: Any, background: pd.DataFrame):
    import shap

    encoded_background = encode_explanation_frame(background)

    def predict_positive(values: Any) -> np.ndarray:
        frame = decode_explanation_values(values)
        return model.predict_proba(frame)[:, 1]

    return shap.Explainer(
        predict_positive,
        encoded_background,
        algorithm="permutation",
        feature_names=FEATURE_COLUMNS,
    )


def explain_rows(model: Any, background: pd.DataFrame, rows: pd.DataFrame):
    explainer = build_explainer(model, background)
    minimum_evaluations = 2 * len(FEATURE_COLUMNS) + 1
    return explainer(encode_explanation_frame(rows), max_evals=minimum_evaluations)


def generate_explanations(
    model: Any,
    X_training: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    sample_size: int = 80,
    random_state: int = 42,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    background = X_training.sample(n=min(40, len(X_training)), random_state=random_state)
    explained_rows = X_test.sample(n=min(sample_size, len(X_test)), random_state=random_state)
    explanations = explain_rows(model, background, explained_rows)
    values = np.asarray(explanations.values)

    global_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "display_name": [DISPLAY_NAMES[name] for name in FEATURE_COLUMNS],
            "mean_absolute_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    global_importance.to_csv(REPORT_DIR / "shap_global_importance.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_frame = global_importance.sort_values("mean_absolute_shap")
    positions = np.arange(len(plot_frame))
    ax.barh(positions, plot_frame["mean_absolute_shap"], color="#35618f")
    ax.set_yticks(positions, labels=plot_frame["display_name"].tolist())
    ax.set(title="Global SHAP importance — best model", xlabel="Mean |SHAP value|", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    synthetic = default_synthetic_record()
    synthetic_explanation = explain_rows(model, background, synthetic)
    synthetic_values = np.asarray(synthetic_explanation.values)[0]
    local = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "display_name": [DISPLAY_NAMES[name] for name in FEATURE_COLUMNS],
            "value": [synthetic.iloc[0][name] for name in FEATURE_COLUMNS],
            "shap_value": synthetic_values,
        }
    ).sort_values("shap_value")
    local.to_csv(REPORT_DIR / "shap_synthetic_record.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = np.where(local["shap_value"].to_numpy() >= 0, "#e45756", "#4c78a8")
    positions = np.arange(len(local))
    ax.barh(positions, local["shap_value"], color=colors)
    ax.set_yticks(positions, labels=local["display_name"].tolist())
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        title="Local SHAP explanation — fictional demo record",
        xlabel="Contribution to elevated-HbA1c probability",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_local_synthetic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    background.to_csv(ARTIFACT_DIR / "shap_background.csv", index=False)
    synthetic.iloc[0].to_json(ARTIFACT_DIR / "synthetic_record.json", indent=2)
    probability = float(model.predict_proba(synthetic)[:, 1][0])
    summary = {
        "background_rows": int(len(background)),
        "globally_explained_test_rows": int(len(explained_rows)),
        "synthetic_probability": probability,
        "synthetic_record": synthetic.iloc[0].to_dict(),
        "base_value": float(np.asarray(synthetic_explanation.base_values).reshape(-1)[0]),
    }
    (REPORT_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
