"""Training-only exploratory analysis and report-ready figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DISPLAY_NAMES,
    FIGURE_DIR,
    NUMERIC_FEATURES,
    REPORT_DIR,
    TARGET_COLUMN,
)


def generate_eda(
    training_frame: pd.DataFrame,
    *,
    figure_dir: Path = FIGURE_DIR,
    report_dir: Path = REPORT_DIR,
) -> None:
    """Create EDA using training rows only to avoid test-informed decisions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    missing = (
        training_frame.drop(columns=[TARGET_COLUMN])
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("missing_percent")
        .to_frame()
    )
    missing.to_csv(report_dir / "training_missingness.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    missing.loc[missing["missing_percent"].gt(0)].sort_values("missing_percent").plot.barh(
        y="missing_percent", legend=False, color="#35618f", ax=ax
    )
    ax.set(title="Missing values in training data", xlabel="Missing (%)", ylabel="")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_missingness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    target_counts = training_frame[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Not elevated", "Elevated"], target_counts.values, color=["#4c78a8", "#e45756"])
    ax.bar_label(bars, labels=[f"{value:,}" for value in target_counts.values])
    ax.set(title="Elevated HbA1c target balance — training set", ylabel="Respondents")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_target_balance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = training_frame[NUMERIC_FEATURES].describe().T
    summary["missing"] = training_frame[NUMERIC_FEATURES].isna().sum()
    summary.to_csv(report_dir / "training_numeric_summary.csv")

    selected = ["age", "bmi", "waist_cm", "mean_systolic_bp", "sedentary_min_day", "income_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for axis, feature in zip(axes.ravel(), selected):
        sns.histplot(
            data=training_frame,
            x=feature,
            hue=TARGET_COLUMN,
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            ax=axis,
        )
        axis.set_title(DISPLAY_NAMES[feature])
        axis.set_xlabel("")
    fig.suptitle("Training-set feature distributions by target", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_numeric_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    correlations = training_frame[NUMERIC_FEATURES + [TARGET_COLUMN]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlations, cmap="vlag", center=0, square=True, ax=ax)
    ax.set_title("Training-set numeric correlation matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rates: list[pd.DataFrame] = []
    for feature in CATEGORICAL_FEATURES:
        grouped = (
            training_frame.assign(
                **{feature: training_frame[feature].fillna("Not collected/unknown")}
            )
            .groupby(feature, dropna=False)[TARGET_COLUMN]
            .agg(["mean", "size"])
            .reset_index()
            .rename(columns={feature: "group", "mean": "elevated_rate", "size": "n"})
        )
        grouped.insert(0, "feature", feature)
        rates.append(grouped)
    categorical_rates = pd.concat(rates, ignore_index=True)
    categorical_rates.to_csv(report_dir / "training_categorical_target_rates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    for axis, feature in zip(axes.ravel(), CATEGORICAL_FEATURES):
        frame = categorical_rates.loc[categorical_rates["feature"].eq(feature)].copy()
        frame = frame.sort_values("elevated_rate", ascending=True)
        sns.barplot(data=frame, x="elevated_rate", y="group", color="#4c78a8", ax=axis)
        axis.set(title=DISPLAY_NAMES[feature], xlabel="Elevated-HbA1c rate", ylabel="")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle("Training-set target rates by categorical feature", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_categorical_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def outlier_audit(training_frame: pd.DataFrame) -> pd.DataFrame:
    """Describe conventional IQR outliers without deleting valid observations."""
    rows = []
    for feature in NUMERIC_FEATURES:
        values = training_frame[feature].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "feature": feature,
                "q1": float(q1),
                "q3": float(q3),
                "iqr_lower_fence": float(lower),
                "iqr_upper_fence": float(upper),
                "iqr_outlier_rows": int(((values < lower) | (values > upper)).sum()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)

