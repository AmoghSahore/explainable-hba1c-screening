"""Threshold selection, model evaluation, uncertainty, and subgroup auditing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DISPLAY_NAMES, FIGURE_DIR, RANDOM_STATE


def choose_f2_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    """Choose a recall-sensitive decision threshold using validation data only."""
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        fbeta_score(y_true, probability >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_intervals(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    repeats: int = 500,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[float]]:
    """Return percentile bootstrap CIs for the main reported metrics."""
    y_array = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(random_state)
    values = {metric: [] for metric in ("roc_auc", "precision", "recall", "f1")}
    for _ in range(repeats):
        index = rng.integers(0, len(y_array), len(y_array))
        sampled_y = y_array[index]
        if np.unique(sampled_y).size < 2:
            continue
        sampled_probability = probability[index]
        sampled = evaluate_probabilities(sampled_y, sampled_probability, threshold)
        for metric in values:
            values[metric].append(float(sampled[metric]))
    return {
        metric: [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]
        for metric, scores in values.items()
    }


def subgroup_audit(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Evaluate error rates for available demographic and socioeconomic groups."""
    audit_frame = X_test.copy()
    audit_frame["age_group"] = pd.cut(
        audit_frame["age"],
        bins=[18, 39, 59, np.inf],
        labels=["19-39", "40-59", "60+"],
    ).astype("string")
    audit_frame["income_group"] = pd.cut(
        audit_frame["income_ratio"],
        bins=[-np.inf, 1.3, 3.5, np.inf],
        labels=["Lower income ratio", "Middle income ratio", "Higher income ratio"],
    ).astype("string")
    audit_frame["truth"] = np.asarray(y_test)
    audit_frame["probability"] = probability

    records: list[dict[str, Any]] = []
    for attribute in ("sex", "age_group", "education", "income_group"):
        grouped = audit_frame.assign(
            **{attribute: audit_frame[attribute].fillna("Missing/not collected")}
        ).groupby(attribute, dropna=False)
        for group, frame in grouped:
            if len(frame) < 20 or frame["truth"].nunique() < 2:
                continue
            metrics = evaluate_probabilities(
                frame["truth"], frame["probability"].to_numpy(), threshold
            )
            records.append(
                {
                    "attribute": attribute,
                    "group": str(group),
                    "n": int(len(frame)),
                    "positive_rate": float(frame["truth"].mean()),
                    "roc_auc": metrics["roc_auc"],
                    "precision": metrics["precision"],
                    "recall_tpr": metrics["recall"],
                    "false_negative_rate": float(1 - metrics["recall"]),
                    "false_positive_rate": float(1 - metrics["specificity"]),
                }
            )
    return pd.DataFrame.from_records(records)


def save_evaluation_plots(
    y_test: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    output_dir: Path = FIGURE_DIR,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name, probability in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        axes[0].plot(fpr, tpr, label=f"{name} ({auc:.3f})")
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ap = average_precision_score(y_test, probability)
        axes[1].plot(recall, precision, label=f"{name} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].set(title="ROC curves — untouched test set", xlabel="False-positive rate", ylabel="Recall")
    axes[1].set(title="Precision–recall curves — untouched test set", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "model_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    count = len(model_probabilities)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False)
    for axis, (name, probability) in zip(axes.ravel(), model_probabilities.items()):
        matrix = confusion_matrix(y_test, probability >= thresholds[name], labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set(title=name, xlabel="Predicted", ylabel="Actual")
        axis.set_xticklabels(["Not elevated", "Elevated"])
        axis.set_yticklabels(["Not elevated", "Elevated"], rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probability in model_probabilities.items():
        observed, predicted = calibration_curve(y_test, probability, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set(
        title="Calibration — untouched test set",
        xlabel="Mean predicted probability",
        ylabel="Observed elevated-HbA1c rate",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

