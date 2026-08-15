"""Train, compare, explain, and package all faculty-required models."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACT_DIR,
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
)
from src.data_pipeline import build_analytic_dataset
from src.eda import generate_eda, outlier_audit
from src.evaluation import (
    bootstrap_intervals,
    choose_f2_threshold,
    evaluate_probabilities,
    save_evaluation_plots,
    subgroup_audit,
    write_json,
)
from src.explain import generate_explanations
from src.modeling import tune_required_models, tune_stacking_model


MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "stacking": "Heterogeneous Stacking",
}


def make_splits(
    dataset: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create reproducible 70/15/15 stratified train/validation/test splits."""
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].copy()
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )
    validation_fraction_of_development = 0.15 / 0.85
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=validation_fraction_of_development,
        stratify=y_development,
        random_state=random_state,
    )
    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }


def _split_summary(splits: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name in ("train", "validation", "test"):
        y = splits[f"y_{name}"]
        result[name] = {
            "rows": int(len(y)),
            "positive_rows": int(y.sum()),
            "positive_rate": float(y.mean()),
        }
    return result


def _model_card(
    best_name: str,
    decision_threshold: float,
    results: pd.DataFrame,
    ensemble_comparison: dict[str, Any],
) -> str:
    table = results[
        ["model_label", "roc_auc", "precision", "recall", "f1", "specificity", "threshold"]
    ].to_markdown(index=False, floatfmt=".3f")
    comparison = "did" if ensemble_comparison["stacking_outperformed_baseline_roc_auc"] else "did not"
    return f"""# Model card: elevated-HbA1c screening

## Intended use

The model prioritizes adults aged over 18 for confirmatory HbA1c testing when universal laboratory testing is difficult. It is a screening-support prototype, not a diagnosis, prognosis, or treatment recommendation.

## Selected model

- Model: **{MODEL_LABELS[best_name]}**
- Validation-selected decision threshold: **{decision_threshold:.3f}**
- Target: measured HbA1c >= 5.7%

## Untouched test-set results

{table}

The heterogeneous stacking model {comparison} outperform the Logistic Regression baseline in test ROC-AUC. This conclusion is descriptive for the held-out sample and does not establish superiority in another population.

## Key limitations

- Cross-sectional NHANES data support detection of current elevated HbA1c, not future diabetes prediction.
- Activity and smoking measures are self-reported.
- The model has not received external or prospective clinical validation.
- HbA1c can be affected by anemia, kidney/liver disease, pregnancy, blood disorders, medicines, blood loss, and transfusion; those factors are not fully represented here.
- Only the approved variables are used. Racial fairness cannot be assessed because race/ethnicity is not included.
- Human review and confirmatory testing are mandatory before any health action.
"""


def run_training(
    *,
    cv_folds: int = 5,
    n_iter: int = 10,
    bootstrap_repeats: int = 500,
    shap_sample_size: int = 80,
    n_jobs: int = -1,
) -> dict[str, Any]:
    dataset, data_audit = build_analytic_dataset()
    dataset.to_csv(PROCESSED_DATA_DIR / "nhanes_hba1c_adults.csv", index=False)
    write_json(PROCESSED_DATA_DIR / "data_audit.json", data_audit)

    splits = make_splits(dataset)
    split_summary = _split_summary(splits)
    write_json(PROCESSED_DATA_DIR / "split_summary.json", split_summary)

    training_eda = splits["X_train"].copy()
    training_eda[TARGET_COLUMN] = splits["y_train"]
    generate_eda(training_eda)
    outlier_audit(training_eda).to_csv(REPORT_DIR / "training_outlier_audit.csv", index=False)

    search_output = tune_required_models(
        splits["X_train"],
        splits["y_train"],
        cv_folds=cv_folds,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )
    models = search_output.estimators
    stacking, stacking_tuning = tune_stacking_model(
        models,
        splits["X_train"],
        splits["y_train"],
        splits["X_validation"],
        splits["y_validation"],
        cv_folds=cv_folds,
        n_jobs=n_jobs,
    )
    models["stacking"] = stacking

    validation_results: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for name, model in models.items():
        probability = model.predict_proba(splits["X_validation"])[:, 1]
        threshold = choose_f2_threshold(splits["y_validation"], probability)
        thresholds[name] = threshold
        metrics = evaluate_probabilities(splits["y_validation"], probability, threshold)
        validation_results.append({"model": name, **metrics})

    validation_frame = pd.DataFrame(validation_results).sort_values("roc_auc", ascending=False)
    validation_frame.to_csv(REPORT_DIR / "validation_model_comparison.csv", index=False)
    best_name = str(validation_frame.iloc[0]["model"])

    X_development = pd.concat([splits["X_train"], splits["X_validation"]], axis=0)
    y_development = pd.concat([splits["y_train"], splits["y_validation"]], axis=0)

    final_models: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    test_results: list[dict[str, Any]] = []
    confidence_intervals: dict[str, Any] = {}
    for name, model in models.items():
        final_model = clone(model)
        final_model.fit(X_development, y_development)
        final_models[name] = final_model
        probability = final_model.predict_proba(splits["X_test"])[:, 1]
        test_probabilities[name] = probability
        metrics = evaluate_probabilities(splits["y_test"], probability, thresholds[name])
        test_results.append(
            {"model": name, "model_label": MODEL_LABELS[name], **metrics}
        )
        confidence_intervals[name] = bootstrap_intervals(
            splits["y_test"],
            probability,
            thresholds[name],
            repeats=bootstrap_repeats,
        )

    results = pd.DataFrame(test_results).sort_values("roc_auc", ascending=False)
    results.to_csv(REPORT_DIR / "test_model_comparison.csv", index=False)
    write_json(REPORT_DIR / "test_metric_confidence_intervals.json", confidence_intervals)

    labelled_probabilities = {
        MODEL_LABELS[name]: probability for name, probability in test_probabilities.items()
    }
    labelled_thresholds = {MODEL_LABELS[name]: threshold for name, threshold in thresholds.items()}
    save_evaluation_plots(
        splits["y_test"], labelled_probabilities, labelled_thresholds
    )

    best_model = final_models[best_name]
    best_probability = test_probabilities[best_name]
    fairness = subgroup_audit(
        splits["X_test"],
        splits["y_test"],
        best_probability,
        thresholds[best_name],
    )
    fairness.to_csv(REPORT_DIR / "fairness_subgroup_metrics.csv", index=False)

    logistic_auc = float(
        results.loc[results["model"].eq("logistic_regression"), "roc_auc"].iloc[0]
    )
    stacking_auc = float(results.loc[results["model"].eq("stacking"), "roc_auc"].iloc[0])
    ensemble_comparison = {
        "baseline_test_roc_auc": logistic_auc,
        "stacking_test_roc_auc": stacking_auc,
        "stacking_minus_baseline_roc_auc": stacking_auc - logistic_auc,
        "stacking_outperformed_baseline_roc_auc": bool(stacking_auc > logistic_auc),
    }
    write_json(REPORT_DIR / "ensemble_vs_baseline.json", ensemble_comparison)

    selected_metrics = results.loc[results["model"].eq(best_name)].iloc[0].to_dict()
    screened = len(splits["y_test"])
    impact = {
        "basis": "Untouched test set, scaled to 100 screened adults",
        "lab_referrals_per_100": float(
            100
            * (selected_metrics["true_positive"] + selected_metrics["false_positive"])
            / screened
        ),
        "elevated_cases_captured_per_100": float(
            100 * selected_metrics["true_positive"] / screened
        ),
        "elevated_cases_missed_per_100": float(
            100 * selected_metrics["false_negative"] / screened
        ),
        "false_referrals_per_100": float(
            100 * selected_metrics["false_positive"] / screened
        ),
        "warning": "Simulated sample impact, not observed clinical impact.",
    }
    write_json(REPORT_DIR / "screening_impact.json", impact)

    model_path = ARTIFACT_DIR / "model_pipeline.joblib"
    joblib.dump(best_model, model_path)
    metadata = {
        "selected_model": best_name,
        "selected_model_label": MODEL_LABELS[best_name],
        "decision_threshold": thresholds[best_name],
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "target_definition": data_audit["target_definition"],
        "age_filter": data_audit["age_filter"],
        "random_state": RANDOM_STATE,
        "split_summary": split_summary,
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
    }
    write_json(ARTIFACT_DIR / "model_metadata.json", metadata)
    write_json(REPORT_DIR / "cv_model_summary.json", search_output.cv_summary)
    write_json(REPORT_DIR / "stacking_tuning.json", stacking_tuning)

    explanation_summary = generate_explanations(
        best_model,
        X_development,
        splits["X_test"],
        sample_size=shap_sample_size,
    )

    card = _model_card(best_name, thresholds[best_name], results, ensemble_comparison)
    (REPORT_DIR / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    run_summary = {
        "selected_model": best_name,
        "selected_threshold": thresholds[best_name],
        "data_audit": data_audit,
        "split_summary": split_summary,
        "ensemble_comparison": ensemble_comparison,
        "screening_impact": impact,
        "explanation_summary": explanation_summary,
    }
    write_json(REPORT_DIR / "run_summary.json", run_summary)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--shap-sample-size", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced search/uncertainty settings for a smoke run.",
    )
    args = parser.parse_args()
    if args.quick:
        args.cv_folds = 3
        args.n_iter = 3
        args.bootstrap_repeats = 100
        args.shap_sample_size = 20

    summary = run_training(
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        bootstrap_repeats=args.bootstrap_repeats,
        shap_sample_size=args.shap_sample_size,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

