"""Leakage-safe preprocessing, tuning, and heterogeneous stacking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Clip numeric values at quantiles learned only from the fitting fold."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Quantile bounds must satisfy 0 <= lower < upper <= 1")
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clip", QuantileClipper(lower=0.01, upper=0.99)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Not collected/unknown"),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _pipeline(model: Any) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


@dataclass
class SearchOutput:
    estimators: dict[str, Any]
    cv_summary: list[dict[str, Any]]


def tune_required_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv_folds: int = 5,
    n_iter: int = 10,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
) -> SearchOutput:
    """Tune the baseline, required bagging model, and required boosting model."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    logistic = GridSearchCV(
        estimator=_pipeline(
            LogisticRegression(max_iter=3_000, solver="lbfgs", random_state=random_state)
        ),
        param_grid={
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )

    random_forest = RandomizedSearchCV(
        estimator=_pipeline(
            RandomForestClassifier(
                random_state=random_state,
                n_jobs=1,
            )
        ),
        param_distributions={
            "model__n_estimators": [250, 400, 600],
            "model__max_depth": [None, 6, 10, 16],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", 0.6, 0.9],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        n_iter=n_iter,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    )

    xgboost = RandomizedSearchCV(
        estimator=_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
            )
        ),
        param_distributions={
            "model__n_estimators": [150, 250, 400, 600],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__min_child_weight": [1, 3, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
            "model__scale_pos_weight": [1.0, positive_weight],
        },
        n_iter=n_iter,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        error_score="raise",
        return_train_score=False,
    )

    searches = {
        "logistic_regression": logistic,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }
    estimators: dict[str, Any] = {}
    cv_summary: list[dict[str, Any]] = []
    for name, search in searches.items():
        search.fit(X_train, y_train)
        estimators[name] = search.best_estimator_
        best_index = int(search.best_index_)
        cv_summary.append(
            {
                "model": name,
                "mean_cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][best_index]),
                "std_cv_roc_auc": float(search.cv_results_["std_test_roc_auc"][best_index]),
                "mean_cv_average_precision": float(
                    search.cv_results_["mean_test_average_precision"][best_index]
                ),
                "mean_cv_recall_at_0_5": float(
                    search.cv_results_["mean_test_recall"][best_index]
                ),
                "best_params": search.best_params_,
            }
        )
    return SearchOutput(estimators=estimators, cv_summary=cv_summary)


def tune_stacking_model(
    base_estimators: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
) -> tuple[StackingClassifier, list[dict[str, float]]]:
    """Tune the meta-learner using validation data and OOF training predictions."""
    from sklearn.metrics import roc_auc_score

    candidates: list[tuple[float, StackingClassifier, float]] = []
    history: list[dict[str, float]] = []
    for c_value in (0.1, 1.0, 10.0):
        model = StackingClassifier(
            estimators=[
                ("lr", clone(base_estimators["logistic_regression"])),
                ("rf", clone(base_estimators["random_forest"])),
                ("xgb", clone(base_estimators["xgboost"])),
            ],
            final_estimator=LogisticRegression(
                C=c_value,
                max_iter=3_000,
                random_state=random_state,
            ),
            stack_method="predict_proba",
            cv=StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            ),
            passthrough=False,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        score = float(roc_auc_score(y_validation, validation_probability))
        candidates.append((c_value, model, score))
        history.append({"final_estimator_C": c_value, "validation_roc_auc": score})
    _, best_model, _ = max(candidates, key=lambda item: item[2])
    return best_model, history

