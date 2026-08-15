from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_pipeline import build_analytic_dataset
from src.evaluation import choose_f2_threshold, evaluate_probabilities
from src.modeling import build_preprocessor
from src.train import make_splits


def test_splits_are_stratified_and_disjoint():
    dataset, _ = build_analytic_dataset()
    splits = make_splits(dataset)
    assert sum(len(splits[f"y_{name}"]) for name in ("train", "validation", "test")) == len(dataset)
    for name in ("train", "validation", "test"):
        assert abs(splits[f"y_{name}"].mean() - dataset[TARGET_COLUMN].mean()) < 0.01
    assert set(splits["X_train"].index).isdisjoint(splits["X_validation"].index)
    assert set(splits["X_train"].index).isdisjoint(splits["X_test"].index)
    assert set(splits["X_validation"].index).isdisjoint(splits["X_test"].index)


def test_preprocessing_and_baseline_smoke_fit():
    dataset, _ = build_analytic_dataset()
    sample = dataset.sample(n=500, random_state=7)
    model = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1_000)),
        ]
    )
    model.fit(sample[FEATURE_COLUMNS], sample[TARGET_COLUMN])
    probability = model.predict_proba(sample[FEATURE_COLUMNS])[:, 1]
    assert probability.shape == (500,)
    assert np.isfinite(probability).all()
    threshold = choose_f2_threshold(sample[TARGET_COLUMN], probability)
    metrics = evaluate_probabilities(sample[TARGET_COLUMN], probability, threshold)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0.05 <= threshold <= 0.95

