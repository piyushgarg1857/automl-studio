"""Leakage-safe evaluation utilities for tabular AutoML workflows.

All learned preprocessing is kept inside an sklearn Pipeline so it is fitted
only on each training fold (and never on the holdout test set).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def detect_problem_type(
    target: pd.Series, *, task: str = "auto", max_classes: int = 20
) -> str:
    """Return classification/regression; allow explicit override.

    Numeric targets with many unique values are treated as regression. A
    numeric target with few unique values is ambiguous, so callers can pass
    task='classification' or task='regression' when domain knowledge matters.
    """
    if task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'")
    if task != "auto":
        return task
    if target.dropna().empty:
        raise ValueError("Target column has no non-missing values")
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "classification"
    return "classification" if target.nunique(dropna=True) <= max_classes else "regression"


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a mixed-type transformer with fold-local imputation/encoding."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("Dataset has no usable feature columns")
    return ColumnTransformer(transformers, remainder="drop")


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: Mapping[str, object],
    *,
    problem_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    scale_numeric: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Evaluate estimators with a holdout set and leakage-safe cross-validation.

    Returns a leaderboard and per-model artifacts containing the fitted
    pipeline, holdout labels/predictions, and any error encountered.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if X.empty or len(X) < 4:
        raise ValueError("At least four rows and one feature are required")
    if y.isna().any():
        raise ValueError("Target contains missing values; clean or filter it first")
    task = detect_problem_type(y, task=problem_type)
    # XGBoost requires class labels encoded as consecutive integers (0..n-1).
    # Encode classification targets consistently before splitting/CV.
    label_encoder = None
    original_y = y.copy()
    if task == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(
            label_encoder.fit_transform(y),
            index=y.index,
            name=y.name,
        )
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    rows, artifacts = [], {}
    if task == "classification":
        min_class_count = int(y_train.value_counts().min())
        folds = min(cv_folds, min_class_count)
        if folds < 2:
            raise ValueError(
                "Each class needs at least two training examples for cross-validation"
            )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = "f1_weighted"
    else:
        folds = min(cv_folds, len(X_train))
        if folds < 2:
            raise ValueError("Not enough training rows for cross-validation")
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = "r2"

    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocess", build_preprocessor(X_train, scale_numeric=scale_numeric)),
             ("model", estimator)]
        )
        try:
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score="raise"
            )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            if task == "classification":
                row = {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision (weighted)": precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    ),
                    "F1 (weighted)": f1_score(
                        y_test, predictions, average="weighted", zero_division=0
                    ),
                    "CV F1 Mean": float(np.mean(cv_scores)),
                    "CV F1 Std": float(np.std(cv_scores)),
                }
            else:
                row = {
                    "Model": name,
                    "R²": r2_score(y_test, predictions),
                    "MAE": mean_absolute_error(y_test, predictions),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                    "CV R² Mean": float(np.mean(cv_scores)),
                    "CV R² Std": float(np.std(cv_scores)),
                }
            rows.append(row)
            if label_encoder is not None:
                # Keep original class names for display in the UI.
                display_y_test = pd.Series(
                    label_encoder.inverse_transform(np.asarray(y_test, dtype=int)),
                    index=y_test.index,
                    name=original_y.name,
                )
                display_predictions = label_encoder.inverse_transform(
                    np.asarray(predictions, dtype=int)
                )
            else:
                display_y_test = y_test.copy()
                display_predictions = np.asarray(predictions)

            artifacts[name] = {
                "pipeline": pipeline,
                "X_test": X_test.copy(),
                "y_test": display_y_test,
                "predictions": np.asarray(display_predictions),
                "label_encoder": label_encoder,
                "error": None,
            }
        except Exception as exc:  # preserve failures so users can diagnose them
            artifacts[name] = {"pipeline": None, "error": f"{type(exc).__name__}: {exc}"}
            rows.append({"Model": name, "Error": artifacts[name]["error"]})

    leaderboard = pd.DataFrame(rows)
    score_col = "CV F1 Mean" if task == "classification" else "CV R² Mean"
    if score_col in leaderboard:
        leaderboard = leaderboard.sort_values(score_col, ascending=False, na_position="last")
    return leaderboard.reset_index(drop=True), artifacts



def tune_estimator(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    param_distributions: Mapping[str, object],
    *,
    problem_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3,
    n_iter: int = 10,
    scale_numeric: bool = True,
):
    """Tune an estimator using only the training split and fold-local preprocessing.

    The holdout split is deliberately not passed to RandomizedSearchCV. Returns
    the fitted search object, target encoder (classification only), and task.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if X.empty or len(X) < 4:
        raise ValueError("At least four rows and one feature are required")
    if y.isna().any():
        raise ValueError("Target contains missing values; clean or filter it first")

    task = detect_problem_type(y, task=problem_type)
    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
        stratify = y
    else:
        stratify = None

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if task == "classification":
        min_class_count = int(y_train.value_counts().min())
        folds = min(cv_folds, min_class_count)
        if folds < 2:
            raise ValueError("Each class needs at least two training examples for tuning CV")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = "f1_weighted"
    else:
        folds = min(cv_folds, len(X_train))
        if folds < 2:
            raise ValueError("Not enough training rows for tuning CV")
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = "r2"

    pipeline = Pipeline([
        ("preprocess", build_preprocessor(X_train, scale_numeric=scale_numeric)),
        ("model", estimator),
    ])
    prefixed_params = {
        (key if key.startswith("model__") else f"model__{key}"): value
        for key, value in param_distributions.items()
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=prefixed_params,
        n_iter=max(1, int(n_iter)),
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
        refit=True,
    )
    search.fit(X_train, y_train)
    return search, label_encoder, task
