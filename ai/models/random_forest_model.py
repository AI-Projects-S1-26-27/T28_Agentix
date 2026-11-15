"""HealthSensing: Random Forest Classical ML Benchmark Model.

This module provides an experimental Random Forest classifier for physiological
stress classification using Leave-One-Participant-Out cross-validation.

Clinical Disclaimer:
--------------------
This module provides an experimental physiological stress classification benchmark
for algorithmic evaluation and research. It does NOT constitute medical diagnosis
and must not be used for clinical decision-making.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

DEFAULT_PHYSIOLOGICAL_COLUMNS: tuple[str, ...] = (
    "heart_rate_bpm",
    "eda_microsiemens",
    "body_temperature_c",
)

FORBIDDEN_COLUMNS: frozenset[str] = frozenset({
    "stress_label",
    "participant_id",
    "session_id",
    "timestamp",
    "source",
    "block",
    "label",
    "true_label",
    "true_class",
    "predicted_class",
})


def create_rf_classifier(
    n_estimators: int = 300,
    class_weight: str | dict | None = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Construct a scikit-learn RandomForestClassifier with default benchmark parameters.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of trees in the forest.
    class_weight : str | dict | None, default="balanced_subsample"
        Weights associated with classes to address baseline/stress imbalance.
    random_state : int, default=42
        Seed for pseudo-random number generator for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs to run.
    **kwargs : Any
        Additional keyword arguments passed to RandomForestClassifier.

    Returns
    -------
    RandomForestClassifier
        Configured, unfitted estimator.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )


class StressRandomForestClassifier:
    """Wrapper around scikit-learn RandomForestClassifier with input validation and leakage prevention.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of trees in the forest.
    class_weight : str | dict | None, default="balanced_subsample"
        Class weights strategy.
    random_state : int, default=42
        Random state seed.
    n_jobs : int, default=-1
        Number of parallel execution jobs.
    imputer_strategy : str, default="median"
        Strategy for missing value imputation fitted on training data only.
    **rf_kwargs : Any
        Additional arguments passed to RandomForestClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        class_weight: str | dict | None = "balanced_subsample",
        random_state: int = 42,
        n_jobs: int = -1,
        imputer_strategy: str = "median",
        **rf_kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.imputer_strategy = imputer_strategy
        self.rf_kwargs = rf_kwargs

        self.imputer = SimpleImputer(strategy=imputer_strategy)
        self.classifier = create_rf_classifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **rf_kwargs,
        )
        self.feature_names_: list[str] | None = None
        self.fitted_with_dataframe_: bool = False
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def _validate_features(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, list[str] | None]:
        """Validate feature matrix to ensure type consistency, ordering, and no infinities."""
        if self.is_fitted_:
            # Enforce input type consistency: DataFrame vs NumPy array
            if self.fitted_with_dataframe_ and not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"Model was fitted with a pandas DataFrame; prediction requires a pandas DataFrame, "
                    f"received {type(X).__name__}."
                )
            if not self.fitted_with_dataframe_ and isinstance(X, pd.DataFrame):
                raise TypeError(
                    "Model was fitted with a NumPy array; prediction requires a 2D NumPy array, "
                    "received pandas DataFrame."
                )

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input feature DataFrame is empty.")

            forbidden_found = FORBIDDEN_COLUMNS.intersection(X.columns)
            if forbidden_found:
                raise ValueError(
                    f"Feature matrix contains forbidden non-physiological columns: {sorted(forbidden_found)}"
                )

            feature_names = list(X.columns)
            if self.is_fitted_ and self.feature_names_ is not None and feature_names != self.feature_names_:
                raise ValueError(
                    f"Feature columns and order do not match fitted model features.\n"
                    f"Expected: {self.feature_names_}\n"
                    f"Received: {feature_names}"
                )

            try:
                X_arr = X.to_numpy(dtype=float)
            except (ValueError, TypeError) as err:
                raise TypeError(f"All feature values must be numeric: {err}") from err

            if np.isinf(X_arr).any():
                raise ValueError("Feature matrix contains infinite (+inf or -inf) values.")

            return X_arr, feature_names

        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input feature array is empty.")
            if X.ndim != 2:
                raise ValueError(f"Input feature array must be 2D; received {X.ndim}D array.")

            if self.is_fitted_ and self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature count mismatch: model was fitted with {self.n_features_in_} features, "
                    f"received array with {X.shape[1]} features."
                )

            if not np.issubdtype(X.dtype, np.number):
                raise TypeError(f"Feature array must be numeric; received dtype {X.dtype}.")

            X_arr = X.astype(float)
            if np.isinf(X_arr).any():
                raise ValueError("Feature matrix contains infinite (+inf or -inf) values.")

            return X_arr, self.feature_names_

        else:
            raise TypeError(f"Input X must be a pandas DataFrame or numpy array; received {type(X).__name__}.")

    def _validate_targets(
        self,
        y: pd.Series | np.ndarray | Sequence[Any],
        require_both_classes: bool = True,
    ) -> np.ndarray:
        """Validate target labels for binary classification (0=baseline, 1=stress).

        Parameters
        ----------
        y : pd.Series | np.ndarray | Sequence[Any]
            Target labels to validate.
        require_both_classes : bool, default=True
            Whether to require both binary classes {0, 1} to be present.

        Returns
        -------
        np.ndarray
            Validated 1D integer array containing only 0 and 1.
        """
        if y is None:
            raise ValueError("Target array y cannot be None.")

        if isinstance(y, (pd.Series, pd.Index)):
            y_raw = y.to_numpy()
        elif isinstance(y, np.ndarray):
            y_raw = y
        elif isinstance(y, (list, tuple)):
            y_raw = np.array(y)
        else:
            raise TypeError(f"Targets y must be a Series, array, or list; received {type(y)}.")

        if y_raw.size == 0:
            raise ValueError("Target array y is empty.")
        if y_raw.ndim != 1:
            raise ValueError(f"Targets y must be 1-dimensional; received {y_raw.ndim}D array.")

        # Reject non-numeric dtypes (objects, strings, etc.)
        if not np.issubdtype(y_raw.dtype, np.number) and not np.issubdtype(y_raw.dtype, np.bool_):
            raise TypeError(f"Target array y must be numeric; received dtype {y_raw.dtype}.")

        # Reject missing, NaN, or infinite values
        if np.isnan(y_raw).any() or np.isinf(y_raw).any():
            raise ValueError("Target array y contains missing, NaN, or infinite values.")

        # Reject fractional labels or out-of-range labels (do not silently truncate e.g. 0.8 -> 0)
        if not np.all(np.isin(y_raw, [0, 1])):
            raise ValueError(
                "Targets y must contain only binary class labels {0, 1} without fractional or out-of-range values."
            )

        y_arr = y_raw.astype(int)

        if require_both_classes:
            unique_classes = set(np.unique(y_arr))
            if unique_classes != {0, 1}:
                raise ValueError(
                    f"Fitting requires both binary classes {{0, 1}} to be present; found classes: {sorted(unique_classes)}."
                )

        return y_arr

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "StressRandomForestClassifier":
        """Fit imputer and Random Forest classifier on training data only.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix with shape (n_samples, n_features).
        y : pd.Series | np.ndarray
            Binary target labels with shape (n_samples,).

        Returns
        -------
        StressRandomForestClassifier
            Fitted classifier instance.
        """
        X_arr, feature_names = self._validate_features(X)
        y_arr = self._validate_targets(y, require_both_classes=True)

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Sample count mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} elements."
            )

        # Detect features that are entirely missing in the training data
        all_missing = np.isnan(X_arr).all(axis=0)
        if all_missing.any():
            missing_idx = np.where(all_missing)[0]
            missing_cols = [
                feature_names[i] if feature_names else f"feature_{i}"
                for i in missing_idx
            ]
            raise ValueError(
                f"Feature column(s) contain only missing values in the training set: {missing_cols}. "
                f"Features that are entirely missing cannot be imputed."
            )

        self.fitted_with_dataframe_ = isinstance(X, pd.DataFrame)
        self.feature_names_ = feature_names if self.fitted_with_dataframe_ else None
        self.n_features_in_ = X_arr.shape[1]

        # Fit imputer strictly on training data
        X_imputed = self.imputer.fit_transform(X_arr)

        # Fit random forest estimator
        self.classifier.fit(X_imputed, y_arr)
        self.classes_ = self.classifier.classes_
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary stress class labels (0=baseline, 1=stress).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix to classify.

        Returns
        -------
        np.ndarray
            Predicted binary class labels (0 or 1).
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier must be fitted before calling predict.")

        X_arr, _ = self._validate_features(X)
        X_imputed = self.imputer.transform(X_arr)
        return self.classifier.predict(X_imputed)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities for input features.

        Guarantees that column 0 corresponds to class 0 (baseline) and column 1
        corresponds to class 1 (stress).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix to evaluate.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) where:
            - Column 0: P(class = 0, baseline)
            - Column 1: P(class = 1, stress)
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier must be fitted before calling predict_proba.")

        X_arr, _ = self._validate_features(X)
        X_imputed = self.imputer.transform(X_arr)
        probs = self.classifier.predict_proba(X_imputed)

        # Explicitly map column indices to guaranteed class order [0, 1]
        class_to_col = {cls_val: idx for idx, cls_val in enumerate(self.classifier.classes_)}
        return np.column_stack([probs[:, class_to_col[0]], probs[:, class_to_col[1]]])

    def feature_importances(self) -> pd.Series:
        """Return Gini feature importances indexed by feature name if available.

        Returns
        -------
        pd.Series
            Feature importances sorted descending.
        """
        if not self.is_fitted_:
            raise RuntimeError("Classifier must be fitted before accessing feature_importances.")

        importances = self.classifier.feature_importances_
        if self.feature_names_ is not None:
            return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def save(self, filepath: str | Path) -> None:
        """Serialize fitted model to disk.

        Parameters
        ----------
        filepath : str | Path
            Destination file path.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> "StressRandomForestClassifier":
        """Load serialized model from disk.

        Parameters
        ----------
        filepath : str | Path
            Source model file path.

        Returns
        -------
        StressRandomForestClassifier
            Loaded model instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is of type {type(model)}, expected {cls}.")
        return model
