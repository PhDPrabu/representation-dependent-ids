"""
Preprocessing utilities for the representation-dependent IDS experiments.

This module implements the data-handling rules documented in the manuscript
and the available experiment source code.

Core rules
----------
1. Predictor metadata columns are removed before model fitting.
2. Label columns are identified explicitly and kept separate from predictors.
3. Predictor values are converted to numeric values.
4. Positive/negative infinity is converted to NaN.
5. Data-dependent imputation/scaling parameters are learned from the
   training partition only.
6. Validation and test partitions are transformed using the training-fitted
   parameters without refitting.
7. Deep-learning inputs are returned as float32 arrays and can subsequently
   be reshaped to (samples, features, 1) by the deep-learning runner.

Important implementation distinction
------------------------------------
The available classical source (`04_run_pilot.py`) implements preprocessing
inside sklearn Pipelines:

    Logistic Regression:
        median imputation -> Min-Max scaling -> LR

    Random Forest:
        median imputation -> RF

    XGBoost:
        median imputation -> XGB

The deep-learning source explicitly performs:

    numeric conversion -> inf/NaN handling -> training-only median
    imputation -> training-only Min-Max scaling -> float32

Therefore this module exposes both:
    - a reusable explicit preprocessing path for deep learning and general
      analysis; and
    - helper functions for the same training-only transformations.

The classical model-specific pipelines remain defined in
`classical_models.py` and should be used directly by the classical runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


DEFAULT_LABEL_COLUMNS = (
    "_class",
    "Label",
    "label",
)

DEFAULT_METADATA_COLUMNS = (
    "_source_file",
)


@dataclass
class FittedPreprocessor:
    """
    Container for training-fitted preprocessing objects.

    Attributes
    ----------
    imputer:
        Training-fitted median imputer.
    scaler:
        Training-fitted Min-Max scaler.
    feature_columns:
        Ordered predictor columns used during fitting.
    """

    imputer: SimpleImputer
    scaler: MinMaxScaler
    feature_columns: list[str]


def identify_label_column(
    columns: Iterable[str],
    preferred: Optional[str] = None,
) -> str:
    """
    Identify the label column from the documented label conventions.

    Supported labels:
        _class
        Label
        label

    If `preferred` is supplied, it must be present.
    """
    columns = list(columns)

    if preferred is not None:
        if preferred not in columns:
            raise ValueError(
                f"Requested label column '{preferred}' was not found."
            )
        return preferred

    for candidate in DEFAULT_LABEL_COLUMNS:
        if candidate in columns:
            return candidate

    raise ValueError(
        "No supported label column was found. Expected one of: "
        + ", ".join(DEFAULT_LABEL_COLUMNS)
    )


def identify_feature_columns(
    dataframe: pd.DataFrame,
    label_column: Optional[str] = None,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> list[str]:
    """
    Return the ordered predictor columns.

    Label and documented metadata/source columns are excluded. The original
    column order is preserved.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    resolved_label = identify_label_column(
        dataframe.columns,
        preferred=label_column,
    )

    excluded = {resolved_label}
    excluded.update(metadata_columns)

    return [
        column
        for column in dataframe.columns
        if column not in excluded
    ]


def convert_features_to_numeric(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Convert predictor columns to numeric values.

    Values that cannot be converted are represented as NaN and handled by
    the training-fitted imputation stage.
    """
    output = dataframe.loc[:, list(feature_columns)].copy()

    for column in feature_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    output = output.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return output


def extract_labels(
    dataframe: pd.DataFrame,
    label_column: Optional[str] = None,
) -> pd.Series:
    """Extract the documented target label column without modifying it."""
    resolved_label = identify_label_column(
        dataframe.columns,
        preferred=label_column,
    )

    return dataframe[resolved_label].copy()


def validate_feature_schema(
    reference_columns: Sequence[str],
    dataframe: pd.DataFrame,
    dataset_name: str = "dataset",
) -> None:
    """
    Ensure a validation/test dataframe has the same predictor schema.

    Column order matters because the resulting matrix is consumed by
    numerical models in the documented feature order.
    """
    actual_columns = list(dataframe.columns)
    expected_columns = list(reference_columns)

    if actual_columns != expected_columns:
        missing = [
            column
            for column in expected_columns
            if column not in actual_columns
        ]
        unexpected = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        raise ValueError(
            f"Feature schema mismatch for {dataset_name}. "
            f"Missing columns: {missing}; "
            f"Unexpected columns: {unexpected}."
        )


def fit_training_preprocessor(
    X_train: pd.DataFrame | np.ndarray,
    feature_columns: Optional[Sequence[str]] = None,
) -> FittedPreprocessor:
    """
    Fit median imputation and Min-Max scaling using training data only.

    No validation or test observations are used to estimate either the
    imputation statistics or scaling parameters.
    """
    if isinstance(X_train, pd.DataFrame):
        if feature_columns is None:
            feature_columns = list(X_train.columns)

        train_array = X_train.loc[:, list(feature_columns)].to_numpy(
            dtype=float
        )
        ordered_columns = list(feature_columns)
    else:
        train_array = np.asarray(X_train, dtype=float)

        if train_array.ndim != 2:
            raise ValueError(
                "X_train must have shape (samples, features)."
            )

        if feature_columns is None:
            ordered_columns = [
                f"feature_{index}"
                for index in range(train_array.shape[1])
            ]
        else:
            ordered_columns = list(feature_columns)

    if train_array.ndim != 2:
        raise ValueError(
            "X_train must have shape (samples, features)."
        )

    if len(ordered_columns) != train_array.shape[1]:
        raise ValueError(
            "The number of feature columns does not match X_train."
        )

    train_array = np.where(
        np.isfinite(train_array),
        train_array,
        np.nan,
    )

    imputer = SimpleImputer(strategy="median")
    imputed_train = imputer.fit_transform(train_array)

    scaler = MinMaxScaler()
    scaler.fit(imputed_train)

    return FittedPreprocessor(
        imputer=imputer,
        scaler=scaler,
        feature_columns=ordered_columns,
    )


def transform_features(
    X: pd.DataFrame | np.ndarray,
    preprocessor: FittedPreprocessor,
) -> np.ndarray:
    """
    Transform features using a preprocessor already fitted on training data.

    The function never calls `fit` or `fit_transform`.
    """
    if isinstance(X, pd.DataFrame):
        validate_feature_schema(
            preprocessor.feature_columns,
            X,
            dataset_name="input data",
        )
        array = X.loc[
            :,
            preprocessor.feature_columns,
        ].to_numpy(dtype=float)
    else:
        array = np.asarray(X, dtype=float)

    if array.ndim != 2:
        raise ValueError(
            "X must have shape (samples, features)."
        )

    if array.shape[1] != len(preprocessor.feature_columns):
        raise ValueError(
            "Input feature count does not match the training schema."
        )

    array = np.where(
        np.isfinite(array),
        array,
        np.nan,
    )

    imputed = preprocessor.imputer.transform(array)
    scaled = preprocessor.scaler.transform(imputed)

    return np.asarray(scaled, dtype=np.float32)


def fit_transform_training_features(
    X_train: pd.DataFrame | np.ndarray,
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, FittedPreprocessor]:
    """
    Fit the preprocessing objects on training data and transform training
    data.

    Returns both the transformed training matrix and the fitted preprocessor
    required for validation/test transformation.
    """
    preprocessor = fit_training_preprocessor(
        X_train=X_train,
        feature_columns=feature_columns,
    )

    transformed = transform_features(
        X=X_train,
        preprocessor=preprocessor,
    )

    return transformed, preprocessor


def preprocess_split_dataframes(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: Optional[str] = None,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> dict:
    """
    Prepare train/validation/test DataFrames using training-only fitting.

    Returns
    -------
    dict
        Contains:
            X_train
            X_validation
            X_test
            y_train
            y_validation
            y_test
            feature_columns
            label_column
            preprocessor
    """
    resolved_label = identify_label_column(
        train_df.columns,
        preferred=label_column,
    )

    # Ensure the same label convention is present in all partitions.
    for name, dataframe in (
        ("validation", validation_df),
        ("test", test_df),
    ):
        identify_label_column(
            dataframe.columns,
            preferred=resolved_label,
        )

    feature_columns = identify_feature_columns(
        train_df,
        label_column=resolved_label,
        metadata_columns=metadata_columns,
    )

    train_raw = convert_features_to_numeric(
        train_df,
        feature_columns,
    )

    validation_raw = convert_features_to_numeric(
        validation_df,
        feature_columns,
    )

    test_raw = convert_features_to_numeric(
        test_df,
        feature_columns,
    )

    validate_feature_schema(
        feature_columns,
        train_raw,
        dataset_name="training data",
    )
    validate_feature_schema(
        feature_columns,
        validation_raw,
        dataset_name="validation data",
    )
    validate_feature_schema(
        feature_columns,
        test_raw,
        dataset_name="test data",
    )

    X_train, preprocessor = fit_transform_training_features(
        X_train=train_raw,
        feature_columns=feature_columns,
    )

    X_validation = transform_features(
        X=validation_raw,
        preprocessor=preprocessor,
    )

    X_test = transform_features(
        X=test_raw,
        preprocessor=preprocessor,
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": extract_labels(
            train_df,
            label_column=resolved_label,
        ).to_numpy(),
        "y_validation": extract_labels(
            validation_df,
            label_column=resolved_label,
        ).to_numpy(),
        "y_test": extract_labels(
            test_df,
            label_column=resolved_label,
        ).to_numpy(),
        "feature_columns": feature_columns,
        "label_column": resolved_label,
        "preprocessor": preprocessor,
    }


def reshape_for_deep_learning(
    X: np.ndarray,
) -> np.ndarray:
    """
    Convert a tabular feature matrix to the deep-learning input layout.

    Input:
        (samples, features)

    Output:
        (samples, features, 1)
    """
    array = np.asarray(X, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            "X must have shape (samples, features)."
        )

    return np.expand_dims(array, axis=-1)


def check_no_nonfinite_values(
    X: np.ndarray,
    name: str = "features",
) -> None:
    """Raise an error if transformed features contain non-finite values."""
    array = np.asarray(X)

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values after preprocessing."
        )


if __name__ == "__main__":
    # Self-check demonstrating the intended train-only fitting rule.
    train = pd.DataFrame(
        {
            "_class": [
                "Benign",
                "SYN_Flood",
                "UDP_Flood",
            ],
            "_source_file": [
                "train_a.csv",
                "train_b.csv",
                "train_c.csv",
            ],
            "feature_a": [1.0, 2.0, np.nan],
            "feature_b": [10.0, 20.0, 30.0],
        }
    )

    validation = pd.DataFrame(
        {
            "_class": [
                "Benign",
                "SYN_Flood",
            ],
            "_source_file": [
                "validation_a.csv",
                "validation_b.csv",
            ],
            "feature_a": [1.5, 3.0],
            "feature_b": [15.0, np.inf],
        }
    )

    test = pd.DataFrame(
        {
            "_class": [
                "UDP_Flood",
                "Benign",
            ],
            "_source_file": [
                "test_a.csv",
                "test_b.csv",
            ],
            "feature_a": [2.5, 0.5],
            "feature_b": [25.0, 5.0],
        }
    )

    prepared = preprocess_split_dataframes(
        train_df=train,
        validation_df=validation,
        test_df=test,
        label_column="_class",
    )

    print("Preprocessing self-check:")
    print("Feature columns:", prepared["feature_columns"])
    print("Label column:", prepared["label_column"])
    print("Training shape:", prepared["X_train"].shape)
    print("Validation shape:", prepared["X_validation"].shape)
    print("Test shape:", prepared["X_test"].shape)

    check_no_nonfinite_values(
        prepared["X_train"],
        name="X_train",
    )
    check_no_nonfinite_values(
        prepared["X_validation"],
        name="X_validation",
    )
    check_no_nonfinite_values(
        prepared["X_test"],
        name="X_test",
    )

    deep_input = reshape_for_deep_learning(
        prepared["X_train"]
    )
    print("Deep-learning input shape:", deep_input.shape)
