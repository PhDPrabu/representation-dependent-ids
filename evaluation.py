"""
Common evaluation utilities for the representation-dependent IDS experiments.

The functions in this module implement the evaluation definitions documented
for the three-class experiments:

    Benign, SYN_Flood, UDP_Flood

Primary reported metrics:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    - Weighted F1
    - Balanced Accuracy
    - Multiclass Matthews Correlation Coefficient (MCC)
    - Multiclass Brier Score

This module deliberately keeps metric calculation separate from model
training. Both classical and deep-learning runners can call the same
functions so that the metric definitions remain consistent.

Important:
    Metrics are calculated on the independent evaluation/test partition.
    Any preprocessing parameters must be fitted on the training partition
    and applied unchanged to validation/test data before these functions
    are called.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


DEFAULT_CLASS_NAMES = (
    "Benign",
    "SYN_Flood",
    "UDP_Flood",
)


def _validate_inputs(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: Optional[np.ndarray] = None,
) -> None:
    """Validate the basic shape and consistency of evaluation inputs."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.ndim != 1:
        raise ValueError(
            f"y_true must be one-dimensional; received shape "
            f"{y_true_array.shape}."
        )

    if y_pred_array.ndim != 1:
        raise ValueError(
            f"y_pred must be one-dimensional; received shape "
            f"{y_pred_array.shape}."
        )

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    if len(y_true_array) == 0:
        raise ValueError("Evaluation inputs contain zero samples.")

    if y_prob is not None:
        probability_array = np.asarray(y_prob)

        if probability_array.ndim != 2:
            raise ValueError(
                "y_prob must be a two-dimensional array with shape "
                "(n_samples, n_classes)."
            )

        if probability_array.shape[0] != len(y_true_array):
            raise ValueError(
                "The number of rows in y_prob must match y_true."
            )


def _validate_probability_matrix(
    y_prob: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Validate and return a floating-point class-probability matrix."""
    probability_array = np.asarray(y_prob, dtype=float)

    if probability_array.ndim != 2:
        raise ValueError(
            "y_prob must have shape (n_samples, n_classes)."
        )

    if probability_array.shape[1] != n_classes:
        raise ValueError(
            f"Expected {n_classes} probability columns; received "
            f"{probability_array.shape[1]}."
        )

    if not np.all(np.isfinite(probability_array)):
        raise ValueError("y_prob contains NaN or infinite values.")

    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError(
            "y_prob contains values outside the [0, 1] interval."
        )

    row_sums = probability_array.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            "Each row of y_prob must sum to 1 within numerical tolerance."
        )

    return probability_array


def multiclass_brier_score(
    y_true: Sequence,
    y_prob: np.ndarray,
    class_labels: Optional[Sequence] = None,
) -> float:
    """
    Calculate the multiclass Brier Score used in the study.

    Formula:
        BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2

    where:
        N     = number of evaluation samples
        p_ic  = predicted probability for class c for sample i
        y_ic  = one-hot indicator for the true class

    Parameters
    ----------
    y_true:
        One-dimensional true class labels.
    y_prob:
        Array of class probabilities with shape
        (n_samples, n_classes).
    class_labels:
        Ordered class labels corresponding to the columns of y_prob.

    Returns
    -------
    float
        Multiclass Brier Score.
    """
    y_true_array = np.asarray(y_true)

    if class_labels is None:
        class_labels = DEFAULT_CLASS_NAMES

    class_labels = list(class_labels)
    n_classes = len(class_labels)

    _validate_inputs(y_true, y_true, y_prob)
    probability_array = _validate_probability_matrix(
        y_prob,
        n_classes=n_classes,
    )

    label_to_index = {
        label: index for index, label in enumerate(class_labels)
    }

    try:
        true_indices = np.asarray(
            [label_to_index[label] for label in y_true_array],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(
            f"y_true contains a class label not present in class_labels: "
            f"{exc.args[0]!r}"
        ) from exc

    one_hot = np.zeros_like(probability_array, dtype=float)
    one_hot[np.arange(len(y_true_array)), true_indices] = 1.0

    return float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1)))


def calculate_classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
) -> Dict[str, float]:
    """
    Calculate the classification metrics reported for the study.

    Precision, recall, and F1 are calculated with macro averaging for the
    primary class-balanced evaluation. Weighted F1 is retained as a
    complementary metric. Balanced accuracy and multiclass MCC are also
    reported.

    Returns
    -------
    dict
        Dictionary with manuscript-compatible metric names.
    """
    _validate_inputs(y_true, y_pred)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
    }


def calculate_all_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence] = None,
    ece: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics and, when supplied, probability metrics.

    Parameters
    ----------
    y_true:
        True class labels.
    y_pred:
        Predicted class labels.
    y_prob:
        Optional probability matrix. Required for Brier Score.
    class_labels:
        Ordered class labels corresponding to y_prob columns.
    ece:
        Optional precomputed Expected Calibration Error. ECE calculation
        itself is implemented in calibration.py so that calibration logic
        remains separate from classification metrics.

    Returns
    -------
    dict
        Combined evaluation metrics.
    """
    metrics = calculate_classification_metrics(y_true, y_pred)

    if y_prob is not None:
        if class_labels is None:
            class_labels = DEFAULT_CLASS_NAMES

        metrics["brier_score"] = multiclass_brier_score(
            y_true=y_true,
            y_prob=y_prob,
            class_labels=class_labels,
        )

    if ece is not None:
        metrics["ece"] = float(ece)

    return metrics


def metric_order() -> tuple[str, ...]:
    """
    Return the standard output order for result tables/files.

    Keeping a single order avoids accidental changes in result CSV schemas.
    """
    return (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "mcc",
        "brier_score",
        "ece",
    )


def format_metrics(
    metrics: Dict[str, float],
    decimals: int = 6,
) -> Dict[str, float]:
    """Return metrics rounded for display without changing calculation logic."""
    return {
        key: round(float(metrics[key]), decimals)
        for key in metric_order()
        if key in metrics
    }


if __name__ == "__main__":
    # Small self-check using the three study classes.
    y_true = np.array([
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
    ])

    y_pred = np.array([
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
        "SYN_Flood",
        "SYN_Flood",
        "UDP_Flood",
    ])

    y_prob = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.40, 0.50, 0.10],
        [0.05, 0.90, 0.05],
        [0.05, 0.10, 0.85],
    ])

    results = calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_labels=DEFAULT_CLASS_NAMES,
    )

    print("Evaluation self-check:")
    for name, value in format_metrics(results).items():
        print(f"{name}: {value}")
