"""
Calibration utilities for the representation-dependent IDS experiments.

The manuscript uses a top-label Expected Calibration Error (ECE) with:
    - maximum predicted class probability as confidence;
    - predicted-label correctness as the accuracy indicator;
    - 10 equally spaced confidence bins;
    - sample-weighted absolute difference between empirical accuracy
      and mean confidence.

This module provides the numerical ECE calculation and the bin-level
information required for reliability diagrams.

Calibration is evaluated on the independent evaluation/test partition.
The reported calibration values should be interpreted with respect to the
evaluation distribution used in the experiment. In particular, values from
the controlled class-balanced experiments are not deployment-prevalence
calibration estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


DEFAULT_N_BINS = 10


@dataclass(frozen=True)
class ReliabilityBin:
    """
    Summary of one non-empty confidence bin.

    Attributes
    ----------
    bin_index:
        Zero-based bin index.
    lower_bound:
        Lower confidence boundary.
    upper_bound:
        Upper confidence boundary.
    count:
        Number of samples in the bin.
    mean_confidence:
        Mean top-label confidence in the bin.
    empirical_accuracy:
        Fraction of correct top-label predictions in the bin.
    absolute_gap:
        Absolute difference between empirical accuracy and mean confidence.
    """

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    absolute_gap: float


def _validate_inputs(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize calibration inputs."""
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    probability_array = np.asarray(y_prob, dtype=float)

    if true_array.ndim != 1:
        raise ValueError(
            f"y_true must be one-dimensional; received {true_array.shape}."
        )

    if pred_array.ndim != 1:
        raise ValueError(
            f"y_pred must be one-dimensional; received {pred_array.shape}."
        )

    if probability_array.ndim != 2:
        raise ValueError(
            "y_prob must have shape (n_samples, n_classes)."
        )

    if len(true_array) == 0:
        raise ValueError("Calibration inputs contain zero samples.")

    if len(true_array) != len(pred_array):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    if probability_array.shape[0] != len(true_array):
        raise ValueError(
            "The number of rows in y_prob must match y_true."
        )

    if probability_array.shape[1] < 2:
        raise ValueError(
            "y_prob must contain probabilities for at least two classes."
        )

    if not np.all(np.isfinite(probability_array)):
        raise ValueError(
            "y_prob contains NaN or infinite values."
        )

    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError(
            "y_prob contains values outside the [0, 1] interval."
        )

    if not np.allclose(
        probability_array.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Each row of y_prob must sum to 1 within numerical tolerance."
        )

    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    return true_array, pred_array, probability_array


def top_label_confidence(
    y_prob: np.ndarray,
) -> np.ndarray:
    """
    Return the maximum predicted class probability for each sample.

    This is the confidence definition used by the study's top-label ECE.
    """
    probability_array = np.asarray(y_prob, dtype=float)

    if probability_array.ndim != 2:
        raise ValueError(
            "y_prob must have shape (n_samples, n_classes)."
        )

    return np.max(probability_array, axis=1)


def top_label_correctness(
    y_true: Sequence,
    y_pred: Sequence,
) -> np.ndarray:
    """Return a Boolean correctness indicator for each prediction."""
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)

    if true_array.ndim != 1 or pred_array.ndim != 1:
        raise ValueError(
            "y_true and y_pred must both be one-dimensional."
        )

    if len(true_array) != len(pred_array):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    return true_array == pred_array


def _bin_edges(n_bins: int) -> np.ndarray:
    """Create equally spaced confidence-bin boundaries from 0 to 1."""
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_bins(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> list[ReliabilityBin]:
    """
    Calculate non-empty reliability-diagram bins.

    Bins are equally spaced over [0, 1]. The final bin includes confidence
    equal to 1.0. Empty bins are omitted from the returned list.

    For each non-empty bin:
        mean confidence = mean(max class probability)
        empirical accuracy = mean(predicted == true)
        gap = |empirical accuracy - mean confidence|
    """
    (
        true_array,
        pred_array,
        probability_array,
    ) = _validate_inputs(
        y_true,
        y_pred,
        y_prob,
        n_bins,
    )

    confidence = top_label_confidence(probability_array)
    correctness = top_label_correctness(true_array, pred_array)

    edges = _bin_edges(n_bins)

    # np.digitize places confidence=1.0 into index n_bins. We cap that
    # value so it belongs to the final [lower, upper] interval.
    bin_indices = np.digitize(
        confidence,
        edges[1:-1],
        right=False,
    )

    bins: list[ReliabilityBin] = []

    for index in range(n_bins):
        mask = bin_indices == index

        if not np.any(mask):
            continue

        confidence_values = confidence[mask]
        correctness_values = correctness[mask].astype(float)

        mean_confidence = float(np.mean(confidence_values))
        empirical_accuracy = float(np.mean(correctness_values))
        absolute_gap = abs(
            empirical_accuracy - mean_confidence
        )

        bins.append(
            ReliabilityBin(
                bin_index=index,
                lower_bound=float(edges[index]),
                upper_bound=float(edges[index + 1]),
                count=int(np.sum(mask)),
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=float(absolute_gap),
            )
        )

    return bins


def expected_calibration_error(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """
    Calculate the study's top-label Expected Calibration Error.

    Definition:
        ECE = sum_b (n_b / N)
              * |accuracy(b) - confidence(b)|

    where:
        n_b       = number of samples in bin b
        N         = total number of samples
        accuracy  = empirical prediction accuracy in bin b
        confidence = mean maximum predicted probability in bin b

    Empty bins contribute zero because they contain no samples.
    """
    (
        true_array,
        pred_array,
        probability_array,
    ) = _validate_inputs(
        y_true,
        y_pred,
        y_prob,
        n_bins,
    )

    bins = reliability_bins(
        y_true=true_array,
        y_pred=pred_array,
        y_prob=probability_array,
        n_bins=n_bins,
    )

    total = len(true_array)

    ece = sum(
        (bin_summary.count / total)
        * bin_summary.absolute_gap
        for bin_summary in bins
    )

    return float(ece)


def calibration_summary(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """
    Return ECE and the complete non-empty-bin reliability information.

    The returned dictionary is convenient for saving numerical calibration
    results or passing the bin information directly to a plotting routine.
    """
    ece = expected_calibration_error(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bins=n_bins,
    )

    bins = reliability_bins(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bins=n_bins,
    )

    return {
        "ece": ece,
        "n_bins": n_bins,
        "non_empty_bins": len(bins),
        "bins": bins,
    }


def reliability_arrays(
    y_true: Sequence,
    y_pred: Sequence,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return arrays suitable for reliability-diagram plotting.

    Returns
    -------
    mean_confidence:
        Mean predicted confidence for each non-empty bin.
    empirical_accuracy:
        Empirical accuracy for each non-empty bin.
    counts:
        Number of observations in each non-empty bin.
    """
    bins = reliability_bins(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bins=n_bins,
    )

    mean_confidence = np.asarray(
        [item.mean_confidence for item in bins],
        dtype=float,
    )

    empirical_accuracy = np.asarray(
        [item.empirical_accuracy for item in bins],
        dtype=float,
    )

    counts = np.asarray(
        [item.count for item in bins],
        dtype=int,
    )

    return (
        mean_confidence,
        empirical_accuracy,
        counts,
    )


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

    summary = calibration_summary(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bins=10,
    )

    print("Calibration self-check:")
    print(f"ECE: {summary['ece']:.6f}")
    print(f"Non-empty bins: {summary['non_empty_bins']}")

    for item in summary["bins"]:
        print(
            f"bin={item.bin_index}, "
            f"count={item.count}, "
            f"confidence={item.mean_confidence:.6f}, "
            f"accuracy={item.empirical_accuracy:.6f}, "
            f"gap={item.absolute_gap:.6f}"
        )
