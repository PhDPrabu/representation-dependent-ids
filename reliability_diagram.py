"""
Reliability-diagram utilities for the representation-dependent IDS study.

This module generates conventional reliability diagrams using the same
top-label, 10-bin calibration definition implemented in calibration.py.

Figure 8 scope documented for the revised manuscript:
    Dataset:
        CIC-DDoS2019

    Representations:
        Flow
        Header

    Architectures:
        LR
        RF
        XGB
        CNN
        BiLSTM

    Binning:
        10 equally spaced confidence bins

Axes:
    x = Mean predicted confidence
    y = Empirical accuracy

The diagonal y=x line represents perfect calibration.

The plotting code deliberately does not assign categorical calibration
labels such as "high", "moderate", or "low" calibration. Quantitative
interpretation remains with ECE and Brier Score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from calibration import (
    DEFAULT_N_BINS,
    reliability_arrays,
)


DEFAULT_DATASET = "CIC-DDoS2019"
DEFAULT_REPRESENTATIONS = (
    "Flow",
    "Header",
)
DEFAULT_ARCHITECTURES = (
    "LR",
    "RF",
    "XGB",
    "CNN",
    "BiLSTM",
)

DEFAULT_FIGURE_WIDTH = 19.0
DEFAULT_FIGURE_HEIGHT = 5.2
DEFAULT_DPI = 600


def load_probability_file(
    path: str | Path,
) -> dict:
    """
    Load a saved probability NPZ file produced by run_deep_learning.py.

    Expected arrays:
        y_true
        y_prob
        classes

    Optional stored values:
        macro_f1
        mcc
        brier
        ece
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Probability file not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        required = {
            "y_true",
            "y_prob",
            "classes",
        }

        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"Probability file {path} is missing: {sorted(missing)}"
            )

        result = {
            "y_true": np.asarray(data["y_true"]),
            "y_prob": np.asarray(data["y_prob"]),
            "classes": np.asarray(data["classes"]),
        }

        for key in (
            "macro_f1",
            "mcc",
            "brier",
            "ece",
        ):
            if key in data.files:
                result[key] = float(
                    np.asarray(data[key]).reshape(-1)[0]
                )

    return result


def decode_true_labels(
    y_true: np.ndarray,
    classes: Sequence,
) -> np.ndarray:
    """
    Convert integer encoded y_true values to class labels when necessary.

    The saved deep-learning probability files store y_true as integer class
    indices. String labels are also accepted for flexibility.
    """
    values = np.asarray(y_true)
    class_array = np.asarray(classes)

    if values.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional."
        )

    if np.issubdtype(values.dtype, np.integer):
        if np.any(values < 0) or np.any(
            values >= len(class_array)
        ):
            raise ValueError(
                "Integer y_true contains an invalid class index."
            )

        return class_array[values]

    return values.astype(str)


def predicted_labels_from_probability(
    y_prob: np.ndarray,
    classes: Sequence,
) -> np.ndarray:
    """Return predicted class labels from maximum predicted probability."""
    probabilities = np.asarray(
        y_prob,
        dtype=float,
    )
    class_array = np.asarray(classes)

    if probabilities.ndim != 2:
        raise ValueError(
            "y_prob must have shape (n_samples, n_classes)."
        )

    if probabilities.shape[1] != len(class_array):
        raise ValueError(
            "Number of probability columns does not match classes."
        )

    indices = np.argmax(
        probabilities,
        axis=1,
    )

    return class_array[indices]


def get_reliability_data(
    probability_data: dict,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """
    Calculate reliability-diagram arrays from one saved probability file.
    """
    y_true = decode_true_labels(
        probability_data["y_true"],
        probability_data["classes"],
    )

    y_pred = predicted_labels_from_probability(
        probability_data["y_prob"],
        probability_data["classes"],
    )

    confidence, accuracy, counts = reliability_arrays(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=probability_data["y_prob"],
        n_bins=n_bins,
    )

    return {
        "mean_confidence": confidence,
        "empirical_accuracy": accuracy,
        "counts": counts,
    }


def resolve_probability_file(
    probability_root: str | Path,
    dataset: str,
    representation: str,
    architecture: str,
    seed: int,
) -> Path:
    """
    Resolve a probability file produced by the deep-learning runner.

    Expected primary path:
        probabilities/
            <dataset>/
                <representation>/
                    <architecture>_seed_<seed>.npz

    A small set of alternate layouts is also checked to make the plotting
    utility usable with manually organized released probability files.
    """
    root = Path(probability_root)

    candidates = [
        root
        / dataset
        / representation
        / f"{architecture}_seed_{seed}.npz",
        root
        / dataset
        / representation
        / f"{architecture}_{seed}.npz",
        root
        / f"{dataset}_{representation}_{architecture}_seed_{seed}.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        "Could not locate the requested probability file.\n"
        f"Checked:\n{checked}"
    )


def plot_single_reliability_curve(
    ax,
    reliability_data: dict,
    label: str,
    linewidth: float = 2.2,
    marker_size: float = 5.0,
) -> None:
    """Plot one representation's reliability curve on an existing axis."""
    ax.plot(
        reliability_data["mean_confidence"],
        reliability_data["empirical_accuracy"],
        marker="o",
        linewidth=linewidth,
        markersize=marker_size,
        label=label,
    )


def configure_reliability_axis(
    ax,
    title: str,
) -> None:
    """Apply the common Figure 8 axis configuration."""
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(
        "Mean predicted confidence",
        fontsize=13,
    )

    ax.set_ylabel(
        "Empirical accuracy",
        fontsize=13,
    )

    ax.set_title(
        title,
        fontsize=14,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(
        True,
        alpha=0.25,
    )


def create_figure8(
    probability_root: str | Path,
    seed: int,
    dataset: str = DEFAULT_DATASET,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
    architectures: Sequence[str] = DEFAULT_ARCHITECTURES,
    n_bins: int = DEFAULT_N_BINS,
    figsize: tuple[float, float] = (
        DEFAULT_FIGURE_WIDTH,
        DEFAULT_FIGURE_HEIGHT,
    ),
    dpi: int = DEFAULT_DPI,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Create the revised Figure 8 reliability diagrams.

    One panel is created per architecture. Each panel contains the requested
    representation curves. The current manuscript scope is Flow and Header
    for CIC-DDoS2019 across five architectures.
    """
    if len(architectures) == 0:
        raise ValueError(
            "At least one architecture is required."
        )

    if len(representations) == 0:
        raise ValueError(
            "At least one representation is required."
        )

    if dataset != DEFAULT_DATASET:
        raise ValueError(
            "The revised Figure 8 configuration is scoped to "
            f"{DEFAULT_DATASET}."
        )

    for representation in representations:
        if representation not in DEFAULT_REPRESENTATIONS:
            raise ValueError(
                f"Figure 8 supports Flow and Header; received "
                f"'{representation}'."
            )

    for architecture in architectures:
        if architecture not in DEFAULT_ARCHITECTURES:
            raise ValueError(
                f"Unsupported Figure 8 architecture '{architecture}'."
            )

    figure, axes = plt.subplots(
        1,
        len(architectures),
        figsize=figsize,
        squeeze=False,
    )

    axes = axes[0]

    for axis, architecture in zip(
        axes,
        architectures,
    ):
        configure_reliability_axis(
            axis,
            title=architecture,
        )

        for representation in representations:
            probability_file = resolve_probability_file(
                probability_root=probability_root,
                dataset=dataset,
                representation=representation,
                architecture=architecture,
                seed=seed,
            )

            probability_data = load_probability_file(
                probability_file
            )

            reliability_data = get_reliability_data(
                probability_data=probability_data,
                n_bins=n_bins,
            )

            plot_single_reliability_curve(
                ax=axis,
                reliability_data=reliability_data,
                label=representation,
            )

        axis.legend(
            fontsize=12,
        )

    figure.suptitle(
        "Reliability Diagrams — CIC-DDoS2019",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.93),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return figure


def plot_from_npz_pairs(
    probability_files: dict[str, dict[str, str | Path]],
    n_bins: int = DEFAULT_N_BINS,
    figsize: tuple[float, float] = (
        DEFAULT_FIGURE_WIDTH,
        DEFAULT_FIGURE_HEIGHT,
    ),
    dpi: int = DEFAULT_DPI,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Create a reliability figure directly from explicitly supplied files.

    Parameters
    ----------
    probability_files:
        Mapping:
            architecture -> representation -> NPZ path

        Example:
            {
                "LR": {
                    "Flow": "...",
                    "Header": "...",
                },
                ...
            }

    This helper is useful when released probability files do not follow the
    default repository directory layout.
    """
    architectures = list(probability_files.keys())

    if not architectures:
        raise ValueError(
            "probability_files cannot be empty."
        )

    figure, axes = plt.subplots(
        1,
        len(architectures),
        figsize=figsize,
        squeeze=False,
    )

    axes = axes[0]

    for axis, architecture in zip(
        axes,
        architectures,
    ):
        configure_reliability_axis(
            axis,
            title=architecture,
        )

        for representation in DEFAULT_REPRESENTATIONS:
            if representation not in probability_files[architecture]:
                continue

            data = load_probability_file(
                probability_files[architecture][representation]
            )

            reliability_data = get_reliability_data(
                data,
                n_bins=n_bins,
            )

            plot_single_reliability_curve(
                ax=axis,
                reliability_data=reliability_data,
                label=representation,
            )

        axis.legend(
            fontsize=12,
        )

    figure.suptitle(
        "Reliability Diagrams — CIC-DDoS2019",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.93),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return figure


def parse_architectures(
    value: str,
) -> tuple[str, ...]:
    """Parse and validate Figure 8 architecture names."""
    values = tuple(
        item.strip()
        for item in value.split(",")
        if item.strip()
    )

    invalid = [
        value
        for value in values
        if value not in DEFAULT_ARCHITECTURES
    ]

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported architectures: {invalid}"
        )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one architecture is required."
        )

    return values


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the Figure 8 command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the revised CIC-DDoS2019 Flow/Header "
            "reliability diagrams."
        )
    )

    parser.add_argument(
        "--probability-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing saved test probability NPZ files."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Seed/run whose probability outputs are plotted. "
            "Default: 2026."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figures/Figure8_CICDDoS2019_Flow_Header_Reliability.png"
        ),
        help="Output image path.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help="Number of equally spaced confidence bins.",
    )

    parser.add_argument(
        "--architectures",
        type=parse_architectures,
        default=DEFAULT_ARCHITECTURES,
        help="Comma-separated architecture keys.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    figure = create_figure8(
        probability_root=args.probability_root,
        seed=args.seed,
        dataset=DEFAULT_DATASET,
        representations=DEFAULT_REPRESENTATIONS,
        architectures=args.architectures,
        n_bins=args.n_bins,
        output_path=args.output,
    )

    plt.close(figure)

    print(
        f"Reliability diagram written to: {args.output}"
    )


if __name__ == "__main__":
    main()
