"""
Primary classical-machine-learning experiment runner.

This runner connects the repository components for the controlled
Dataset × Representation × Architecture × Run experiment:

    representation CSV
        -> model pipeline
        -> independent test prediction
        -> evaluation metrics
        -> optional probability calibration
        -> run-level result record

Supported classical architectures:
    LR   Logistic Regression
    RF   Random Forest
    XGB  XGBoost

Supported representations:
    Flow
    Header
    Hybrid

Supported datasets:
    CIC-DDoS2019
    CICIoT2023
    CIC-IoT-DIAD2024

Seeds:
    2026 ... 2035

Important reproducibility note
------------------------------
The available classical source (`04_run_pilot.py`) is explicitly identified
as a PILOT and directly demonstrates the model configurations used by this
runner. It uses seed 2026 only. This runner generalizes the documented model
configuration to the repository's 10-seed experimental protocol; it should
therefore be described as a repository reproducibility implementation, not
as a claim that the pilot file itself generated every historical manuscript
result.

Data expectations
-----------------
The runner expects already constructed representation files. It does not
construct representations or fit preprocessing statistics globally.

Expected filename patterns:

    CIC-DDoS2019:
        CICIDS2019_{split}_{representation}.csv

    CICIoT2023:
        CICIoT2023_{representation}_{split}.csv

    CIC-IoT-DIAD2024:
        DIAD2024_{split}_{representation}.csv

The runner supports label columns:
    _class
    Label
    label

`_source_file` is excluded from predictors.

Validation data
---------------
The model is fitted using the training partition only. The validation
partition is loaded for schema validation and reporting metadata, but the
primary fixed-hyperparameter classical models do not use validation data for
model selection.

This is intentional: representation-specific hyperparameter optimization
is treated separately as supplementary B6 analysis.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from classical_models import (
    MODEL_NAMES,
    build_model,
    model_display_name,
    model_configuration,
)
from evaluation import (
    calculate_all_metrics,
    format_metrics,
)
from calibration import expected_calibration_error


# ---------------------------------------------------------------------------
# Repository defaults
# ---------------------------------------------------------------------------

DEFAULT_REPRESENTATION_ROOT = Path("data")
DEFAULT_OUTPUT_ROOT = Path("results")

DATASETS = (
    "CIC-DDoS2019",
    "CICIoT2023",
    "CIC-IoT-DIAD2024",
)

REPRESENTATIONS = (
    "Flow",
    "Header",
    "Hybrid",
)

SEEDS = (
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
)

SPLITS = (
    "train",
    "validation",
    "test",
)

LABEL_CANDIDATES = (
    "_class",
    "Label",
    "label",
)

METRIC_COLUMNS = (
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


# ---------------------------------------------------------------------------
# Dataset filename resolution
# ---------------------------------------------------------------------------

def candidate_paths(
    data_root: Path,
    dataset: str,
    representation: str,
    split: str,
) -> list[Path]:
    """
    Return candidate paths using the filename conventions documented in the
    available experiment source.
    """
    if split not in SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. "
            f"Expected one of {SPLITS}."
        )

    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported representation '{representation}'."
        )

    if dataset == "CIC-DDoS2019":
        names = [
            f"CICIDS2019_{split}_{representation}.csv",
            f"CICIDS2019_{representation}_{split}.csv",
        ]

    elif dataset == "CICIoT2023":
        names = [
            f"CICIoT2023_{representation}_{split}.csv",
            f"CICIoT2023_{split}_{representation}.csv",
        ]

    elif dataset == "CIC-IoT-DIAD2024":
        names = [
            f"DIAD2024_{split}_{representation}.csv",
            f"DIAD2024_{representation}_{split}.csv",
        ]

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'."
        )

    candidates = []

    for name in names:
        candidates.extend(
            [
                data_root / dataset / representation / name,
                data_root / dataset / name,
                data_root / name,
            ]
        )

    # Preserve order while removing duplicates.
    unique = []
    seen = set()

    for path in candidates:
        resolved_key = str(path)

        if resolved_key not in seen:
            unique.append(path)
            seen.add(resolved_key)

    return unique


def resolve_data_file(
    data_root: Path,
    dataset: str,
    representation: str,
    split: str,
) -> Path:
    """Find the first existing representation CSV."""
    candidates = candidate_paths(
        data_root=data_root,
        dataset=dataset,
        representation=representation,
        split=split,
    )

    for path in candidates:
        if path.exists():
            return path

    formatted = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"No representation file found for "
        f"{dataset} / {representation} / {split}.\n"
        f"Checked:\n{formatted}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def identify_label_column(
    dataframe: pd.DataFrame,
    preferred: Optional[str] = None,
) -> str:
    """Identify the target label column."""
    if preferred is not None:
        if preferred not in dataframe.columns:
            raise ValueError(
                f"Requested label column '{preferred}' not found."
            )
        return preferred

    for candidate in LABEL_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        "No supported label column found. Expected one of: "
        + ", ".join(LABEL_CANDIDATES)
    )


def load_partition(
    path: Path,
    label_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load one partition and separate predictors from the label.

    `_source_file` and label columns are excluded from the predictor matrix.
    """
    dataframe = pd.read_csv(path)

    resolved_label = identify_label_column(
        dataframe=dataframe,
        preferred=label_column,
    )

    excluded = {
        resolved_label,
        "_source_file",
    }

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            f"No predictor features remain after excluding metadata "
            f"from {path}."
        )

    features = dataframe.loc[:, feature_columns].copy()

    # Match the available pilot source:
    # numeric coercion followed by inf -> NaN. The model pipeline performs
    # the model-specific median imputation and scaling.
    for column in feature_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    labels = dataframe[resolved_label].copy()

    return features, labels, resolved_label


def load_experiment_partitions(
    data_root: Path,
    dataset: str,
    representation: str,
    label_column: Optional[str] = None,
) -> dict:
    """
    Load train/validation/test partitions and verify a common feature schema.
    """
    paths = {
        split: resolve_data_file(
            data_root=data_root,
            dataset=dataset,
            representation=representation,
            split=split,
        )
        for split in SPLITS
    }

    loaded = {}

    for split, path in paths.items():
        X, y, resolved_label = load_partition(
            path=path,
            label_column=label_column,
        )

        loaded[split] = {
            "X": X,
            "y": y,
            "path": path,
            "label_column": resolved_label,
        }

    train_columns = list(
        loaded["train"]["X"].columns
    )

    for split in ("validation", "test"):
        columns = list(
            loaded[split]["X"].columns
        )

        if columns != train_columns:
            raise ValueError(
                f"Feature schema mismatch between training and "
                f"{split} partitions for "
                f"{dataset} / {representation}."
            )

    return loaded


# ---------------------------------------------------------------------------
# Random-state control
# ---------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Set Python and NumPy random states for a reproducibility run."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Label/probability handling
# ---------------------------------------------------------------------------

def align_probability_columns(
    model,
    probability_matrix: np.ndarray,
    class_labels: list,
) -> np.ndarray:
    """
    Align model probability columns to the requested class-label order.

    For normal sklearn classifiers, `model.classes_` defines the probability
    column order. If the model exposes classes not found in class_labels,
    an explicit error is raised.
    """
    probability_matrix = np.asarray(
        probability_matrix,
        dtype=float,
    )

    if not hasattr(model, "classes_"):
        return probability_matrix

    model_classes = list(model.classes_)

    if model_classes == list(class_labels):
        return probability_matrix

    if set(model_classes) != set(class_labels):
        raise ValueError(
            "Model classes do not match the expected evaluation classes. "
            f"Model classes: {model_classes}; "
            f"Expected: {class_labels}"
        )

    index = {
        label: position
        for position, label in enumerate(model_classes)
    }

    reorder = [
        index[label]
        for label in class_labels
    ]

    return probability_matrix[:, reorder]


def encode_target_labels(
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert labels to a stable string representation.

    No target-derived transformation is fitted here. The operation preserves
    the observed class names so sklearn can perform the multiclass fit.
    """
    train = y_train.astype(str).to_numpy()
    validation = y_validation.astype(str).to_numpy()
    test = y_test.astype(str).to_numpy()

    return train, validation, test


def observed_class_labels(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[str]:
    """
    Return class labels in the documented study order when available.

    Additional labels are retained in sorted order rather than silently
    discarded.
    """
    preferred = [
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
    ]

    observed = set(
        np.concatenate(
            [
                np.asarray(y_train, dtype=str),
                np.asarray(y_test, dtype=str),
            ]
        )
    )

    ordered = [
        label
        for label in preferred
        if label in observed
    ]

    additional = sorted(
        observed.difference(ordered)
    )

    return ordered + additional


# ---------------------------------------------------------------------------
# Single-run experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    data_root: Path,
    dataset: str,
    representation: str,
    model_name: str,
    seed: int,
    label_column: Optional[str] = None,
) -> dict:
    """
    Run one Dataset × Representation × Architecture × Seed experiment.

    The model is fitted on the training partition and evaluated on the
    independent test partition. Validation data are loaded and schema
    checked but are not used for hyperparameter selection.
    """
    set_random_seed(seed)

    started = time.perf_counter()

    partitions = load_experiment_partitions(
        data_root=data_root,
        dataset=dataset,
        representation=representation,
        label_column=label_column,
    )

    X_train = partitions["train"]["X"]
    y_train = partitions["train"]["y"]

    X_validation = partitions["validation"]["X"]
    y_validation = partitions["validation"]["y"]

    X_test = partitions["test"]["X"]
    y_test = partitions["test"]["y"]

    y_train_array, y_validation_array, y_test_array = (
        encode_target_labels(
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )
    )

    classes = observed_class_labels(
        y_train=y_train_array,
        y_test=y_test_array,
    )

    if len(classes) != 3:
        raise ValueError(
            "The documented primary experiment is three-class. "
            f"Observed classes for this run: {classes}"
        )

    model = build_model(
        model_name=model_name,
        random_state=seed,
    )

    model.fit(
        X_train,
        y_train_array,
    )

    # Validation is deliberately not used for model selection in the
    # controlled fixed-hyperparameter primary experiment. Its size is
    # nevertheless recorded for auditability.
    validation_rows = len(X_validation)

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )

    y_prob = align_probability_columns(
        model=model,
        probability_matrix=y_prob,
        class_labels=classes,
    )

    # sklearn prediction outputs are already labels for the fitted model.
    # Convert to strings for consistency with y_test.
    y_pred = np.asarray(
        y_pred,
        dtype=str,
    )

    metrics = calculate_all_metrics(
        y_true=y_test_array,
        y_pred=y_pred,
        y_prob=y_prob,
        class_labels=classes,
    )

    metrics["ece"] = expected_calibration_error(
        y_true=y_test_array,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bins=10,
    )

    elapsed = time.perf_counter() - started

    result = {
        "dataset": dataset,
        "representation": representation,
        "architecture": model_name,
        "architecture_name": model_display_name(model_name),
        "seed": seed,
        "n_features": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "validation_rows": int(validation_rows),
        "test_rows": int(len(X_test)),
        "label_column": partitions["train"]["label_column"],
        "training_time_seconds": float(elapsed),
        "status": "completed",
        "error": "",
    }

    result.update(
        {
            key: float(value)
            for key, value in metrics.items()
        }
    )

    return result


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def build_experiment_grid(
    datasets: tuple[str, ...] = DATASETS,
    representations: tuple[str, ...] = REPRESENTATIONS,
    models: tuple[str, ...] = MODEL_NAMES,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    """Build the complete controlled experiment grid."""
    return [
        {
            "dataset": dataset,
            "representation": representation,
            "model": model_name,
            "seed": seed,
        }
        for dataset in datasets
        for representation in representations
        for model_name in models
        for seed in seeds
    ]


def run_experiment_grid(
    data_root: Path = DEFAULT_REPRESENTATION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    datasets: tuple[str, ...] = DATASETS,
    representations: tuple[str, ...] = REPRESENTATIONS,
    models: tuple[str, ...] = MODEL_NAMES,
    seeds: tuple[int, ...] = SEEDS,
    label_column: Optional[str] = None,
    stop_on_error: bool = False,
) -> pd.DataFrame:
    """
    Run the requested classical experiment grid.

    Results are saved incrementally so that an interrupted long experiment
    does not lose all completed runs.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_root
        / "classical_primary_results.csv"
    )

    grid = build_experiment_grid(
        datasets=datasets,
        representations=representations,
        models=models,
        seeds=seeds,
    )

    records: list[dict] = []

    for number, configuration in enumerate(
        grid,
        start=1,
    ):
        total = len(grid)

        print(
            f"[{number}/{total}] "
            f"{configuration['dataset']} | "
            f"{configuration['representation']} | "
            f"{configuration['model']} | "
            f"seed={configuration['seed']}"
        )

        try:
            result = run_single_experiment(
                data_root=data_root,
                dataset=configuration["dataset"],
                representation=configuration["representation"],
                model_name=configuration["model"],
                seed=configuration["seed"],
                label_column=label_column,
            )

            records.append(result)

            # Save after every completed run.
            pd.DataFrame(records).to_csv(
                results_path,
                index=False,
            )

            print(
                f"    Macro-F1={result['macro_f1']:.6f} | "
                f"MCC={result['mcc']:.6f}"
            )

        except Exception as exc:
            error_record = {
                "dataset": configuration["dataset"],
                "representation": configuration["representation"],
                "architecture": configuration["model"],
                "architecture_name": model_display_name(
                    configuration["model"]
                ),
                "seed": configuration["seed"],
                "status": "failed",
                "error": repr(exc),
            }

            records.append(error_record)

            pd.DataFrame(records).to_csv(
                results_path,
                index=False,
            )

            print(
                f"    ERROR: {exc}"
            )

            if stop_on_error:
                raise

    result_dataframe = pd.DataFrame(records)

    if not result_dataframe.empty:
        result_dataframe.to_csv(
            results_path,
            index=False,
        )

    print(
        f"\nResults written to: {results_path}"
    )

    return result_dataframe


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_seed_list(value: str) -> tuple[int, ...]:
    """Parse comma-separated integer seeds."""
    seeds = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        seeds.append(int(item))

    if not seeds:
        raise argparse.ArgumentTypeError(
            "At least one seed must be supplied."
        )

    return tuple(seeds)


def parse_name_list(
    value: str,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    """Parse a comma-separated list and validate allowed values."""
    values = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        if item not in allowed:
            raise argparse.ArgumentTypeError(
                f"Unsupported value '{item}'. "
                f"Allowed values: {', '.join(allowed)}."
            )

        values.append(item)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one value must be supplied."
        )

    return tuple(values)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled classical ML "
            "Dataset × Representation × Architecture × Seed experiment."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_REPRESENTATION_ROOT,
        help="Root directory containing released representation CSV files.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for run-level results.",
    )

    parser.add_argument(
        "--datasets",
        type=lambda value: parse_name_list(
            value,
            DATASETS,
        ),
        default=DATASETS,
        help="Comma-separated dataset names.",
    )

    parser.add_argument(
        "--representations",
        type=lambda value: parse_name_list(
            value,
            REPRESENTATIONS,
        ),
        default=REPRESENTATIONS,
        help="Comma-separated representations.",
    )

    parser.add_argument(
        "--models",
        type=lambda value: parse_name_list(
            value,
            MODEL_NAMES,
        ),
        default=MODEL_NAMES,
        help="Comma-separated classical model keys.",
    )

    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=SEEDS,
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Optional explicit label column. If omitted, the runner "
            "detects _class, Label, or label."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a grid configuration fails.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    print("Classical reproducibility runner")
    print("--------------------------------")
    print("Models:")
    for model_name in args.models:
        print(
            f"  {model_name}: "
            f"{model_display_name(model_name)}"
        )

    print("\nModel configurations:")
    for model_name in args.models:
        print(
            model_configuration()[model_name]
        )

    dataframe = run_experiment_grid(
        data_root=args.data_root,
        output_root=args.output_root,
        datasets=args.datasets,
        representations=args.representations,
        models=args.models,
        seeds=args.seeds,
        label_column=args.label_column,
        stop_on_error=args.stop_on_error,
    )

    if dataframe.empty:
        print("No experiment records were produced.")
        return

    completed = (
        dataframe["status"] == "completed"
    ).sum()

    failed = (
        dataframe["status"] == "failed"
    ).sum()

    print(
        f"\nCompleted runs: {completed}"
    )
    print(
        f"Failed runs: {failed}"
    )

    if completed:
        print("\nCompleted-run metric summary:")
        available = [
            column
            for column in METRIC_COLUMNS
            if column in dataframe.columns
        ]

        summary = dataframe.loc[
            dataframe["status"] == "completed",
            available,
        ].mean(numeric_only=True)

        for metric, value in summary.items():
            print(
                f"  {metric}: {value:.6f}"
            )


if __name__ == "__main__":
    main()
