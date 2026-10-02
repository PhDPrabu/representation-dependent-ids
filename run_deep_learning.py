"""
Deep-learning experiment runner for the representation-dependent IDS study.

This runner connects:
    - released representation CSV files
    - preprocessing.py
    - deep_models.py
    - evaluation.py
    - calibration.py

Experimental design:
    Dataset × Representation × Architecture × Seed

Deep-learning architectures:
    CNN
    BiLSTM

Documented seeds:
    2026 ... 2035

Documented training configuration:
    Adam, learning_rate=0.001
    categorical_crossentropy
    batch_size=128
    maximum epochs=50
    early stopping on validation loss
    patience=8
    restore_best_weights=True

Important:
    The available full deep-learning source directly documents a 10-seed
    experiment over three datasets, three representations, and the two
    deep-learning architectures. This runner reorganizes that documented
    execution flow into the repository structure while preserving the
    model/training configuration.

Data-dependent preprocessing is fitted on the training partition only.
Validation and test data are transformed with the training-fitted
imputer/scaler.

The validation partition is used by Keras for early stopping, as in the
documented source. The test partition remains untouched until final
evaluation.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from calibration import expected_calibration_error
from deep_models import (
    MODEL_NAMES,
    build_model,
    compile_model,
    create_early_stopping,
)
from evaluation import calculate_all_metrics
from preprocessing import (
    identify_feature_columns,
    identify_label_column,
    convert_features_to_numeric,
    extract_labels,
    fit_training_preprocessor,
    transform_features,
    reshape_for_deep_learning,
)


DEFAULT_DATA_ROOT = Path("data")
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


def set_all_random_seeds(seed: int) -> None:
    """
    Set Python, NumPy, and TensorFlow random seeds.

    TensorFlow deterministic operations are requested where supported.
    Exact bitwise reproducibility can still depend on TensorFlow/CUDA/
    hardware/software versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TensorFlow versions may not expose this functionality.
        pass


def clear_tensorflow_state() -> None:
    """Release the current Keras model graph/state before another run."""
    tf.keras.backend.clear_session()
    gc.collect()


def candidate_paths(
    data_root: Path,
    dataset: str,
    representation: str,
    split: str,
) -> list[Path]:
    """Return candidate representation-file paths."""
    if split not in SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of {SPLITS}."
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

    unique = []
    seen = set()

    for path in candidates:
        key = str(path)

        if key not in seen:
            unique.append(path)
            seen.add(key)

    return unique


def resolve_data_file(
    data_root: Path,
    dataset: str,
    representation: str,
    split: str,
) -> Path:
    """Resolve an existing representation CSV."""
    candidates = candidate_paths(
        data_root=data_root,
        dataset=dataset,
        representation=representation,
        split=split,
    )

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"No representation file found for "
        f"{dataset} / {representation} / {split}.\n"
        f"Checked:\n{checked}"
    )


def load_partition(
    path: Path,
    label_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Load a partition while retaining metadata only for label extraction.

    Predictor columns exclude the label and `_source_file`.
    """
    dataframe = pd.read_csv(path)

    resolved_label = identify_label_column(
        dataframe.columns,
        preferred=label_column,
    )

    feature_columns = identify_feature_columns(
        dataframe=dataframe,
        label_column=resolved_label,
        metadata_columns=("_source_file",),
    )

    features = convert_features_to_numeric(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    labels = extract_labels(
        dataframe=dataframe,
        label_column=resolved_label,
    )

    return features, labels, resolved_label


def load_experiment_partitions(
    data_root: Path,
    dataset: str,
    representation: str,
    label_column: Optional[str] = None,
) -> dict:
    """
    Load train/validation/test and enforce a common ordered feature schema.
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


def encode_labels(
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Encode the three target classes using training-defined class ordering.

    The documented source fits the label encoder on training labels and
    applies it to validation/test labels. The repository uses the same
    principle explicitly here.
    """
    train_labels = y_train.astype(str).to_numpy()
    validation_labels = y_validation.astype(str).to_numpy()
    test_labels = y_test.astype(str).to_numpy()

    preferred_order = [
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
    ]

    observed_train = set(train_labels.tolist())

    classes = [
        label
        for label in preferred_order
        if label in observed_train
    ]

    additional = sorted(
        observed_train.difference(classes)
    )

    classes.extend(additional)

    if len(classes) != 3:
        raise ValueError(
            "The documented experiment expects three target classes. "
            f"Training labels contain: {classes}"
        )

    label_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    unknown_validation = set(validation_labels).difference(label_to_index)
    unknown_test = set(test_labels).difference(label_to_index)

    if unknown_validation:
        raise ValueError(
            "Validation contains labels not present in training: "
            f"{sorted(unknown_validation)}"
        )

    if unknown_test:
        raise ValueError(
            "Test contains labels not present in training: "
            f"{sorted(unknown_test)}"
        )

    y_train_encoded = np.asarray(
        [label_to_index[label] for label in train_labels],
        dtype=np.int32,
    )

    y_validation_encoded = np.asarray(
        [label_to_index[label] for label in validation_labels],
        dtype=np.int32,
    )

    y_test_encoded = np.asarray(
        [label_to_index[label] for label in test_labels],
        dtype=np.int32,
    )

    return (
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
        classes,
    )


def one_hot_labels(
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Convert integer class labels to one-hot float32 arrays."""
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError(
            "Encoded labels contain an invalid class index."
        )

    output = np.zeros(
        (len(labels), n_classes),
        dtype=np.float32,
    )

    output[
        np.arange(len(labels)),
        labels,
    ] = 1.0

    return output


def run_single_experiment(
    data_root: Path,
    dataset: str,
    representation: str,
    model_name: str,
    seed: int,
    output_root: Optional[Path] = None,
    label_column: Optional[str] = None,
) -> dict:
    """
    Run one Dataset × Representation × Architecture × Seed DL experiment.

    The validation partition is supplied to Keras for early stopping.
    The test partition is used only after training for final metrics.
    """
    set_all_random_seeds(seed)

    started = time.perf_counter()

    partitions = load_experiment_partitions(
        data_root=data_root,
        dataset=dataset,
        representation=representation,
        label_column=label_column,
    )

    X_train_raw = partitions["train"]["X"]
    X_validation_raw = partitions["validation"]["X"]
    X_test_raw = partitions["test"]["X"]

    y_train_raw = partitions["train"]["y"]
    y_validation_raw = partitions["validation"]["y"]
    y_test_raw = partitions["test"]["y"]

    (
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
        classes,
    ) = encode_labels(
        y_train=y_train_raw,
        y_validation=y_validation_raw,
        y_test=y_test_raw,
    )

    # Explicit training-only preprocessing.
    preprocessor = fit_training_preprocessor(
        X_train=X_train_raw,
        feature_columns=list(X_train_raw.columns),
    )

    X_train = transform_features(
        X=X_train_raw,
        preprocessor=preprocessor,
    )

    X_validation = transform_features(
        X=X_validation_raw,
        preprocessor=preprocessor,
    )

    X_test = transform_features(
        X=X_test_raw,
        preprocessor=preprocessor,
    )

    X_train = reshape_for_deep_learning(X_train)
    X_validation = reshape_for_deep_learning(X_validation)
    X_test = reshape_for_deep_learning(X_test)

    n_classes = len(classes)
    n_features = X_train.shape[1]

    y_train_one_hot = one_hot_labels(
        y_train_encoded,
        n_classes=n_classes,
    )

    y_validation_one_hot = one_hot_labels(
        y_validation_encoded,
        n_classes=n_classes,
    )

    model = build_model(
        model_name=model_name,
        n_features=n_features,
        n_classes=n_classes,
    )

    compile_model(
        model=model,
        learning_rate=0.001,
    )

    early_stopping = create_early_stopping(
        patience=8,
    )

    history = model.fit(
        X_train,
        y_train_one_hot,
        validation_data=(
            X_validation,
            y_validation_one_hot,
        ),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Final evaluation on untouched test data.
    y_prob = model.predict(
        X_test,
        batch_size=128,
        verbose=0,
    )

    y_pred_encoded = np.argmax(
        y_prob,
        axis=1,
    )

    y_pred_labels = np.asarray(
        [
            classes[index]
            for index in y_pred_encoded
        ],
        dtype=str,
    )

    y_test_labels = np.asarray(
        [
            classes[index]
            for index in y_test_encoded
        ],
        dtype=str,
    )

    metrics = calculate_all_metrics(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        y_prob=y_prob,
        class_labels=classes,
    )

    metrics["ece"] = expected_calibration_error(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        y_prob=y_prob,
        n_bins=10,
    )

    elapsed = time.perf_counter() - started

    result = {
        "dataset": dataset,
        "representation": representation,
        "architecture": model_name,
        "seed": seed,
        "n_features": int(n_features),
        "input_shape": str(tuple(X_train.shape[1:])),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "epochs_requested": 50,
        "epochs_trained": int(len(history.history["loss"])),
        "batch_size": 128,
        "learning_rate": 0.001,
        "early_stopping_patience": 8,
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

    # Optional probability persistence. This is deliberately opt-in because
    # prediction arrays can be large for the full experiment.
    if output_root is not None:
        probability_root = (
            output_root
            / "probabilities"
            / dataset
            / representation
        )

        probability_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            probability_root
            / f"{model_name}_seed_{seed}.npz",
            y_true=y_test_encoded,
            y_prob=y_prob,
            classes=np.asarray(classes),
            macro_f1=metrics["macro_f1"],
            mcc=metrics["mcc"],
            brier=metrics["brier_score"],
            ece=metrics["ece"],
        )

    clear_tensorflow_state()

    return result


def build_experiment_grid(
    datasets: tuple[str, ...] = DATASETS,
    representations: tuple[str, ...] = REPRESENTATIONS,
    models: tuple[str, ...] = MODEL_NAMES,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    """Build the complete deep-learning experiment grid."""
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
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    datasets: tuple[str, ...] = DATASETS,
    representations: tuple[str, ...] = REPRESENTATIONS,
    models: tuple[str, ...] = MODEL_NAMES,
    seeds: tuple[int, ...] = SEEDS,
    label_column: Optional[str] = None,
    stop_on_error: bool = False,
    save_probabilities: bool = False,
) -> pd.DataFrame:
    """
    Run the deep-learning experiment grid and save results incrementally.
    """
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_root
        / "deep_learning_primary_results.csv"
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
            run_output_root = (
                output_root
                if save_probabilities
                else None
            )

            result = run_single_experiment(
                data_root=data_root,
                dataset=configuration["dataset"],
                representation=configuration["representation"],
                model_name=configuration["model"],
                seed=configuration["seed"],
                output_root=run_output_root,
                label_column=label_column,
            )

            records.append(result)

            pd.DataFrame(records).to_csv(
                results_path,
                index=False,
            )

            print(
                f"    Macro-F1={result['macro_f1']:.6f} | "
                f"MCC={result['mcc']:.6f} | "
                f"epochs={result['epochs_trained']}"
            )

        except Exception as exc:
            error_record = {
                "dataset": configuration["dataset"],
                "representation": configuration["representation"],
                "architecture": configuration["model"],
                "seed": configuration["seed"],
                "status": "failed",
                "error": repr(exc),
            }

            records.append(error_record)

            pd.DataFrame(records).to_csv(
                results_path,
                index=False,
            )

            clear_tensorflow_state()

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


def parse_seed_list(value: str) -> tuple[int, ...]:
    """Parse comma-separated integer seeds."""
    values = []

    for item in value.split(","):
        item = item.strip()

        if item:
            values.append(int(item))

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one seed must be supplied."
        )

    return tuple(values)


def parse_name_list(
    value: str,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    """Parse and validate a comma-separated list."""
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
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled deep-learning "
            "Dataset × Representation × Architecture × Seed experiment."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing released representation CSV files.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for results and optional probabilities.",
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
        help="Comma-separated deep-learning model keys.",
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
        "--save-probabilities",
        action="store_true",
        help=(
            "Save test-set y_true, probabilities, classes, and calibration "
            "metrics for each completed run."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a grid configuration fails.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    print("Deep-learning reproducibility runner")
    print("------------------------------------")
    print(
        f"Datasets: {', '.join(args.datasets)}"
    )
    print(
        f"Representations: {', '.join(args.representations)}"
    )
    print(
        f"Models: {', '.join(args.models)}"
    )
    print(
        f"Seeds: {', '.join(map(str, args.seeds))}"
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
        save_probabilities=args.save_probabilities,
    )

    if dataframe.empty:
        print("No experiment records were produced.")
        return

    completed = int(
        (dataframe["status"] == "completed").sum()
    )

    failed = int(
        (dataframe["status"] == "failed").sum()
    )

    print(
        f"\nCompleted runs: {completed}"
    )
    print(
        f"Failed runs: {failed}"
    )

    if completed:
        completed_rows = dataframe[
            dataframe["status"] == "completed"
        ]

        for metric in (
            "accuracy",
            "macro_f1",
            "mcc",
            "brier_score",
            "ece",
        ):
            if metric in completed_rows.columns:
                value = completed_rows[metric].mean()
                print(
                    f"  mean {metric}: {value:.6f}"
                )


if __name__ == "__main__":
    main()
