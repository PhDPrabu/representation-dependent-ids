"""
Representation construction utilities for the reproducibility repository.

This module records and implements the representation-building logic that is
explicitly available from the study's representation-construction source.

Currently documented feature mapping
------------------------------------
The available source (`02_build_representations.py`) explicitly provides
the following CICIoT2023 representation definitions:

    Header:
        Header_Length
        Protocol Type
        Time_To_Live
        fin_flag_number
        syn_flag_number
        rst_flag_number
        psh_flag_number
        ack_flag_number
        ece_flag_number
        cwr_flag_number
        ack_count
        syn_count
        fin_count
        rst_count
        HTTP
        HTTPS
        DNS
        Telnet
        SMTP
        SSH
        IRC
        TCP
        UDP
        DHCP
        ARP
        ICMP
        IGMP
        IPv
        LLC

    Flow:
        Rate
        Tot sum
        Min
        Max
        AVG
        Std
        Tot size
        IAT
        Number
        Variance

    Hybrid:
        Header + Flow

The source also documents:
    - class-balanced sampling of 56,863 instances per target class;
    - target classes: Benign, SYN_Flood, UDP_Flood;
    - chunk size = 50,000;
    - random seed = 2026;
    - split ratio = 70% / 10% / 20%;
    - group-aware handling of identical feature vectors across partitions;
    - cross-split duplicate verification;
    - `_class` and `_source_file` retained in released representation
      files as metadata/labels rather than predictors;
    - no scaling/normalization/imputation during representation construction.

Scope limitation
----------------
The currently available representation-builder source explicitly contains
the CICIoT2023 feature lists. It does NOT provide verified feature lists for
CIC-DDoS2019 or CIC-IoT-DIAD2024.

Accordingly, this module does not invent feature mappings for those datasets.
For those datasets, the repository expects their released processed
representation files and corresponding feature-mapping documentation to be
provided separately.

This distinction is deliberate: a reproducibility repository should not
silently manufacture feature definitions that are not supported by the
available study source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Documented CICIoT2023 feature groups
# ---------------------------------------------------------------------------

CICIOT2023_HEADER_FEATURES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
]

CICIOT2023_FLOW_FEATURES = [
    "Rate",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

CICIOT2023_HYBRID_FEATURES = (
    CICIOT2023_HEADER_FEATURES
    + CICIOT2023_FLOW_FEATURES
)


REPRESENTATION_NAMES = (
    "Flow",
    "Header",
    "Hybrid",
)

SUPPORTED_EXPLICIT_DATASET = "CICIoT2023"


def get_feature_list(
    dataset: str,
    representation: str,
) -> list[str]:
    """
    Return the documented feature list for a dataset/representation pair.

    At present, only CICIoT2023 is supported because it is the only dataset
    for which the available representation-builder source provides explicit
    feature lists.
    """
    dataset_key = dataset.strip()
    representation_key = representation.strip()

    if dataset_key != SUPPORTED_EXPLICIT_DATASET:
        raise NotImplementedError(
            "An explicit representation feature mapping is currently "
            f"available only for {SUPPORTED_EXPLICIT_DATASET}. "
            f"No feature list is invented for '{dataset_key}'."
        )

    feature_map = {
        "Flow": CICIOT2023_FLOW_FEATURES,
        "Header": CICIOT2023_HEADER_FEATURES,
        "Hybrid": CICIOT2023_HYBRID_FEATURES,
    }

    if representation_key not in feature_map:
        raise ValueError(
            f"Unsupported representation '{representation}'. "
            f"Supported representations: {', '.join(REPRESENTATION_NAMES)}."
        )

    return list(feature_map[representation_key])


def validate_feature_mapping(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    label_column: str = "_class",
    metadata_columns: Sequence[str] = ("_source_file",),
) -> None:
    """
    Validate that a source dataframe contains the requested representation
    features and documented metadata columns.

    Extra source columns are allowed because the builder selects the
    representation explicitly.
    """
    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "The source dataframe is missing required representation "
            f"features: {missing}"
        )

    if label_column not in dataframe.columns:
        raise ValueError(
            f"Required label column '{label_column}' was not found."
        )

    missing_metadata = [
        column
        for column in metadata_columns
        if column not in dataframe.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Required representation metadata columns are missing: "
            f"{missing_metadata}"
        )


def construct_representation(
    dataframe: pd.DataFrame,
    dataset: str,
    representation: str,
    label_column: str = "_class",
    metadata_columns: Sequence[str] = ("_source_file",),
) -> pd.DataFrame:
    """
    Construct one representation from a source dataframe.

    The returned dataframe contains:
        representation features
        label column
        documented metadata columns

    No normalization, imputation, or other data-dependent transformation is
    performed here. Those operations belong to the downstream preprocessing
    pipeline.
    """
    feature_columns = get_feature_list(
        dataset=dataset,
        representation=representation,
    )

    validate_feature_mapping(
        dataframe=dataframe,
        feature_columns=feature_columns,
        label_column=label_column,
        metadata_columns=metadata_columns,
    )

    selected_columns = (
        list(feature_columns)
        + [label_column]
        + list(metadata_columns)
    )

    return dataframe.loc[:, selected_columns].copy()


def construct_all_ciciot2023_representations(
    dataframe: pd.DataFrame,
    label_column: str = "_class",
    metadata_columns: Sequence[str] = ("_source_file",),
) -> dict[str, pd.DataFrame]:
    """
    Construct Flow, Header, and Hybrid CICIoT2023 representations.
    """
    return {
        representation: construct_representation(
            dataframe=dataframe,
            dataset=SUPPORTED_EXPLICIT_DATASET,
            representation=representation,
            label_column=label_column,
            metadata_columns=metadata_columns,
        )
        for representation in REPRESENTATION_NAMES
    }


# ---------------------------------------------------------------------------
# Class-balanced sampling and partitioning documented by the available
# CICIoT2023 representation-builder source.
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_PER_CLASS = 56863
DEFAULT_CHUNK_SIZE = 50000
DEFAULT_RANDOM_SEED = 2026

DEFAULT_CLASS_PATTERNS = {
    "Benign": "*BenignTraffic*.csv",
    "SYN_Flood": "*SYN_Flood*.csv",
    "UDP_Flood": "*UDP_Flood*.csv",
}

DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "test": 0.20,
}


def validate_split_ratios(
    ratios: Mapping[str, float],
) -> None:
    """Validate the documented train/validation/test split ratios."""
    required = ("train", "validation", "test")

    missing = [
        name
        for name in required
        if name not in ratios
    ]

    if missing:
        raise ValueError(
            f"Missing split ratios: {missing}"
        )

    total = sum(float(ratios[name]) for name in required)

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0; received {total}."
        )

    for name in required:
        if float(ratios[name]) <= 0:
            raise ValueError(
                f"Split ratio '{name}' must be positive."
            )


def calculate_split_sizes(
    n_samples: int,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, int]:
    """
    Calculate integer train/validation/test sizes from documented ratios.

    The largest-remainder approach is used so the returned sizes sum exactly
    to n_samples.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    validate_split_ratios(ratios)

    names = ("train", "validation", "test")
    raw = np.asarray(
        [
            n_samples * float(ratios[name])
            for name in names
        ],
        dtype=float,
    )

    base = np.floor(raw).astype(int)
    remainder = int(n_samples - base.sum())

    if remainder > 0:
        fractional = raw - base
        order = np.argsort(-fractional)

        for index in order[:remainder]:
            base[index] += 1

    return {
        name: int(base[index])
        for index, name in enumerate(names)
    }


def sample_dataframe_by_class(
    dataframe: pd.DataFrame,
    class_column: str = "_class",
    sample_per_class: int = DEFAULT_SAMPLE_PER_CLASS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a class-balanced sample from an already loaded dataframe.

    This helper is intended for smaller in-memory workflows. The original
    study source uses reservoir sampling for memory-safe processing of large
    files; this function does not replace that streaming implementation.
    """
    if class_column not in dataframe.columns:
        raise ValueError(
            f"Class column '{class_column}' not found."
        )

    if sample_per_class <= 0:
        raise ValueError(
            "sample_per_class must be positive."
        )

    rng = np.random.default_rng(random_seed)
    selected = []

    for class_name in (
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
    ):
        class_rows = dataframe[
            dataframe[class_column] == class_name
        ]

        if len(class_rows) < sample_per_class:
            raise ValueError(
                f"Class '{class_name}' contains {len(class_rows)} rows, "
                f"but {sample_per_class} are required."
            )

        indices = rng.choice(
            len(class_rows),
            size=sample_per_class,
            replace=False,
        )

        selected.append(
            class_rows.iloc[indices]
        )

    result = pd.concat(
        selected,
        axis=0,
        ignore_index=True,
    )

    return result


def split_class_balanced_dataframe(
    dataframe: pd.DataFrame,
    class_column: str = "_class",
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Split each target class separately using the documented 70/10/20 ratio.

    Splitting classes independently preserves class balance in each partition.
    """
    validate_split_ratios(ratios)

    rng = np.random.default_rng(random_seed)

    partitions = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for class_name in (
        "Benign",
        "SYN_Flood",
        "UDP_Flood",
    ):
        class_rows = dataframe[
            dataframe[class_column] == class_name
        ].copy()

        if class_rows.empty:
            raise ValueError(
                f"No rows found for target class '{class_name}'."
            )

        permutation = rng.permutation(len(class_rows))
        shuffled = class_rows.iloc[permutation].reset_index(drop=True)

        sizes = calculate_split_sizes(
            n_samples=len(shuffled),
            ratios=ratios,
        )

        start = 0

        for partition_name in (
            "train",
            "validation",
            "test",
        ):
            end = start + sizes[partition_name]

            partitions[partition_name].append(
                shuffled.iloc[start:end]
            )

            start = end

    result = {}

    for partition_name in (
        "train",
        "validation",
        "test",
    ):
        result[partition_name] = pd.concat(
            partitions[partition_name],
            axis=0,
            ignore_index=True,
        )

    return result


def duplicate_keys(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
) -> set[tuple]:
    """
    Return exact feature-vector keys for duplicate checking.

    Labels and metadata are intentionally excluded. This permits detection of
    identical predictor vectors across partitions.
    """
    values = dataframe.loc[
        :,
        list(feature_columns),
    ].copy()

    # Normalize NaN to a common sentinel so identical missing-value patterns
    # are represented consistently in tuple keys.
    values = values.where(
        pd.notna(values),
        "__NAN__",
    )

    return set(
        map(
            tuple,
            values.to_numpy(dtype=object),
        )
    )


def find_cross_partition_duplicates(
    partitions: Mapping[str, pd.DataFrame],
    feature_columns: Sequence[str],
) -> dict[str, set[tuple]]:
    """
    Identify exact feature-vector duplicates between partition pairs.

    Returns a dictionary keyed by:
        train__validation
        train__test
        validation__test
    """
    names = ("train", "validation", "test")

    keys = {
        name: duplicate_keys(
            partitions[name],
            feature_columns,
        )
        for name in names
    }

    return {
        "train__validation": keys["train"] & keys["validation"],
        "train__test": keys["train"] & keys["test"],
        "validation__test": keys["validation"] & keys["test"],
    }


def verify_no_cross_partition_duplicates(
    partitions: Mapping[str, pd.DataFrame],
    feature_columns: Sequence[str],
) -> None:
    """
    Raise an error when exact feature-vector duplicates occur across splits.
    """
    duplicates = find_cross_partition_duplicates(
        partitions=partitions,
        feature_columns=feature_columns,
    )

    non_empty = {
        name: values
        for name, values in duplicates.items()
        if values
    }

    if non_empty:
        summary = {
            name: len(values)
            for name, values in non_empty.items()
        }

        raise ValueError(
            "Cross-partition duplicate feature vectors detected: "
            f"{summary}"
        )


def save_representation_partitions(
    partitions: Mapping[str, pd.DataFrame],
    output_directory: str | Path,
    dataset_prefix: str = "CICIoT2023",
    representation: str = "Flow",
) -> list[Path]:
    """
    Save train/validation/test representation files using the documented
    CICIoT2023 naming convention.

    Filenames:
        CICIoT2023_<representation>_train.csv
        CICIoT2023_<representation>_validation.csv
        CICIoT2023_<representation>_test.csv
    """
    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    for partition_name in (
        "train",
        "validation",
        "test",
    ):
        if partition_name not in partitions:
            raise ValueError(
                f"Missing partition '{partition_name}'."
            )

        path = (
            output_path
            / f"{dataset_prefix}_{representation}_{partition_name}.csv"
        )

        partitions[partition_name].to_csv(
            path,
            index=False,
        )

        paths.append(path)

    return paths


def representation_summary(
    dataset: str = SUPPORTED_EXPLICIT_DATASET,
) -> dict[str, object]:
    """
    Return a machine-readable summary of the explicitly documented
    representation definitions.
    """
    if dataset != SUPPORTED_EXPLICIT_DATASET:
        return {
            "dataset": dataset,
            "status": "feature_mapping_not_available_in_current_source",
            "representations": list(REPRESENTATION_NAMES),
        }

    return {
        "dataset": dataset,
        "status": "explicit_feature_mapping_available",
        "representations": {
            representation: {
                "n_features": len(
                    get_feature_list(
                        dataset,
                        representation,
                    )
                ),
                "features": get_feature_list(
                    dataset,
                    representation,
                ),
            }
            for representation in REPRESENTATION_NAMES
        },
    }


if __name__ == "__main__":
    print("Representation-builder self-check")
    print("---------------------------------")

    for representation in REPRESENTATION_NAMES:
        features = get_feature_list(
            SUPPORTED_EXPLICIT_DATASET,
            representation,
        )

        print(
            f"{representation}: "
            f"{len(features)} documented features"
        )

    print("\nDocumented class-balanced sampling:")
    print(f"  samples per class: {DEFAULT_SAMPLE_PER_CLASS}")
    print(f"  chunk size: {DEFAULT_CHUNK_SIZE}")
    print(f"  random seed: {DEFAULT_RANDOM_SEED}")

    print("\nDocumented split:")
    print(
        f"  train={DEFAULT_SPLIT_RATIOS['train']:.2f}, "
        f"validation={DEFAULT_SPLIT_RATIOS['validation']:.2f}, "
        f"test={DEFAULT_SPLIT_RATIOS['test']:.2f}"
    )

    print("\nCICIoT2023 representation summary:")
    summary = representation_summary()

    for representation, details in summary["representations"].items():
        print(
            f"  {representation}: "
            f"{details['n_features']} features"
        )

    print(
        "\nNote: feature mappings for CIC-DDoS2019 and "
        "CIC-IoT-DIAD2024 are not fabricated by this module."
    )
