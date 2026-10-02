"""
Sensitivity and robustness analysis utilities.

This module implements the sensitivity analyses documented for the
representation-dependent intrusion-detection study.

Supported analyses
------------------
1. Feature-order sensitivity
   Tests whether predictive performance changes when the feature columns
   are presented in different orders while preserving the same feature
   values.

2. Feature-level predictive-concentration audit
   Quantifies how strongly individual features are associated with the
   target labels using training data only. This is an audit of feature-level
   concentration and is not presented as proof of data leakage.

The functions are designed to operate on explicit experimental inputs and
run-level result files. They do not contain historical manuscript results.

Important methodological rules
------------------------------
* Test-set labels are never used to fit preprocessing or feature-selection
  transformations.
* Feature-order sensitivity changes column order only; it does not alter
  feature values.
* Predictive-concentration statistics intended for an audit are fitted or
  computed from training data only.
* No arbitrary "acceptable sensitivity" threshold is imposed.
* The analyses are descriptive robustness checks unless a statistical test
  is explicitly requested and supported by the supplied run-level data.

Feature-order sensitivity
-------------------------
For a fixed Dataset × Representation × Architecture condition, the
experiment can be repeated under alternative feature permutations.

For each permutation, the supplied model runner should be used to obtain
the same evaluation metric on the same independent test partition.

This module provides:
    * deterministic feature permutations
    * permutation manifests
    * comparison of baseline vs permuted results
    * absolute and relative performance changes
    * summary statistics across permutations

The module deliberately does not assume that a particular historical
permutation count or performance value was used unless supplied by the user.

Feature-level predictive-concentration audit
---------------------------------------------
The audit provides feature-level association summaries using training data.
For multiclass targets, mutual information is the primary generic
non-parametric association measure when scikit-learn is available.

Optional additional measures:
    * ANOVA F statistic
    * permutation-based target association

These are audit quantities, not causal measures and not standalone evidence
of leakage.

Expected result columns
-----------------------
For feature-order result comparisons:

    Dataset
    Representation
    Architecture
    Permutation
    Metric

or equivalent aliases.

For feature-level audit:

    feature matrix X
    training labels y

The module is intentionally independent of the primary model runners so
that the sensitivity procedures can be reused with both classical and deep
learning experiments.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_N_PERMUTATIONS = 10
DEFAULT_RANDOM_SEED = 2026

DEFAULT_ID_COLUMNS = (
    "_source_file",
    "_class",
    "Label",
    "label",
)

DEFAULT_FEATURE_ORDER_METRIC = "Macro_F1"


def stable_feature_order_hash(
    feature_names: Sequence[str],
) -> str:
    """
    Create a stable hash for a feature ordering.

    The hash depends on order, so:

        [A, B, C] != [C, B, A]

    while the underlying feature set remains the same.
    """
    payload = "\n".join(
        str(name)
        for name in feature_names
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()[:16]


def validate_feature_names(
    feature_names: Sequence[str],
) -> list[str]:
    """Validate and return an ordered feature-name list."""
    names = [
        str(name)
        for name in feature_names
    ]

    if not names:
        raise ValueError(
            "feature_names cannot be empty."
        )

    if len(set(names)) != len(names):
        duplicates = [
            name
            for name in set(names)
            if names.count(name) > 1
        ]

        raise ValueError(
            "Duplicate feature names detected: "
            f"{duplicates}"
        )

    return names


def generate_feature_permutations(
    feature_names: Sequence[str],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_identity: bool = True,
) -> list[list[str]]:
    """
    Generate deterministic feature-order permutations.

    The original order is retained as permutation 0 when
    include_identity=True.

    No feature values are changed.
    """
    names = validate_feature_names(
        feature_names
    )

    if n_permutations < 1:
        raise ValueError(
            "n_permutations must be at least 1."
        )

    rng = np.random.default_rng(
        random_seed
    )

    permutations: list[list[str]] = []

    if include_identity:
        permutations.append(
            names.copy()
        )

    while len(permutations) < (
        n_permutations
        + (1 if include_identity else 0)
    ):
        candidate = names.copy()
        rng.shuffle(
            candidate
        )

        candidate_hash = stable_feature_order_hash(
            candidate
        )

        existing_hashes = {
            stable_feature_order_hash(
                permutation
            )
            for permutation in permutations
        }

        if candidate_hash not in existing_hashes:
            permutations.append(
                candidate
            )

    return permutations


def permutation_manifest(
    feature_names: Sequence[str],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    include_identity: bool = True,
) -> pd.DataFrame:
    """
    Create a reproducibility manifest for generated feature permutations.
    """
    permutations = generate_feature_permutations(
        feature_names=feature_names,
        n_permutations=n_permutations,
        random_seed=random_seed,
        include_identity=include_identity,
    )

    rows = []

    for index, permutation in enumerate(
        permutations
    ):
        rows.append(
            {
                "Permutation": index,
                "Is_Identity": (
                    include_identity
                    and index == 0
                ),
                "Random_Seed": random_seed,
                "n_features": len(permutation),
                "Feature_Order_Hash": (
                    stable_feature_order_hash(
                        permutation
                    )
                ),
                "Feature_Order": "|".join(
                    permutation
                ),
            }
        )

    return pd.DataFrame(rows)


def apply_feature_order(
    dataframe: pd.DataFrame,
    feature_order: Sequence[str],
) -> pd.DataFrame:
    """
    Reorder feature columns without modifying feature values.

    Non-feature metadata columns are retained after the reordered feature
    columns.
    """
    feature_order = validate_feature_names(
        feature_order
    )

    missing = [
        feature
        for feature in feature_order
        if feature not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Feature columns missing from dataframe: "
            f"{missing}"
        )

    feature_set = set(
        feature_order
    )

    remaining = [
        column
        for column in dataframe.columns
        if column not in feature_set
    ]

    return dataframe[
        feature_order + remaining
    ].copy()


def verify_same_feature_values(
    original: pd.DataFrame,
    reordered: pd.DataFrame,
    feature_names: Sequence[str],
) -> bool:
    """
    Verify that feature-order permutation changed column order only.

    Values are compared after aligning both frames to the same canonical
    feature order.
    """
    names = validate_feature_names(
        feature_names
    )

    for name in names:
        if name not in original.columns:
            raise ValueError(
                f"'{name}' missing from original dataframe."
            )

        if name not in reordered.columns:
            raise ValueError(
                f"'{name}' missing from reordered dataframe."
            )

    left = original[
        names
    ].reset_index(
        drop=True
    )

    right = reordered[
        names
    ].reset_index(
        drop=True
    )

    return bool(
        np.allclose(
            left.to_numpy(
                dtype=float
            ),
            right.to_numpy(
                dtype=float
            ),
            equal_nan=True,
        )
    )


def standardise_sensitivity_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardise common feature-order result column aliases."""
    aliases = {
        "Dataset": (
            "Dataset",
            "dataset",
        ),
        "Representation": (
            "Representation",
            "representation",
            "Rep",
        ),
        "Architecture": (
            "Architecture",
            "architecture",
            "Model",
            "model",
        ),
        "Permutation": (
            "Permutation",
            "permutation",
            "Permutation_ID",
            "permutation_id",
        ),
        "Metric": (
            "Metric",
            "metric",
        ),
        "Seed": (
            "Seed",
            "seed",
            "Run",
            "run",
        ),
    }

    normalised = {
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): column
        for column in dataframe.columns
    }

    rename_map = {}

    for canonical, candidates in aliases.items():
        for candidate in candidates:
            key = (
                candidate
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            if key in normalised:
                source = normalised[key]

                if source != canonical:
                    rename_map[source] = canonical

                break

    return dataframe.rename(
        columns=rename_map
    ).copy()


def calculate_permutation_changes(
    dataframe: pd.DataFrame,
    metric: str = DEFAULT_FEATURE_ORDER_METRIC,
    baseline_permutation: int = 0,
    group_columns: Sequence[str] = (
        "Dataset",
        "Representation",
        "Architecture",
        "Seed",
    ),
) -> pd.DataFrame:
    """
    Compare each feature-order permutation with the baseline.

    Expected input:
        one row per experimental condition and permutation.

    The output includes:
        baseline metric
        permutation metric
        absolute change
        relative percentage change
    """
    data = standardise_sensitivity_columns(
        dataframe
    )

    required = list(
        group_columns
    ) + [
        "Permutation",
        metric,
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing feature-order result columns: "
            f"{missing}"
        )

    data[metric] = pd.to_numeric(
        data[metric],
        errors="coerce",
    )

    if data[metric].isna().any():
        raise ValueError(
            f"Metric '{metric}' contains missing/non-numeric values."
        )

    baseline = data[
        data["Permutation"] == baseline_permutation
    ][
        list(group_columns) + [metric]
    ].copy()

    if baseline.empty:
        raise ValueError(
            f"No baseline permutation {baseline_permutation} found."
        )

    baseline = baseline.rename(
        columns={
            metric: "Baseline_Metric"
        }
    )

    merged = data.merge(
        baseline,
        on=list(group_columns),
        how="left",
        validate="many_to_one",
    )

    merged["Metric_Change"] = (
        merged[metric]
        - merged["Baseline_Metric"]
    )

    merged["Relative_Change_Percent"] = np.where(
        np.isclose(
            merged["Baseline_Metric"],
            0.0,
        ),
        np.nan,
        (
            merged["Metric_Change"]
            / merged["Baseline_Metric"]
            * 100.0
        ),
    )

    return merged


def summarise_permutation_sensitivity(
    comparison: pd.DataFrame,
    metric: str = DEFAULT_FEATURE_ORDER_METRIC,
    group_columns: Sequence[str] = (
        "Dataset",
        "Representation",
        "Architecture",
    ),
    exclude_baseline: bool = True,
) -> pd.DataFrame:
    """
    Summarise feature-order sensitivity across permutations.

    No categorical interpretation is assigned.

    Reported quantities:
        n_permutations
        mean_metric
        sd_metric
        min_metric
        max_metric
        mean_absolute_change
        max_absolute_change
        mean_relative_change_percent
        max_absolute_relative_change_percent
    """
    data = comparison.copy()

    if exclude_baseline:
        data = data[
            data["Permutation"] != 0
        ].copy()

    rows = []

    for keys, group in data.groupby(
        list(group_columns),
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        metric_values = group[
            metric
        ].to_numpy(
            dtype=float
        )

        absolute_changes = group[
            "Metric_Change"
        ].to_numpy(
            dtype=float
        )

        relative_changes = group[
            "Relative_Change_Percent"
        ].to_numpy(
            dtype=float
        )

        row = {
            column: value
            for column, value in zip(
                group_columns,
                keys,
            )
        }

        row.update(
            {
                "n_permutations": len(
                    metric_values
                ),
                "mean_metric": float(
                    np.mean(
                        metric_values
                    )
                ),
                "sd_metric": (
                    float(
                        np.std(
                            metric_values,
                            ddof=1,
                        )
                    )
                    if len(metric_values) > 1
                    else np.nan
                ),
                "min_metric": float(
                    np.min(
                        metric_values
                    )
                ),
                "max_metric": float(
                    np.max(
                        metric_values
                    )
                ),
                "mean_absolute_change": float(
                    np.mean(
                        np.abs(
                            absolute_changes
                        )
                    )
                ),
                "max_absolute_change": float(
                    np.max(
                        np.abs(
                            absolute_changes
                        )
                    )
                ),
                "mean_relative_change_percent": (
                    float(
                        np.nanmean(
                            relative_changes
                        )
                    )
                    if np.isfinite(
                        relative_changes
                    ).any()
                    else np.nan
                ),
                "max_absolute_relative_change_percent": (
                    float(
                        np.nanmax(
                            np.abs(
                                relative_changes
                            )
                        )
                    )
                    if np.isfinite(
                        relative_changes
                    ).any()
                    else np.nan
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_numeric_training_features(
    X_train: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare training features for the feature-level audit.

    Only the supplied training data are used. No information from validation
    or test data is required.
    """
    if feature_columns is None:
        feature_columns = [
            column
            for column in X_train.columns
            if column not in DEFAULT_ID_COLUMNS
        ]

    feature_columns = validate_feature_names(
        feature_columns
    )

    missing = [
        column
        for column in feature_columns
        if column not in X_train.columns
    ]

    if missing:
        raise ValueError(
            f"Training feature columns missing: {missing}"
        )

    X = X_train[
        feature_columns
    ].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median values are calculated from training data only.
    medians = X.median(
        numeric_only=True
    )

    X = X.fillna(
        medians
    )

    if X.isna().any().any():
        unresolved = [
            column
            for column in X.columns
            if X[column].isna().any()
        ]

        raise ValueError(
            "Unable to impute non-finite training features: "
            f"{unresolved}"
        )

    return X, feature_columns


def encode_training_labels(
    y_train: Sequence,
) -> np.ndarray:
    """
    Encode training labels to integer classes.

    The encoder is fitted only on the supplied training labels.
    """
    from sklearn.preprocessing import LabelEncoder

    y = np.asarray(
        y_train
    )

    if y.ndim != 1:
        raise ValueError(
            "y_train must be one-dimensional."
        )

    encoder = LabelEncoder()

    return encoder.fit_transform(
        y
    )


def feature_mutual_information(
    X_train: pd.DataFrame,
    y_train: Sequence,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute training-only mutual information between each feature and target.

    Mutual information is a feature-association audit quantity. It should not
    be interpreted as a causal effect or as proof of information leakage.
    """
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
        )

    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for mutual information analysis."
        ) from exc

    X, feature_columns = prepare_numeric_training_features(
        X_train
    )

    y = encode_training_labels(
        y_train
    )

    values = mutual_info_classif(
        X,
        y,
        random_state=random_state,
    )

    result = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Mutual_Information": values,
        }
    )

    result = result.sort_values(
        "Mutual_Information",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result["Rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    return result[
        [
            "Rank",
            "Feature",
            "Mutual_Information",
        ]
    ]


def feature_anova_association(
    X_train: pd.DataFrame,
    y_train: Sequence,
) -> pd.DataFrame:
    """
    Compute one-way ANOVA F statistics for multiclass training labels.

    This is an association audit only. It does not establish causality or
    leakage.
    """
    try:
        from sklearn.feature_selection import (
            f_classif,
        )

    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ANOVA feature association."
        ) from exc

    X, feature_columns = prepare_numeric_training_features(
        X_train
    )

    y = encode_training_labels(
        y_train
    )

    f_values, p_values = f_classif(
        X,
        y,
    )

    result = pd.DataFrame(
        {
            "Feature": feature_columns,
            "ANOVA_F": f_values,
            "ANOVA_p_value": p_values,
        }
    )

    result = result.sort_values(
        "ANOVA_F",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result["Rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    return result[
        [
            "Rank",
            "Feature",
            "ANOVA_F",
            "ANOVA_p_value",
        ]
    ]


def merge_feature_audit_results(
    mutual_information: Optional[pd.DataFrame] = None,
    anova: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge optional feature-association audit tables by feature name.
    """
    if mutual_information is None and anova is None:
        raise ValueError(
            "At least one audit result is required."
        )

    if mutual_information is not None:
        result = mutual_information.copy()

    else:
        result = anova.copy()

    if (
        mutual_information is not None
        and anova is not None
    ):
        result = result.merge(
            anova[
                [
                    "Feature",
                    "ANOVA_F",
                    "ANOVA_p_value",
                ]
            ],
            on="Feature",
            how="outer",
        )

    return result.sort_values(
        "Feature"
    ).reset_index(
        drop=True
    )


def save_permutation_manifest(
    manifest: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a feature-order permutation manifest."""
    path = Path(
        path
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        path,
        index=False,
    )


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save a sensitivity-analysis dataframe as CSV."""
    path = Path(
        path
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate feature-order sensitivity manifests or "
            "analyse completed permutation results."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Generate a deterministic feature-order manifest.",
    )

    manifest_parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help=(
            "Text file containing one feature name per line."
        ),
    )

    manifest_parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Number of non-identity random permutations.",
    )

    manifest_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    manifest_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/sensitivity/feature_order_manifest.csv"
        ),
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Analyse completed feature-order sensitivity results.",
    )

    compare_parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )

    compare_parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_FEATURE_ORDER_METRIC,
    )

    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/sensitivity/feature_order"
        ),
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.command == "manifest":
        feature_lines = (
            args.features
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        feature_names = [
            line.strip()
            for line in feature_lines
            if line.strip()
        ]

        manifest = permutation_manifest(
            feature_names=feature_names,
            n_permutations=args.n_permutations,
            random_seed=args.seed,
        )

        save_permutation_manifest(
            manifest,
            args.output,
        )

        print(
            f"Feature-order manifest written to: {args.output}"
        )

    elif args.command == "compare":
        results = pd.read_csv(
            args.results
        )

        comparison = calculate_permutation_changes(
            results,
            metric=args.metric,
        )

        summary = summarise_permutation_sensitivity(
            comparison,
            metric=args.metric,
        )

        output_dir = Path(
            args.output_dir
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_dataframe(
            comparison,
            output_dir / "permutation_comparison.csv",
        )

        save_dataframe(
            summary,
            output_dir / "permutation_sensitivity_summary.csv",
        )

        print(
            f"Sensitivity comparison written to: "
            f"{output_dir / 'permutation_comparison.csv'}"
        )


def _self_check() -> None:
    """
    Internal validation using synthetic data only.
    """
    features = [
        "f1",
        "f2",
        "f3",
        "f4",
    ]

    permutations = generate_feature_permutations(
        feature_names=features,
        n_permutations=4,
        random_seed=2026,
    )

    assert len(permutations) == 5
    assert permutations[0] == features

    hashes = [
        stable_feature_order_hash(
            permutation
        )
        for permutation in permutations
    ]

    assert len(set(hashes)) == len(hashes)

    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "f3": [5.0, 6.0],
            "f4": [7.0, 8.0],
            "_class": ["A", "B"],
        }
    )

    reordered = apply_feature_order(
        frame,
        ["f4", "f2", "f1", "f3"],
    )

    assert verify_same_feature_values(
        frame,
        reordered,
        features,
    )

    result_rows = []

    for permutation in range(4):
        result_rows.append(
            {
                "Dataset": "CIC-DDoS2019",
                "Representation": "Flow",
                "Architecture": "LR",
                "Seed": 2026,
                "Permutation": permutation,
                "Metric": 0.90 - 0.001 * permutation,
            }
        )

    comparison = calculate_permutation_changes(
        pd.DataFrame(result_rows),
        metric="Metric",
    )

    assert np.isclose(
        comparison.loc[
            comparison["Permutation"] == 1,
            "Metric_Change",
        ].iloc[0],
        -0.001,
    )

    summary = summarise_permutation_sensitivity(
        comparison,
        metric="Metric",
    )

    assert len(summary) == 1

    print(
        "sensitivity_analysis.py self-check passed."
    )


if __name__ == "__main__":
    main()
