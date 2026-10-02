"""
Representation Stability Index (RSI) analysis utilities.

This module implements the manuscript's revised RSI definition as a
domain-specific descriptive dispersion measure for representation-level
performance.

RSI is NOT treated as:
    * a variance estimator,
    * a universal stability criterion,
    * a threshold-based categorical classifier,
    * or an independently validated statistical test.

For a given Dataset × Architecture condition, let the representation-level
metric values be:

    x_1, x_2, ..., x_R

where R = 3 representations:

    Flow, Header, Hybrid.

The RSI is defined as the relative dispersion of the representation-level
values:

    RSI = SD(x_1, ..., x_R) / Mean(x_1, ..., x_R)

where SD is the sample standard deviation across representations.

For the manuscript's reported RSI table, the intended aggregation is across
the three benchmark datasets for each architecture. The resulting values are
reported as:

    mean ± SD

across the dataset-specific RSI values.

The current manuscript does not assign categorical labels or operational
thresholds to RSI. Earlier threshold/category concepts are intentionally not
implemented here.

Expected result structure
-------------------------
The primary input is a long-format table containing:

    Dataset
    Representation
    Architecture
    Seed
    Metric

where Metric is typically Macro_F1 or MCC.

Two analysis modes are supported:

1. Run-level RSI:
   For each Dataset × Architecture × Seed, calculate RSI across the three
   representation-level values.

2. Dataset-level RSI:
   First average each Dataset × Representation × Architecture condition
   across the available runs/seeds, then calculate RSI across Flow/Header/
   Hybrid.

The second mode corresponds to the manuscript's presentation of RSI as a
dataset-level descriptive measure, followed by aggregation across datasets.

No historical result values are embedded in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPRESENTATIONS = (
    "Flow",
    "Header",
    "Hybrid",
)

DEFAULT_DATASETS = (
    "CIC-DDoS2019",
    "CICIoT2023",
    "CIC-IoT-DIAD2024",
)

DEFAULT_ARCHITECTURES = (
    "LR",
    "RF",
    "XGB",
    "CNN",
    "BiLSTM",
)

DEFAULT_METRICS = (
    "Macro_F1",
    "MCC",
)


def normalise_column_name(
    name: str,
) -> str:
    """Normalise a column name for alias matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    """Find a dataframe column using format-insensitive aliases."""
    lookup = {
        normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = normalise_column_name(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            "Required column not found. "
            f"Accepted names: {list(candidates)}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def standardise_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert common result-column variants to canonical names.
    """
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
        "Seed": (
            "Seed",
            "seed",
            "Run",
            "run",
        ),
        "Macro_F1": (
            "Macro_F1",
            "macro_f1",
            "Macro F1",
            "F1_Macro",
        ),
        "MCC": (
            "MCC",
            "mcc",
            "Matthews_Correlation_Coefficient",
        ),
    }

    rename_map = {}

    for canonical, candidates in aliases.items():
        column = find_column(
            dataframe,
            candidates,
            required=False,
        )

        if column is not None and column != canonical:
            rename_map[column] = canonical

    result = dataframe.rename(
        columns=rename_map
    ).copy()

    for column in (
        "Dataset",
        "Representation",
        "Architecture",
    ):
        find_column(
            result,
            (column,),
            required=True,
        )

    return result


def validate_rsi_input(
    dataframe: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Validate the long-format input required for RSI analysis.

    The function does not aggregate or alter metric values.
    """
    data = standardise_columns(
        dataframe
    )

    if metric not in data.columns:
        raise ValueError(
            f"Metric '{metric}' is not present in the input."
        )

    required = (
        "Dataset",
        "Representation",
        "Architecture",
    )

    for column in required:
        if data[column].isna().any():
            raise ValueError(
                f"Missing values found in '{column}'."
            )

    data[metric] = pd.to_numeric(
        data[metric],
        errors="coerce",
    )

    if data[metric].isna().any():
        raise ValueError(
            f"Metric '{metric}' contains missing or non-numeric values."
        )

    return data


def representation_values_complete(
    dataframe: pd.DataFrame,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
) -> None:
    """
    Validate that every Dataset × Architecture condition contains exactly
    one value for each requested representation.

    This function is intended for an already representation-level aggregated
    table, not raw repeated-run data.
    """
    counts = (
        dataframe
        .groupby(
            [
                "Dataset",
                "Architecture",
            ],
            dropna=False,
        )["Representation"]
        .nunique()
    )

    incomplete = counts[
        counts != len(representations)
    ]

    if not incomplete.empty:
        raise ValueError(
            "Incomplete representation sets detected. "
            "Each Dataset × Architecture condition must contain "
            f"{len(representations)} representations. "
            f"Problematic conditions:\n{incomplete}"
        )


def calculate_rsi(
    values: Sequence[float],
) -> float:
    """
    Calculate RSI across representation-level values.

        RSI = sample SD / arithmetic mean

    A zero mean is rejected because relative dispersion would be undefined.
    """
    array = np.asarray(
        values,
        dtype=float,
    )

    if array.ndim != 1:
        raise ValueError(
            "RSI input must be one-dimensional."
        )

    if len(array) < 2:
        raise ValueError(
            "At least two representation values are required."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "RSI values must be finite."
        )

    mean = float(
        np.mean(array)
    )

    if np.isclose(
        mean,
        0.0,
    ):
        raise ZeroDivisionError(
            "RSI is undefined when the representation mean is zero."
        )

    sd = float(
        np.std(
            array,
            ddof=1,
        )
    )

    return sd / mean


def calculate_rsi_components(
    values: Sequence[float],
) -> dict[str, float]:
    """Return representation mean, SD, and RSI."""
    array = np.asarray(
        values,
        dtype=float,
    )

    if len(array) < 2:
        raise ValueError(
            "At least two representation values are required."
        )

    mean = float(
        np.mean(array)
    )

    sd = float(
        np.std(
            array,
            ddof=1,
        )
    )

    if np.isclose(
        mean,
        0.0,
    ):
        raise ZeroDivisionError(
            "RSI is undefined when the representation mean is zero."
        )

    return {
        "representation_mean": mean,
        "representation_sd": sd,
        "RSI": sd / mean,
    }


def aggregate_runs_to_dataset_representation(
    dataframe: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Average repeated runs within each Dataset × Representation × Architecture.

    This is the appropriate first step when the input is the primary
    run-level experiment table and the desired RSI is dataset-level.
    """
    data = validate_rsi_input(
        dataframe,
        metric=metric,
    )

    group_columns = [
        "Dataset",
        "Representation",
        "Architecture",
    ]

    result = (
        data
        .groupby(
            group_columns,
            dropna=False,
        )[metric]
        .agg(
            n_runs="count",
            mean="mean",
            sd="std",
        )
        .reset_index()
    )

    result = result.rename(
        columns={
            "mean": metric,
            "sd": f"{metric}_run_sd",
        }
    )

    return result


def calculate_dataset_level_rsi(
    dataframe: pd.DataFrame,
    metric: str,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
) -> pd.DataFrame:
    """
    Calculate one RSI for every Dataset × Architecture condition.

    Input:
        A Dataset × Representation × Architecture table with one metric
        value per condition.

    Output columns:
        Dataset
        Architecture
        Flow
        Header
        Hybrid
        representation_mean
        representation_sd
        RSI
    """
    data = validate_rsi_input(
        dataframe,
        metric=metric,
    )

    if "Seed" in data.columns:
        # Explicitly reject accidental use of the raw run-level table here.
        duplicated = data.duplicated(
            subset=[
                "Dataset",
                "Representation",
                "Architecture",
            ],
            keep=False,
        )

        if duplicated.any():
            raise ValueError(
                "calculate_dataset_level_rsi expects one value per "
                "Dataset × Representation × Architecture condition. "
                "For run-level results, first call "
                "aggregate_runs_to_dataset_representation()."
            )

    representation_values_complete(
        data,
        representations=representations,
    )

    rows = []

    for (dataset, architecture), group in data.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        values = []

        row = {
            "Dataset": dataset,
            "Architecture": architecture,
        }

        for representation in representations:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Expected exactly one {representation} value for "
                    f"{dataset} × {architecture}."
                )

            value = float(
                subset.iloc[0][metric]
            )

            row[representation] = value
            values.append(value)

        components = calculate_rsi_components(
            values
        )

        row.update(
            components
        )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_run_level_rsi(
    dataframe: pd.DataFrame,
    metric: str,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
) -> pd.DataFrame:
    """
    Calculate RSI independently for every Dataset × Architecture × Seed.

    This preserves the repeated-run structure and is useful for examining
    the distribution of RSI before dataset-level aggregation.
    """
    data = validate_rsi_input(
        dataframe,
        metric=metric,
    )

    if "Seed" not in data.columns:
        raise ValueError(
            "Run-level RSI requires a Seed/Run column."
        )

    rows = []

    group_columns = [
        "Dataset",
        "Architecture",
        "Seed",
    ]

    for keys, group in data.groupby(
        group_columns,
        sort=True,
    ):
        dataset, architecture, seed = keys

        row = {
            "Dataset": dataset,
            "Architecture": architecture,
            "Seed": seed,
        }

        values = []

        for representation in representations:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Expected exactly one {representation} value for "
                    f"{dataset} × {architecture} × Seed {seed}."
                )

            value = float(
                subset.iloc[0][metric]
            )

            row[representation] = value
            values.append(value)

        components = calculate_rsi_components(
            values
        )

        row.update(
            components
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_rsi_across_datasets(
    dataset_rsi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate dataset-level RSI values by architecture.

    The manuscript-style result is:

        mean ± SD

    across the three benchmark datasets.

    No categorical stability label is assigned.
    """
    required = (
        "Dataset",
        "Architecture",
        "RSI",
    )

    for column in required:
        if column not in dataset_rsi.columns:
            raise ValueError(
                f"Missing RSI column '{column}'."
            )

    rows = []

    for architecture, group in dataset_rsi.groupby(
        "Architecture",
        sort=True,
    ):
        values = group["RSI"].to_numpy(
            dtype=float
        )

        if len(values) == 0:
            continue

        rows.append(
            {
                "Architecture": architecture,
                "n_datasets": len(values),
                "RSI_mean": float(
                    np.mean(values)
                ),
                "RSI_SD": (
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    )
                    if len(values) > 1
                    else np.nan
                ),
                "RSI_min": float(
                    np.min(values)
                ),
                "RSI_max": float(
                    np.max(values)
                ),
                "RSI_mean_plus_minus_SD": (
                    f"{np.mean(values):.3f} ± "
                    f"{np.std(values, ddof=1):.3f}"
                    if len(values) > 1
                    else f"{np.mean(values):.3f} ± nan"
                ),
            }
        )

    return pd.DataFrame(rows)


def format_rsi_table(
    summary: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Produce a compact presentation table containing RSI as mean ± SD.
    """
    required = (
        "Architecture",
        "RSI_mean",
        "RSI_SD",
    )

    for column in required:
        if column not in summary.columns:
            raise ValueError(
                f"Missing column '{column}'."
            )

    result = summary[
        [
            "Architecture",
            "RSI_mean",
            "RSI_SD",
        ]
    ].copy()

    result["RSI (mean ± SD)"] = result.apply(
        lambda row: (
            f"{row['RSI_mean']:.{decimals}f} ± "
            f"{row['RSI_SD']:.{decimals}f}"
        ),
        axis=1,
    )

    return result[
        [
            "Architecture",
            "RSI (mean ± SD)",
        ]
    ]


def calculate_rsi_analysis(
    dataframe: pd.DataFrame,
    metric: str,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
) -> dict[str, pd.DataFrame]:
    """
    Execute the complete manuscript-oriented RSI workflow.

    Returns:
        run_level_rsi
        dataset_representation_means
        dataset_level_rsi
        architecture_summary
        formatted_summary
    """
    data = validate_rsi_input(
        dataframe,
        metric=metric,
    )

    run_level = calculate_run_level_rsi(
        data,
        metric=metric,
        representations=representations,
    )

    dataset_representation = (
        aggregate_runs_to_dataset_representation(
            data,
            metric=metric,
        )
    )

    dataset_level = calculate_dataset_level_rsi(
        dataset_representation,
        metric=metric,
        representations=representations,
    )

    summary = summarise_rsi_across_datasets(
        dataset_level
    )

    formatted = format_rsi_table(
        summary
    )

    return {
        "run_level_rsi": run_level,
        "dataset_representation_means": dataset_representation,
        "dataset_level_rsi": dataset_level,
        "architecture_summary": summary,
        "formatted_summary": formatted,
    }


def save_rsi_analysis(
    outputs: dict[str, pd.DataFrame],
    output_dir: str | Path,
    metric: str,
) -> None:
    """Save all RSI-analysis tables as CSV files."""
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_metric = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    filenames = {
        "run_level_rsi": (
            f"{safe_metric}_rsi_run_level.csv"
        ),
        "dataset_representation_means": (
            f"{safe_metric}_dataset_representation_means.csv"
        ),
        "dataset_level_rsi": (
            f"{safe_metric}_rsi_dataset_level.csv"
        ),
        "architecture_summary": (
            f"{safe_metric}_rsi_architecture_summary.csv"
        ),
        "formatted_summary": (
            f"{safe_metric}_rsi_formatted_summary.csv"
        ),
    }

    for key, filename in filenames.items():
        outputs[key].to_csv(
            output_dir / filename,
            index=False,
        )


def analyse_results_file(
    results_path: str | Path,
    output_dir: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load a primary run-level result CSV and perform RSI analysis for each
    requested metric available in the file.
    """
    results_path = Path(
        results_path
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    raw = pd.read_csv(
        results_path
    )

    outputs = {}

    for metric in metrics:
        if metric not in standardise_columns(
            raw
        ).columns:
            continue

        metric_outputs = calculate_rsi_analysis(
            raw,
            metric=metric,
        )

        metric_dir = (
            Path(output_dir)
            / metric.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        save_rsi_analysis(
            metric_outputs,
            metric_dir,
            metric=metric,
        )

        outputs[metric] = metric_outputs

    if not outputs:
        raise ValueError(
            "None of the requested RSI metrics were found in the result file."
        )

    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the descriptive Representation Stability "
            "Index (RSI) across Flow, Header, and Hybrid representations."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Run-level result CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/rsi_analysis"
        ),
        help="Directory for RSI-analysis outputs.",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated metrics. "
            "Default: Macro_F1,MCC"
        ),
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    metrics = tuple(
        metric.strip()
        for metric in args.metrics.split(",")
        if metric.strip()
    )

    outputs = analyse_results_file(
        results_path=args.results,
        output_dir=args.output_dir,
        metrics=metrics,
    )

    for metric, metric_outputs in outputs.items():
        print(
            f"\n{metric} — RSI summary"
        )
        print(
            metric_outputs[
                "formatted_summary"
            ].to_string(index=False)
        )


def _self_check() -> None:
    """
    Internal validation using synthetic values only.

    The synthetic values are not manuscript results.
    """
    rows = []

    for dataset_index, dataset in enumerate(
        DEFAULT_DATASETS
    ):
        for architecture_index, architecture in enumerate(
            DEFAULT_ARCHITECTURES
        ):
            for seed in (
                2026,
                2027,
                2028,
            ):
                base = (
                    0.90
                    + 0.001 * dataset_index
                    + 0.002 * architecture_index
                    + 0.0001 * (seed - 2026)
                )

                values = {
                    "Flow": base,
                    "Header": base + 0.01,
                    "Hybrid": base + 0.02,
                }

                for representation, value in values.items():
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Representation": representation,
                            "Architecture": architecture,
                            "Seed": seed,
                            "Macro_F1": value,
                            "MCC": value - 0.05,
                        }
                    )

    data = pd.DataFrame(rows)

    outputs = calculate_rsi_analysis(
        data,
        metric="Macro_F1",
    )

    assert len(
        outputs["run_level_rsi"]
    ) == 45

    assert len(
        outputs["dataset_level_rsi"]
    ) == 15

    assert len(
        outputs["architecture_summary"]
    ) == 5

    # Direct formula check.
    expected = calculate_rsi(
        [0.90, 0.91, 0.92]
    )

    direct = (
        np.std(
            [0.90, 0.91, 0.92],
            ddof=1,
        )
        / np.mean(
            [0.90, 0.91, 0.92]
        )
    )

    assert np.isclose(
        expected,
        direct,
    )

    print(
        "rsi_analysis.py self-check passed."
    )


if __name__ == "__main__":
    main()
