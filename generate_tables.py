"""
Table-generation utilities for the representation-dependent IDS manuscript.

This module converts run-level experimental results into manuscript-oriented
tables without embedding historical result values.

Primary experimental factors
-----------------------------
Dataset:
    CIC-DDoS2019
    CICIoT2023
    CIC-IoT-DIAD2024

Representation:
    Flow
    Header
    Hybrid

Architecture:
    LR
    RF
    XGB
    CNN
    BiLSTM

Primary aggregation rules documented for the revised manuscript
----------------------------------------------------------------
Table 10:
    Dataset-level classification performance.
    Each Dataset × Architecture value is the arithmetic mean of the
    corresponding native-dimensionality Flow, Header, and Hybrid
    representation-level results.

Table 11:
    Representation-level native-dimensionality results.

Table 13:
    Cross-representation stability characteristics. The table is generated
    from representation-level results and identifies the highest and lowest
    representation by the requested metric. No categorical stability label
    is assigned.

Table 14:
    Aggregate Macro-F1 across datasets separately for Flow, Header, and
    Hybrid, together with the highest/lowest representation and range.

Table 15:
    RSI across Flow/Header/Hybrid. Dataset-level RSI values are calculated
    from representation-level dataset means and then summarized across the
    three datasets as mean ± SD.

Table 16:
    Dataset-level Brier Score values. Each Dataset × Architecture value is
    the arithmetic mean of the corresponding Flow/Header/Hybrid
    representation-level Brier values.

Table 17:
    Dataset-level ECE values using the same arithmetic aggregation rule as
    Table 16.

Important
---------
This module does not contain hard-coded manuscript results. It generates
tables only from supplied run-level or representation-level result data.
Therefore, it cannot silently "repair" or invent missing values.

Run-level input should contain one row per:
    Dataset × Representation × Architecture × Seed

The generated tables can therefore be regenerated after the repository
results are populated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_DATASETS = (
    "CIC-DDoS2019",
    "CICIoT2023",
    "CIC-IoT-DIAD2024",
)

DEFAULT_REPRESENTATIONS = (
    "Flow",
    "Header",
    "Hybrid",
)

DEFAULT_ARCHITECTURES = (
    "LR",
    "RF",
    "XGB",
    "CNN",
    "BiLSTM",
)

DEFAULT_METRICS = (
    "Accuracy",
    "Macro_F1",
    "MCC",
    "Brier_Score",
    "ECE",
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
    """Find a column using case- and formatting-insensitive aliases."""
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


def standardise_result_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise common result-column variants.

    Canonical columns:
        Dataset
        Representation
        Architecture
        Seed
        Accuracy
        Macro_F1
        MCC
        Brier_Score
        ECE
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
        "Accuracy": (
            "Accuracy",
            "accuracy",
            "ACC",
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
        "Brier_Score": (
            "Brier_Score",
            "brier",
            "brier_score",
            "Brier",
        ),
        "ECE": (
            "ECE",
            "ece",
            "Expected_Calibration_Error",
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

    return dataframe.rename(
        columns=rename_map
    ).copy()


def validate_results(
    dataframe: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Validate a run-level result table without aggregating it."""
    data = standardise_result_columns(
        dataframe
    )

    for column in (
        "Dataset",
        "Representation",
        "Architecture",
        "Seed",
    ):
        if column not in data.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    available_metrics = [
        metric
        for metric in metrics
        if metric in data.columns
    ]

    if not available_metrics:
        raise ValueError(
            "None of the requested metrics are present."
        )

    for metric in available_metrics:
        data[metric] = pd.to_numeric(
            data[metric],
            errors="coerce",
        )

        if data[metric].isna().any():
            raise ValueError(
                f"Metric '{metric}' contains missing/non-numeric values."
            )

    return data


def aggregate_representation_runs(
    dataframe: pd.DataFrame,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """
    Aggregate repeated seeds to one value per
    Dataset × Representation × Architecture.

    Arithmetic mean across runs is used.

    The number of contributing runs is retained separately for auditing.
    """
    data = validate_results(
        dataframe,
        metrics=metrics,
    )

    available = [
        metric
        for metric in metrics
        if metric in data.columns
    ]

    group_columns = [
        "Dataset",
        "Representation",
        "Architecture",
    ]

    aggregations = {
        metric: ["mean", "std", "count"]
        for metric in available
    }

    result = (
        data
        .groupby(
            group_columns,
            dropna=False,
        )
        .agg(aggregations)
    )

    result.columns = [
        f"{metric}_{stat}"
        for metric, stat in result.columns
    ]

    return result.reset_index()


def metric_mean_column(
    dataframe: pd.DataFrame,
    metric: str,
) -> str:
    """Return the run-mean column name for an aggregated table."""
    column = f"{metric}_mean"

    if column not in dataframe.columns:
        raise ValueError(
            f"Aggregated metric column '{column}' not found."
        )

    return column


def make_table10_dataset_level_performance(
    representation_means: pd.DataFrame,
    metric: str = "Macro_F1",
) -> pd.DataFrame:
    """
    Generate Table 10-style dataset-level performance.

    For each Dataset × Architecture:

        mean(Flow, Header, Hybrid)

    The calculation is performed from representation-level means, not from
    already aggregated dataset-level table values.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    expected = set(
        DEFAULT_REPRESENTATIONS
    )

    rows = []

    for (dataset, architecture), group in representation_means.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        observed = set(
            group["Representation"]
        )

        missing = expected.difference(
            observed
        )

        if missing:
            raise ValueError(
                f"Table 10 cannot be generated for "
                f"{dataset} × {architecture}; missing "
                f"representations: {sorted(missing)}"
            )

        values = []

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    "Expected exactly one representation-level "
                    f"value for {dataset} × {architecture} × "
                    f"{representation}."
                )

            values.append(
                float(
                    subset.iloc[0][value_column]
                )
            )

        rows.append(
            {
                "Dataset": dataset,
                "Architecture": architecture,
                metric: float(
                    np.mean(values)
                ),
                "Representation_Count": len(
                    values
                ),
            }
        )

    return pd.DataFrame(rows)


def make_table11_representation_level_results(
    representation_means: pd.DataFrame,
    metric: str = "Macro_F1",
) -> pd.DataFrame:
    """
    Generate Table 11-style native-dimensionality representation results.

    Rows remain at Dataset × Representation × Architecture.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    result = representation_means[
        [
            "Dataset",
            "Representation",
            "Architecture",
            value_column,
        ]
    ].copy()

    result = result.rename(
        columns={
            value_column: metric
        }
    )

    return result.sort_values(
        [
            "Dataset",
            "Representation",
            "Architecture",
        ]
    ).reset_index(
        drop=True
    )


def make_table13_stability_characteristics(
    representation_means: pd.DataFrame,
    metric: str = "Macro_F1",
) -> pd.DataFrame:
    """
    Generate representation-level highest/lowest characteristics.

    The table reports:
        Flow
        Header
        Hybrid
        Highest Macro-F1 Representation
        Lowest Macro-F1 Representation
        Range

    No stability category is assigned.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    rows = []

    for (dataset, architecture), group in representation_means.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        values = {}

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Incomplete representation set for "
                    f"{dataset} × {architecture}."
                )

            values[representation] = float(
                subset.iloc[0][value_column]
            )

        highest = max(
            values,
            key=values.get,
        )

        lowest = min(
            values,
            key=values.get,
        )

        rows.append(
            {
                "Dataset": dataset,
                "Architecture": architecture,
                **values,
                "Highest Representation": highest,
                "Lowest Representation": lowest,
                "Range": (
                    max(values.values())
                    - min(values.values())
                ),
            }
        )

    return pd.DataFrame(rows)


def make_table14_aggregate_macro_f1(
    representation_means: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Table 14-style aggregate Macro-F1 by representation.

    Each representation is averaged across the three datasets for each
    architecture.

    The highest/lowest representation and the range are then reported for
    each architecture.
    """
    value_column = metric_mean_column(
        representation_means,
        "Macro_F1",
    )

    rows = []

    for architecture, group in representation_means.groupby(
        "Architecture",
        sort=True,
    ):
        representation_values = {}

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != len(
                DEFAULT_DATASETS
            ):
                raise ValueError(
                    f"Table 14 requires one value per dataset for "
                    f"{architecture} × {representation}. "
                    f"Found {len(subset)}."
                )

            representation_values[representation] = float(
                subset[value_column].mean()
            )

        highest = max(
            representation_values,
            key=representation_values.get,
        )

        lowest = min(
            representation_values,
            key=representation_values.get,
        )

        rows.append(
            {
                "Architecture": architecture,
                "Flow": representation_values["Flow"],
                "Header": representation_values["Header"],
                "Hybrid": representation_values["Hybrid"],
                "Highest Macro-F1 Representation": highest,
                "Lowest Macro-F1 Representation": lowest,
                "Range": (
                    max(
                        representation_values.values()
                    )
                    - min(
                        representation_values.values()
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def make_table15_rsi(
    representation_means: pd.DataFrame,
    metric: str = "Macro_F1",
) -> pd.DataFrame:
    """
    Generate Table 15-style RSI.

    For each Dataset × Architecture:

        RSI = sample SD across Flow/Header/Hybrid
              divided by their arithmetic mean.

    The dataset-level RSI values are then summarized by architecture as
    mean ± SD across the three datasets.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    dataset_rows = []

    for (dataset, architecture), group in representation_means.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        values = []

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Incomplete representation set for "
                    f"{dataset} × {architecture}."
                )

            values.append(
                float(
                    subset.iloc[0][value_column]
                )
            )

        mean_value = float(
            np.mean(values)
        )

        if np.isclose(
            mean_value,
            0.0,
        ):
            raise ZeroDivisionError(
                f"RSI undefined for {dataset} × {architecture}."
            )

        sd_value = float(
            np.std(
                values,
                ddof=1,
            )
        )

        dataset_rows.append(
            {
                "Dataset": dataset,
                "Architecture": architecture,
                "RSI": sd_value / mean_value,
            }
        )

    dataset_rsi = pd.DataFrame(
        dataset_rows
    )

    rows = []

    for architecture, group in dataset_rsi.groupby(
        "Architecture",
        sort=True,
    ):
        values = group["RSI"].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "Architecture": architecture,
                "RSI Mean": float(
                    np.mean(values)
                ),
                "RSI SD": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "RSI (Mean ± SD)": (
                    f"{np.mean(values):.3f} ± "
                    f"{np.std(values, ddof=1):.3f}"
                ),
            }
        )

    return pd.DataFrame(rows)


def make_table16_brier(
    representation_means: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Table 16-style dataset-level Brier Score.

    Each Dataset × Architecture value is the arithmetic mean of the
    representation-level Flow/Header/Hybrid Brier values.
    """
    return make_dataset_architecture_metric_table(
        representation_means,
        metric="Brier_Score",
        table_number=16,
    )


def make_table17_ece(
    representation_means: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Table 17-style dataset-level ECE.

    Each Dataset × Architecture value is the arithmetic mean of the
    representation-level Flow/Header/Hybrid ECE values.
    """
    return make_dataset_architecture_metric_table(
        representation_means,
        metric="ECE",
        table_number=17,
    )


def make_dataset_architecture_metric_table(
    representation_means: pd.DataFrame,
    metric: str,
    table_number: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate a representation-level metric to Dataset × Architecture.

    The aggregation is explicitly:

        mean(Flow, Header, Hybrid)

    This is used for Brier Score and ECE in Tables 16 and 17.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    rows = []

    for (dataset, architecture), group in representation_means.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        values = []

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Missing or duplicate {representation} value for "
                    f"{dataset} × {architecture}."
                )

            values.append(
                float(
                    subset.iloc[0][value_column]
                )
            )

        rows.append(
            {
                "Dataset": dataset,
                "Architecture": architecture,
                metric: float(
                    np.mean(values)
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result


def make_table15_rsi_with_components(
    representation_means: pd.DataFrame,
    metric: str = "Macro_F1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return both the compact Table 15 summary and dataset-level RSI values.

    The second dataframe is retained for auditability.
    """
    value_column = metric_mean_column(
        representation_means,
        metric,
    )

    rows = []

    for (dataset, architecture), group in representation_means.groupby(
        [
            "Dataset",
            "Architecture",
        ],
        sort=True,
    ):
        values = []

        for representation in DEFAULT_REPRESENTATIONS:
            subset = group[
                group["Representation"] == representation
            ]

            if len(subset) != 1:
                raise ValueError(
                    f"Incomplete representation set for "
                    f"{dataset} × {architecture}."
                )

            values.append(
                float(
                    subset.iloc[0][value_column]
                )
            )

        mean_value = float(
            np.mean(values)
        )

        sd_value = float(
            np.std(
                values,
                ddof=1,
            )
        )

        rows.append(
            {
                "Dataset": dataset,
                "Architecture": architecture,
                "Flow": values[0],
                "Header": values[1],
                "Hybrid": values[2],
                "Representation_Mean": mean_value,
                "Representation_SD": sd_value,
                "RSI": sd_value / mean_value,
            }
        )

    components = pd.DataFrame(
        rows
    )

    summary = (
        components
        .groupby(
            "Architecture",
            sort=True,
        )["RSI"]
        .agg(
            [
                ("RSI Mean", "mean"),
                ("RSI SD", "std"),
                ("n_datasets", "count"),
            ]
        )
        .reset_index()
    )

    summary["RSI (Mean ± SD)"] = summary.apply(
        lambda row: (
            f"{row['RSI Mean']:.3f} ± "
            f"{row['RSI SD']:.3f}"
        ),
        axis=1,
    )

    return summary, components


def round_table(
    dataframe: pd.DataFrame,
    decimals: int = 4,
) -> pd.DataFrame:
    """Round numeric columns for presentation without changing source data."""
    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = result[
        numeric_columns
    ].round(
        decimals
    )

    return result


def save_table(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
) -> Path:
    """Save one generated table as CSV."""
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / filename

    dataframe.to_csv(
        path,
        index=False,
    )

    return path


def generate_all_tables(
    results: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    round_decimals: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Generate the manuscript-oriented Tables 10, 11, 13, 14, 15, 16, and 17.

    The function first creates representation-level run means. All subsequent
    tables are derived from that common audited intermediate table.

    This common source is intentional: it prevents separate table scripts
    from silently applying different aggregation rules.
    """
    data = validate_results(
        results,
        metrics=DEFAULT_METRICS,
    )

    representation_means = aggregate_representation_runs(
        data,
        metrics=DEFAULT_METRICS,
    )

    table10 = make_table10_dataset_level_performance(
        representation_means,
        metric="Macro_F1",
    )

    table11 = make_table11_representation_level_results(
        representation_means,
        metric="Macro_F1",
    )

    table13 = make_table13_stability_characteristics(
        representation_means,
        metric="Macro_F1",
    )

    table14 = make_table14_aggregate_macro_f1(
        representation_means
    )

    table15, table15_components = (
        make_table15_rsi_with_components(
            representation_means,
            metric="Macro_F1",
        )
    )

    table16 = make_table16_brier(
        representation_means
    )

    table17 = make_table17_ece(
        representation_means
    )

    outputs = {
        "representation_level_means": representation_means,
        "table10": table10,
        "table11": table11,
        "table13": table13,
        "table14": table14,
        "table15": table15,
        "table15_components": table15_components,
        "table16": table16,
        "table17": table17,
    }

    if output_dir is not None:
        output_dir = Path(
            output_dir
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filenames = {
            "representation_level_means": (
                "representation_level_means.csv"
            ),
            "table10": "table10_dataset_level_performance.csv",
            "table11": "table11_representation_level_results.csv",
            "table13": "table13_stability_characteristics.csv",
            "table14": "table14_aggregate_macro_f1.csv",
            "table15": "table15_rsi.csv",
            "table15_components": (
                "table15_rsi_dataset_components.csv"
            ),
            "table16": "table16_brier_score.csv",
            "table17": "table17_ece.csv",
        }

        for key, filename in filenames.items():
            save_table(
                round_table(
                    outputs[key],
                    decimals=round_decimals,
                ),
                output_dir,
                filename,
            )

    return outputs


def load_and_generate(
    results_path: str | Path,
    output_dir: str | Path,
    round_decimals: int = 4,
) -> dict[str, pd.DataFrame]:
    """Load a run-level CSV and generate all supported manuscript tables."""
    results_path = Path(
        results_path
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    results = pd.read_csv(
        results_path
    )

    return generate_all_tables(
        results,
        output_dir=output_dir,
        round_decimals=round_decimals,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript-oriented tables from run-level "
            "representation-dependent IDS results."
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
            "results/tables"
        ),
        help="Output directory for generated tables.",
    )

    parser.add_argument(
        "--round",
        type=int,
        default=4,
        dest="round_decimals",
        help="Number of decimal places in exported CSV values.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    outputs = load_and_generate(
        results_path=args.results,
        output_dir=args.output_dir,
        round_decimals=args.round_decimals,
    )

    print(
        f"Generated {len(outputs)} table/intermediate outputs "
        f"in {args.output_dir}"
    )

    for key, dataframe in outputs.items():
        print(
            f"  {key}: {len(dataframe)} rows"
        )


def _self_check() -> None:
    """
    Internal validation using synthetic values only.

    The synthetic values are deliberately simple and are not manuscript
    results.
    """
    rows = []

    for dataset_index, dataset in enumerate(
        DEFAULT_DATASETS
    ):
        for representation_index, representation in enumerate(
            DEFAULT_REPRESENTATIONS
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
                        0.80
                        + 0.01 * dataset_index
                        + 0.005 * representation_index
                        + 0.003 * architecture_index
                        + 0.0001 * (seed - 2026)
                    )

                    rows.append(
                        {
                            "Dataset": dataset,
                            "Representation": representation,
                            "Architecture": architecture,
                            "Seed": seed,
                            "Accuracy": base + 0.02,
                            "Macro_F1": base,
                            "MCC": base - 0.10,
                            "Brier_Score": 0.10 - 0.001 * representation_index,
                            "ECE": 0.05 + 0.001 * representation_index,
                        }
                    )

    data = pd.DataFrame(
        rows
    )

    outputs = generate_all_tables(
        data
    )

    assert len(
        outputs["representation_level_means"]
    ) == 45

    assert len(
        outputs["table10"]
    ) == 15

    assert len(
        outputs["table11"]
    ) == 45

    assert len(
        outputs["table13"]
    ) == 15

    assert len(
        outputs["table14"]
    ) == 5

    assert len(
        outputs["table15"]
    ) == 5

    assert len(
        outputs["table15_components"]
    ) == 15

    assert len(
        outputs["table16"]
    ) == 15

    assert len(
        outputs["table17"]
    ) == 15

    # Table 10 must be the arithmetic mean of the three representations.
    first = outputs["table10"].iloc[0]

    source = outputs[
        "representation_level_means"
    ]

    subset = source[
        (
            source["Dataset"]
            == first["Dataset"]
        )
        & (
            source["Architecture"]
            == first["Architecture"]
        )
    ]

    expected_macro_f1 = subset[
        "Macro_F1_mean"
    ].mean()

    assert np.isclose(
        first["Macro_F1"],
        expected_macro_f1,
    )

    # Table 16 must average Flow/Header/Hybrid Brier values.
    first_brier = outputs[
        "table16"
    ].iloc[0]

    subset = source[
        (
            source["Dataset"]
            == first_brier["Dataset"]
        )
        & (
            source["Architecture"]
            == first_brier["Architecture"]
        )
    ]

    expected_brier = subset[
        "Brier_Score_mean"
    ].mean()

    assert np.isclose(
        first_brier["Brier_Score"],
        expected_brier,
    )

    print(
        "generate_tables.py self-check passed."
    )


if __name__ == "__main__":
    main()
