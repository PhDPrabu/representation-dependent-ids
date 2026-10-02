"""
Classical machine-learning model definitions for the reproducibility
repository.

This module centralizes the classical model configurations documented for
the representation-dependent IDS experiments:

    1. Logistic Regression
    2. Random Forest
    3. XGBoost

The available experimental source (`04_run_pilot.py`) provides the concrete
model/pipeline definitions used here. That source is explicitly identified
as a pilot and therefore this module does NOT claim that the pilot file
itself generated the final manuscript's complete 10-seed result set.

The purpose of this module is to make the documented model configuration
reusable by the primary reproducibility runner.

Model-specific preprocessing is intentionally kept with each model:
    - Logistic Regression: median imputation + Min-Max scaling
    - Random Forest: median imputation
    - XGBoost: median imputation

All data-dependent preprocessing must be fitted on the training partition
only. Validation and test partitions must be transformed using parameters
learned from training data.
"""

from __future__ import annotations

from typing import Dict

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


MODEL_NAMES = (
    "LR",
    "RF",
    "XGB",
)


def build_logistic_regression(random_state: int) -> Pipeline:
    """
    Build the documented Logistic Regression pipeline.

    Configuration:
        median imputation
        Min-Max scaling
        L2 penalty
        lbfgs solver
        max_iter = 1000
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                MinMaxScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest(random_state: int) -> Pipeline:
    """
    Build the documented Random Forest pipeline.

    Configuration:
        median imputation
        200 trees
        max_depth = 20
        criterion = gini
        n_jobs = -1
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    criterion="gini",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost(random_state: int) -> Pipeline:
    """
    Build the documented XGBoost multiclass pipeline.

    Configuration:
        median imputation
        n_estimators = 300
        learning_rate = 0.05
        max_depth = 10
        subsample = 0.8
        objective = multi:softprob
        eval_metric = mlogloss
        n_jobs = -1
    """
    if XGBClassifier is None:
        raise ImportError(
            "XGBoost is required for the XGB model. "
            "Install it with `pip install xgboost`."
        )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=10,
                    subsample=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_model(
    model_name: str,
    random_state: int,
):
    """
    Construct one of the supported classical models.

    Parameters
    ----------
    model_name:
        One of ``LR``, ``RF``, or ``XGB``.
    random_state:
        Run-specific random seed.

    Returns
    -------
    sklearn Pipeline
        Configured model pipeline.
    """
    normalized_name = model_name.strip().upper()

    builders = {
        "LR": build_logistic_regression,
        "RF": build_random_forest,
        "XGB": build_xgboost,
    }

    if normalized_name not in builders:
        raise ValueError(
            f"Unsupported classical model '{model_name}'. "
            f"Supported models: {', '.join(MODEL_NAMES)}."
        )

    return builders[normalized_name](random_state=random_state)


def model_configuration() -> Dict[str, dict]:
    """
    Return a machine-readable summary of the classical configurations.

    This is intended for logging and reproducibility metadata. The actual
    sklearn objects are constructed by the builder functions above.
    """
    return {
        "LR": {
            "name": "Logistic Regression",
            "imputation": "median",
            "scaling": "MinMaxScaler",
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "RF": {
            "name": "Random Forest",
            "imputation": "median",
            "scaling": "none",
            "n_estimators": 200,
            "max_depth": 20,
            "criterion": "gini",
            "n_jobs": -1,
        },
        "XGB": {
            "name": "XGBoost",
            "imputation": "median",
            "scaling": "none",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 10,
            "subsample": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        },
    }


def model_display_name(model_name: str) -> str:
    """Return the manuscript-facing display name for a model key."""
    names = {
        "LR": "Logistic Regression",
        "RF": "Random Forest",
        "XGB": "XGBoost",
    }

    normalized_name = model_name.strip().upper()

    if normalized_name not in names:
        raise ValueError(
            f"Unsupported classical model '{model_name}'."
        )

    return names[normalized_name]


if __name__ == "__main__":
    print("Classical model configuration self-check:")

    for name in MODEL_NAMES:
        model = build_model(name, random_state=2026)
        print(f"- {name}: {model_display_name(name)}")
        print(f"  Pipeline steps: {list(model.named_steps.keys())}")

    print("\nConfiguration summary:")
    for name, configuration in model_configuration().items():
        print(f"\n{name}")
        for key, value in configuration.items():
            print(f"  {key}: {value}")
