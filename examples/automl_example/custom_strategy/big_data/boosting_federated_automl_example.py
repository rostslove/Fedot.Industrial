"""Example: Federated AutoML with PEA-Boost (boosting) ensembling on the
Adult Census Income dataset (same loader as
``clustered_federated_automl_example.py``).

Each round resamples the FULL training set with replacement using
weights derived from the current ensemble's errors, then fits a fresh
AutoML pipeline on the bootstrap. Classification uses AdaBoost-SAMME;
regression would use gradient boosting on residuals.

The partitioning method is ignored on purpose -- boosting rewrites the
training distribution between rounds, which would make a fixed split
counter-productive. ``n_splits`` (a.k.a. the number of FEDOT workers)
is reused as the number of rounds.

Tunable ``ensembling_params`` for ``'boosting'``:

* ``learning_rate``: shrinkage applied to each round's contribution
  (regression only; classification uses SAMME's ``alpha``).
* ``early_stop_eps``: classification rounds with err >= 1 - 1/K - eps
  (worse than random) are dropped.
* ``random_state``: seed for the weighted bootstrap RNG.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)

DATASET_PATH = r'examples/automl_example/custom_strategy/big_data/data/adult.csv'


def _load_adult(path: str, test_size: float = 0.3, random_state: int = 42):
    """Load and preprocess Adult Census Income from CSV.

    Mirrors the loader in ``clustered_federated_automl_example.py`` so
    every PEA strategy in this folder runs on the same data.
    """
    df = pd.read_csv(path)
    df = df.replace('?', np.nan).dropna()

    target_col = 'income'
    df[target_col] = df[target_col].str.strip().map({'<=50K': 0, '>50K': 1})
    df = df.dropna(subset=[target_col])
    df = df.head(3000)
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])

    cat_cols = ['workclass', 'education', 'marital.status', 'occupation',
                'relationship', 'race', 'sex', 'native.country']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y)
    return (X_train, y_train), (X_test, y_test)


def run_boosting_federated_automl_example(timeout: int = 20,
                                           learning_rate: float = 0.1,
                                           random_state: int = 42):
    """Run federated AutoML on Adult Income with PEA-Boost.

    Args:
        timeout: total AutoML budget in minutes. The federated solver
            divides this by ``FEDOT_WORKER_TIMEOUT_PARTITION=5`` and
            clamps to >=0.5 min/round, so practical minimum is ~10
            (giving each round ~2 min). At ``timeout=2`` each AutoML
            round is starved before its initial assumption finishes
            and PEA-Boost falls back to a single-branch ensemble.
        learning_rate: regression-only shrinkage; ignored for the
            Adult classification task but kept on the signature for
            symmetry with the regression variant.
        random_state: seed for the weighted bootstrap RNG.
    """
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'table',
            'problem': 'classification',
            'n_jobs': 1,
            'partitioning_method': 'sequential',
            'ensembling_method': 'boosting',
            'ensembling_params': {
                'learning_rate': learning_rate,
                'random_state': random_state,
            },
        },
    }

    learning_config = {
        'learning_strategy': 'from_scratch',
        'learning_strategy_params': {**DEFAULT_AUTOML_LEARNING_CONFIG,
                                     'timeout': timeout, 'n_jobs': 1},
        'optimisation_loss': {'quality_loss': 'f1'},
    }

    api_config = {
        'industrial_config': industrial_config,
        'automl_config': DEFAULT_CLF_AUTOML_CONFIG,
        'learning_config': learning_config,
        'compute_config': DEFAULT_COMPUTE_CONFIG,
    }

    train_data, test_data = _load_adult(DATASET_PATH, test_size=0.3)
    dataset_dict = dict(train_data=train_data, test_data=test_data)
    return ApiTemplate(
        api_config=api_config,
        metric_list=('f1', 'accuracy'),
    ).eval(dataset=dataset_dict, finetune=False)


if __name__ == '__main__':
    result = run_boosting_federated_automl_example(
        timeout=20, learning_rate=0.1, random_state=42)
    print(result)
