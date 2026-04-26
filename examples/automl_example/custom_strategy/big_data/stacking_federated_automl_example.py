"""Example: Federated AutoML with PEA-Stack (stacking) ensembling on the
Adult Census Income dataset (same loader as
``clustered_federated_automl_example.py``).

This is the original RAF behaviour, now exposed through the same
``ensembling_method`` plug-in surface as bagging and boosting. Branch
predictions on the FULL training set form a row-aligned stacked
feature matrix that feeds a meta-learner (the "head"). The head can
be swapped via ``ensembling_params['head']``.

Tunable ``ensembling_params`` for ``'stacking'``:

* ``head``: FEDOT operation name for the meta-learner (e.g.
  ``'xgboost'`` (problem default for classification), ``'logit'``,
  ``'rf'``, ``'treg'`` (default for regression)).
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

def run_stacking_federated_automl_example(timeout: int = 20,
                                           head: str = 'logit',
                                           partitioning_method: str = 'sequential'):
    """Run federated AutoML on Adult Income with PEA-Stack.

    Args:
        timeout: total AutoML budget in minutes. The federated solver
            divides this by ``FEDOT_WORKER_TIMEOUT_PARTITION=5`` and
            clamps to >=0.5 min/branch; use ``timeout >= 10`` so each
            branch's AutoML composer has at least ~2 min.
        head: FEDOT operation name for the meta-learner. ``'logit'``
            (logistic regression) is a strong, interpretable default
            on the stacked probability matrix; ``'xgboost'`` is the
            problem-default and usually squeezes out a bit more
            accuracy on Adult; ``'rf'`` works well too.
        partitioning_method: any partitioner registered with
            :class:`FeatureSpacePartitioner`.
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
            'partitioning_method': partitioning_method,
            'partitioning_params': {
                'random_state': 42,
                'scale_features': True,
            },
            'ensembling_method': 'stacking',
            'ensembling_params': {'head': head},
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
    result = run_stacking_federated_automl_example(
        timeout=20, head='logit', partitioning_method='sequential')
    print(result)
