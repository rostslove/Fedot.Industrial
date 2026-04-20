"""Example: Federated AutoML with T2 (Uncertainty / Difficulty) partitioning
on the Adult Census Income dataset.

Strategy T2 fits a lightweight "easy" model via cross-validation, derives
a per-sample difficulty score from its error / probability output, and
splits the training set into ``n_splits`` contiguous difficulty buckets.
Each RAF worker therefore specialises on a different difficulty band.

Tunable ``partitioning_params`` for ``'difficulty'``:

* ``cv``: number of CV folds for the weak model (default 3).
* ``order``: ``'hard_first'`` (default) puts the hardest samples in the
  first partition (which RAF uses as ``main_target``), ``'easy_first'``
  does the opposite.
* ``task_type``: ``'classification'`` / ``'regression'`` (auto-inferred).
* ``scale_features``: z-score features before the weak model
  (default True).
* ``random_state``: seed for the weak model and any resampling.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_COMPUTE_CONFIG,
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
)

DATASET_PATH = r'examples\automl_example\custom_strategy\big_data\data\adult.csv'


def _load_adult(path: str, test_size: float = 0.3, random_state: int = 42):
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


def run_difficulty_federated_example(timeout: int = 10,
                                     order: str = 'hard_first',
                                     cv: int = 3):
    """Run federated AutoML with the T2 difficulty partitioner."""
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'table',
            'problem': 'classification',
            'n_jobs': 1,
            'partitioning_method': 'difficulty',
            'partitioning_params': {
                'task_type': 'classification',
                'cv': cv,
                'order': order,
                'random_state': 42,
                'scale_features': True,
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

    result_dict = ApiTemplate(
        api_config=api_config,
        metric_list=('f1', 'accuracy'),
    ).eval(dataset=dataset_dict, finetune=False)

    return result_dict


if __name__ == '__main__':
    result = run_difficulty_federated_example(
        timeout=20, order='hard_first', cv=3)
    print(result)
