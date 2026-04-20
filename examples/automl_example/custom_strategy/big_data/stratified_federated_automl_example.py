"""Example: Federated AutoML with T3 (Stratified Sampling) partitioning
on the Adult Census Income dataset.

Strategy T3 splits the training set with
:class:`sklearn.model_selection.StratifiedKFold` (classification) or its
quantile-binned equivalent (regression).  Every partition therefore
preserves the original target distribution -- the natural baseline for
imbalanced datasets such as Adult Income (~24% positives).

Tunable ``partitioning_params`` for ``'stratified'``:

* ``task_type``: ``'classification'`` / ``'regression'`` (auto-inferred).
* ``regression_bins``: number of quantile bins used to stratify
  regression targets (default 10). Ignored for classification.
* ``shuffle``: whether to shuffle before splitting (default True).
* ``random_state``: seed used when ``shuffle`` is True.
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


def run_stratified_federated_example(timeout: int = 10,
                                     shuffle: bool = True):
    """Run federated AutoML with the T3 stratified partitioner."""
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'table',
            'problem': 'classification',
            'n_jobs': 1,
            'partitioning_method': 'stratified',
            'partitioning_params': {
                'task_type': 'classification',
                'shuffle': shuffle,
                'random_state': 42,
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
    result = run_stratified_federated_example(timeout=20, shuffle=True)
    print(result)
