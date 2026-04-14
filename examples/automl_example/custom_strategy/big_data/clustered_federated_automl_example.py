"""Example: Federated AutoML with cluster-based dataset partitioning
on the Adult Census Income dataset (ARFF format).

The partitioning method is controlled via ``strategy_params``:

* ``'partitioning_method'``:  ``'sequential'`` | ``'kmeans'`` | ``'dbscan'``
* ``'partitioning_params'``:  extra kwargs forwarded to the partitioner.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_COMPUTE_CONFIG,
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
)

DATASET_PATH = r'examples\automl_example\custom_strategy\big_data\data\adult.csv'


def _load_adult(path: str, test_size: float = 0.3, random_state: int = 42):
    """Load and preprocess the Adult Census Income dataset from CSV.

    Returns ``(train_data, test_data)`` tuples of ``(X, y)``.
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
    num_cols = [c for c in X.columns if c not in cat_cols]

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y)

    return (X_train, y_train), (X_test, y_test)


def run_clustered_federated_example(timeout: int = 10,
                                    partitioning_method: str = 'kmeans'):
    """Run federated AutoML with a configurable partitioning strategy.

    Args:
        timeout: total AutoML budget in minutes.
        partitioning_method: one of ``'sequential'``, ``'kmeans'``, ``'dbscan'``.
    """
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'table',
            'problem': 'classification',
            'partitioning_method': partitioning_method,
            'partitioning_params': {
                'random_state': 42,
                'scale_features': True,
            },
        },
    }

    learning_config = {
        'learning_strategy': 'from_scratch',
        'learning_strategy_params': {**DEFAULT_AUTOML_LEARNING_CONFIG, 'timeout': timeout},
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
    result = run_clustered_federated_example(timeout=2, partitioning_method='sequential')
    print(result)
