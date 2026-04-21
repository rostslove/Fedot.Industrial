"""Example: Federated AutoML with TS3 (Model-Based Difficulty) partitioning.

Mirrors :mod:`federated_automl_example` (synthetic multi-series TS,
binary classification, ``data_type='time_series'``). TS3 fits a
lightweight teacher on the full training set -- a shallow
``DecisionTreeClassifier`` for classification or ``Ridge`` for
regression -- then sorts samples by the teacher's error and splits
them into ``n_splits`` contiguous difficulty bands. Unlike
:class:`DifficultyPartitioner` this variant does NOT use
cross-validation: standard K-Fold reshuffles time and is invalid for
a TS curriculum.

Tunable ``partitioning_params`` for ``'ts_difficulty'``:

* ``teacher``: pre-configured estimator implementing ``fit`` /
  ``predict`` (and ``predict_proba`` for classification).
* ``task_type``: ``'classification'`` or ``'regression'`` (auto if
  omitted).
* ``random_state``: seed for the default teacher.
* ``scale_features``: z-score features before fitting the teacher.
"""
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_ts_difficulty_federated_ts_example(timeout: int = 10,
                                           num_samples: int = 1800,
                                           max_ts_len: int = 50):
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            'partitioning_method': 'ts_difficulty',
            'partitioning_params': {
                'task_type': 'classification',
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

    train_data, test_data = TimeSeriesDatasetsGenerator(
        num_samples=num_samples,
        task='classification',
        max_ts_len=max_ts_len,
        binary=True,
        test_size=0.5,
        multivariate=False,
    ).generate_data()

    dataset_dict = dict(train_data=train_data, test_data=test_data)
    result_dict = ApiTemplate(
        api_config=api_config,
        metric_list=('f1', 'accuracy'),
    ).eval(dataset=dataset_dict, finetune=False)

    return result_dict


if __name__ == '__main__':
    result = run_ts_difficulty_federated_ts_example(timeout=2)
    print(result)
