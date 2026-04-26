"""Example: Federated AutoML with PEA-Bag (bagging) ensembling.

The branches are still trained on disjoint partitions (any of the
partitioners listed in ``partitioning_method`` works), but their
predictions are combined by averaging probabilities (``soft``) or
majority-voting labels (``hard``) instead of going through a meta-
learner.

Tunable ``ensembling_params`` for ``'bagging'``:

* ``voting``: ``'soft'`` (default; mean of class probabilities) or
  ``'hard'`` (majority vote on argmax labels).
"""
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_bagging_federated_automl_example(timeout: int = 10,
                                          voting: str = 'soft'):
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            'partitioning_method': 'sequential',
            'partitioning_params': {'random_state': 42},
            'ensembling_method': 'bagging',
            'ensembling_params': {'voting': voting},
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
        num_samples=1800,
        task='classification',
        max_ts_len=50,
        binary=True,
        test_size=0.5,
        multivariate=False,
    ).generate_data()
    dataset_dict = dict(train_data=train_data, test_data=test_data)
    return ApiTemplate(
        api_config=api_config,
        metric_list=('f1', 'accuracy'),
    ).eval(dataset=dataset_dict, finetune=False)


if __name__ == '__main__':
    result = run_bagging_federated_automl_example(timeout=2, voting='soft')
    print(result)
