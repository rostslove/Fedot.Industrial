"""Example: Federated AutoML with PEA-Boost (boosting) ensembling.

Each round resamples the FULL training set with replacement using
weights derived from the current ensemble's errors, then fits a fresh
AutoML pipeline on the bootstrap. Classification uses AdaBoost-SAMME;
regression uses gradient boosting on residuals.

The partitioning method is ignored on purpose -- boosting rewrites the
training distribution between rounds, which would make a fixed split
counter-productive. ``n_splits`` is reused as the number of rounds.

Tunable ``ensembling_params`` for ``'boosting'``:

* ``learning_rate``: shrinkage applied to each round's contribution
  (regression only; classification uses SAMME's ``alpha``).
* ``early_stop_eps``: classification rounds with err >= 1 - 1/K - eps
  (worse than random) are dropped.
* ``random_state``: seed for the weighted bootstrap RNG.
"""
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_boosting_federated_automl_example(timeout: int = 10,
                                           learning_rate: float = 0.1,
                                           random_state: int = 42):
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            # boosting ignores partitioning -- but the pipeline still
            # walks through FeatureSpacePartitioner.create, so keep a
            # cheap default.
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
    result = run_boosting_federated_automl_example(timeout=2)
    print(result)
