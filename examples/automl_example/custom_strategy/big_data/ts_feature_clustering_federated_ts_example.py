"""Example: Federated AutoML with TS2 (Feature-Based Clustering) partitioning.

Mirrors :mod:`federated_automl_example` (synthetic multi-series TS,
binary classification, ``data_type='time_series'``). TS2 treats each
row of ``features`` as one time series and extracts a compact
descriptor vector (mean / std / linear trend / lag-1 autocorrelation /
dominant FFT amplitude / Shannon entropy). KMeans then groups series
with similar patterns (trend / seasonality / volatility) into the
same partition, so every RAF worker specialises on a distinct
"regime".

Tunable ``partitioning_params`` for ``'ts_feature_clustering'``:

* ``features_to_use``: subset of the default descriptors, or a
  callable ``series -> 1-D np.ndarray`` for a fully custom extractor.
* ``random_state``: KMeans seed.
* ``scale_features``: z-score descriptors before clustering (default
  True).
"""
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_ts_feature_clustering_federated_ts_example(
        timeout: int = 10,
        num_samples: int = 1800,
        max_ts_len: int = 50,
        features_to_use=('mean', 'std', 'trend',
                         'lag1_ac', 'dom_fft', 'entropy')):
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            'partitioning_method': 'ts_feature_clustering',
            'partitioning_params': {
                'features_to_use': features_to_use,
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
    result = run_ts_feature_clustering_federated_ts_example(timeout=2)
    print(result)
