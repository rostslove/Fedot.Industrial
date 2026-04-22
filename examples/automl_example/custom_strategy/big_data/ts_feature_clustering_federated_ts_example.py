"""Example: Federated AutoML with TS2 (Feature-Based Clustering) on M4.

M4 is reshaped into a multi-class TS-classification task
(label = frequency group: Daily / Weekly / Monthly / Quarterly / Yearly)
by :func:`load_m4_classification`. TS2 treats each row of ``features``
as one time series and extracts a compact descriptor vector (mean / std
/ linear trend / lag-1 autocorrelation / dominant FFT amplitude /
Shannon entropy). KMeans then groups series with similar patterns
(trend / seasonality / volatility) into the same partition -- which is
particularly expressive on M4 because different frequency groups show
distinctly different descriptor profiles.

``data_type='time_series'`` is intentional: the head's collapse issue
was fixed inside :class:`RAFEnsembler` (row-aligned stacking), so the
heavy Industrial TS pool in the branches now works as designed.

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

try:
    from examples.automl_example.custom_strategy.big_data.m4_classification_utils import load_m4_classification
except ModuleNotFoundError:
    from m4_classification_utils import load_m4_classification


def run_ts_feature_clustering_federated_ts_example(
        timeout: int = 10,
        n_per_group: int = 400,
        window_length: int = 50,
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
            'n_jobs': 1,
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

    train_data, test_data = load_m4_classification(
        n_per_group=n_per_group,
        window_length=window_length,
        test_size=0.3,
        random_state=42,
    )

    dataset_dict = dict(train_data=train_data, test_data=test_data)
    result_dict = ApiTemplate(
        api_config=api_config,
        metric_list=('f1', 'accuracy'),
    ).eval(dataset=dataset_dict, finetune=False)

    return result_dict


if __name__ == '__main__':
    # timeout=2 min is too small -- see note in
    # temporal_split_federated_ts_example.py.
    result = run_ts_feature_clustering_federated_ts_example(timeout=10)
    print(result)
