"""Example: Federated AutoML with TS1 (Temporal Split) partitioning.

Synthetic multi-series TS from :class:`TimeSeriesDatasetsGenerator`,
binary classification with ``data_type='time_series'`` so that each
RAF worker is free to choose from the heavy Industrial TS pool
(InceptionTime, quantile_extractor, minirocket_extractor, ...).

The stacking asymmetry that used to collapse the head is fixed in
:class:`RAFEnsembler` itself: branches now predict on the FULL
training set to build a row-aligned stacked feature matrix for the
head, instead of the old ``MultiModalData`` + ``join_branches`` path
where the head trained on four branches' predictions on four DIFFERENT
partitions (so only column 0 correlated with the target and the head
degenerated to "copy branch 0"). See the fit-time docstring of
:meth:`RAFEnsembler.fit` for details.

Tunable ``partitioning_params`` for ``'temporal'``:

* ``overlap``: fraction of chunk length reused from the preceding
  block (default 0 -- disjoint).
* ``min_chunk_size``: floor on per-chunk sample count; ``n_splits``
  is reduced automatically if the dataset is too small.
"""
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import (
    DEFAULT_AUTOML_LEARNING_CONFIG,
    DEFAULT_CLF_AUTOML_CONFIG,
    DEFAULT_COMPUTE_CONFIG,
)
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def run_temporal_split_federated_ts_example(timeout: int = 10,
                                            num_samples: int = 1800,
                                            max_ts_len: int = 50,
                                            overlap: float = 0.0):
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            'n_jobs': 1,
            'partitioning_method': 'temporal',
            'partitioning_params': {
                'overlap': overlap,
                'min_chunk_size': 50,
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
    # timeout=2 min is too small: _federated_strategy divides it by
    # FEDOT_WORKER_TIMEOUT_PARTITION (=5) and clamps to >=0.5 min per
    # worker -- FEDOT skips composing/tuning and returns only the
    # initial assumption (CNN overfits in 30 s). Use >= 10 min so
    # every worker has >= 2 min.
    result = run_temporal_split_federated_ts_example(timeout=10)
    print(result)
