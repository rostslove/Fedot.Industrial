"""Example: Federated AutoML with PEA-Boost (boosting) ensembling on M4
TS-classification (same loader as
``ts_feature_clustering_federated_ts_example.py``).

M4 is reshaped into a multi-class TS-classification task by
:func:`load_m4_classification` (label = frequency group: Daily /
Weekly / Monthly / Quarterly / Yearly). Each PEA-Boost round bootstrap-
resamples the FULL training set with weights derived from the current
ensemble's errors and trains a fresh AutoML pipeline on the sample.
Classification uses AdaBoost-SAMME on argmax labels.

The partitioning method is ignored on purpose -- boosting rewrites the
training distribution between rounds, which would make a fixed split
counter-productive. ``n_splits`` (a.k.a. the number of FEDOT workers)
is reused as the number of boosting rounds.

``data_type='time_series'`` keeps the heavy Industrial TS pool
available inside each round's AutoML composer (InceptionTime,
quantile_extractor, minirocket_extractor, ...).

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

try:
    from examples.automl_example.custom_strategy.big_data.m4_classification_utils import load_m4_classification
except ModuleNotFoundError:
    from m4_classification_utils import load_m4_classification


def run_boosting_federated_ts_example(
        timeout: int = 10,
        n_per_group: int = 400,
        window_length: int = 50,
        learning_rate: float = 0.1,
        random_state: int = 42):
    """Run federated AutoML on M4-as-classification with PEA-Boost.

    Args:
        timeout: total AutoML budget in minutes. The federated solver
            divides this by ``FEDOT_WORKER_TIMEOUT_PARTITION=5`` and
            clamps to >=0.5 min/round, so practical minimum is ~10
            (giving each round ~2 min). At ``timeout=2`` each AutoML
            round is starved before its initial assumption finishes
            and PEA-Boost falls back to a single-branch ensemble.
        n_per_group: number of M4 series sampled per frequency group.
        window_length: per-series truncation length.
        learning_rate: regression-only shrinkage; ignored for the M4
            classification task but kept on the signature for
            symmetry with the regression variant.
        random_state: seed for the weighted bootstrap RNG.
    """
    industrial_config = {
        'problem': 'classification',
        'learning_strategy': 'federated_automl',
        'strategy': 'federated_automl',
        'strategy_params': {
            'timeout': timeout,
            'data_type': 'time_series',
            'problem': 'classification',
            'n_jobs': 1,
            # Boosting ignores the partitioner, but the RAF pipeline
            # still constructs one -- pick the cheapest option.
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
    result = run_boosting_federated_ts_example(
        timeout=10, learning_rate=0.1, random_state=42)
    print(result)
