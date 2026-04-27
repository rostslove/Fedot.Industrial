"""Example: Federated AutoML with PEA-Stack (stacking) ensembling on M4
TS-classification (same loader as
``ts_feature_clustering_federated_ts_example.py``).

M4 is reshaped into a multi-class TS-classification task by
:func:`load_m4_classification` (label = frequency group: Daily /
Weekly / Monthly / Quarterly / Yearly). Each branch is fitted on its
own partition and predicts on the FULL training set; the row-aligned
stacked feature matrix then feeds a meta-learner (the "head"). The
head can be swapped via ``ensembling_params['head']``.

This is the original RAF behaviour, exposed through the same
``ensembling_method`` plug-in surface as bagging and boosting.
``data_type='time_series'`` keeps the heavy Industrial TS pool
available inside each branch.

Tunable ``ensembling_params`` for ``'stacking'``:

* ``head``: FEDOT operation name for the meta-learner (e.g.
  ``'xgboost'`` (problem default for classification), ``'logit'``,
  ``'rf'``, ``'treg'`` (default for regression)).
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


def run_stacking_federated_ts_example(
        timeout: int = 10,
        n_per_group: int = 400,
        window_length: int = 50,
        head: str = 'logit',
        partitioning_method: str = 'ts_feature_clustering'):
    """Run federated AutoML on M4-as-classification with PEA-Stack.

    Args:
        timeout: total AutoML budget in minutes. The federated solver
            divides this by ``FEDOT_WORKER_TIMEOUT_PARTITION=5`` and
            clamps to >=0.5 min/branch; use ``timeout >= 10`` so the
            heavy TS composer has at least ~2 min per branch.
        n_per_group: number of M4 series sampled per frequency group.
        window_length: per-series truncation length.
        head: FEDOT operation name for the meta-learner. ``'logit'``
            (logistic regression) is a strong, interpretable default
            on the stacked probability matrix; ``'xgboost'`` is the
            problem default and usually squeezes out a bit more
            accuracy; ``'rf'`` works well too.
        partitioning_method: any partitioner registered with
            :class:`FeatureSpacePartitioner` (TS partitioners
            ``temporal``, ``ts_feature_clustering``, ``ts_difficulty``
            are the natural choices on M4).
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
            'partitioning_method': partitioning_method,
            'partitioning_params': {
                'random_state': 42,
                'scale_features': True,
            },
            'ensembling_method': 'stacking',
            'ensembling_params': {'head': head},
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
    result = run_stacking_federated_ts_example(
        timeout=10, head='logit',
        partitioning_method='ts_feature_clustering')
    print(result)
