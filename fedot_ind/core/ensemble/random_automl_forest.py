from copy import deepcopy
from typing import List

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.partitioning import FeatureSpacePartitioner
from fedot_ind.core.repository.constanst_repository import FEDOT_ATOMIZE_OPERATION, FEDOT_HEAD_ENSEMBLE, FEDOT_TASK
from fedot_ind.core.repository.model_repository import default_industrial_availiable_operation


_DATA_TYPE_TO_ENUM = {
    'image': DataTypesEnum.image,
    'table': DataTypesEnum.table,
    'ts': DataTypesEnum.ts,
    'text': DataTypesEnum.text,
}

_DATA_TYPE_TO_SOURCE_PREFIX = {
    DataTypesEnum.image: 'data_source_img',
    DataTypesEnum.table: 'data_source_table',
    DataTypesEnum.ts: 'data_source_ts',
    DataTypesEnum.text: 'data_source_text',
}


class RAFEnsembler:
    """Ensemble of independently-optimised AutoML pipelines (Random AutoML Forest).

    The training dataset is partitioned into ``n_splits`` disjoint subsets.
    Each subset is handled by a separate FEDOT AutoML worker that discovers
    its own best pipeline.  A head model (e.g. XGBoost) combines the
    outputs of all workers into the final prediction.

    Args:
        composing_params: dict with parameters for ensemble.  Recognised
            keys include ``problem``, ``timeout``, ``available_operations``,
            and ``partitioning_method`` / ``partitioning_params``.
        n_splits: number of data partitions (workers).
        batch_size: approximate samples per partition when ``n_splits``
            is not specified explicitly.

    Partitioning methods (``composing_params['partitioning_method']``):

    * ``'sequential'`` -- equal-sized sequential chunks (**default**,
      original behaviour).
    * ``'kmeans'`` -- K-Means clustering in feature space.
    * ``'dbscan'`` -- DBSCAN density-based clustering.
    * ``'difficulty'`` -- uncertainty / difficulty sampling using a
      weak model's error matrix.
    * ``'stratified'`` -- stratified splits preserving the target
      distribution.
    * ``'temporal'`` -- TS1: contiguous blocks along the time axis.
    * ``'ts_feature_clustering'`` -- TS2: cluster series by their
      statistical / spectral descriptors.
    * ``'ts_difficulty'`` -- TS3: bucket samples by residuals from a
      teacher model fit on the whole training set (no CV).

    Additional clustering parameters can be passed via
    ``composing_params['partitioning_params']`` dict.
    """

    def __init__(self,
                 composing_params,
                 n_splits: int = None,
                 batch_size: int = 1000):

        self.current_pipeline = None
        self._branches: List[Pipeline] = []
        self._head: Pipeline = None
        self.problem = composing_params['problem']
        self.task = FEDOT_TASK[composing_params['problem']]
        self.atomized_automl = FEDOT_ATOMIZE_OPERATION[composing_params['problem']]
        self.head = FEDOT_HEAD_ENSEMBLE[composing_params['problem']]

        self.atomized_automl_params = deepcopy(composing_params)

        raw_data_type = self.atomized_automl_params.pop('data_type', 'image')
        if isinstance(raw_data_type, DataTypesEnum):
            self.data_type = raw_data_type
        else:
            self.data_type = _DATA_TYPE_TO_ENUM.get(str(raw_data_type), DataTypesEnum.image)
        self.source_prefix = _DATA_TYPE_TO_SOURCE_PREFIX[self.data_type]

        if 'available_operations' not in self.atomized_automl_params:
            self.atomized_automl_params['available_operations'] = \
                default_industrial_availiable_operation(self._resolve_operations_problem())

        self.partitioning_method = self.atomized_automl_params.pop(
            'partitioning_method', 'sequential')
        self.partitioning_params = self.atomized_automl_params.pop(
            'partitioning_params', {})

        self.n_splits = n_splits
        self.batch_size = batch_size

    def _resolve_operations_problem(self):
        """Pick operations pool based on both problem and data_type.

        For tabular classification/regression we switch to the sklearn-only
        pool (``classification_tabular``/``regression_tabular``) so the AutoML
        composer doesn't consider TS-specific extractors on plain tables.
        """
        if self.data_type == DataTypesEnum.table and self.problem in ('classification', 'regression'):
            return f'{self.problem}_tabular'
        return self.problem

    def fit(self, train_data):
        if self.n_splits is None:
            self.n_splits = round(train_data.features.shape[0] / self.batch_size)

        partitioner = FeatureSpacePartitioner.create(
            method=self.partitioning_method,
            n_splits=self.n_splits,
            params=self.partitioning_params)

        features_splits, target_splits = partitioner.partition(
            train_data.features, train_data.target)
        self.n_splits = len(features_splits)

        # 1. fit one atomized-AutoML per partition (heavy TS models live here).
        self._branches = self._fit_branches(features_splits, target_splits)

        # 2. every branch predicts on the FULL training set -> row-aligned
        #    stacked feature matrix. This is the key difference vs. the
        #    previous MultiModalData + ``join_branches`` design: there each
        #    branch saw a DIFFERENT partition during fit, so after
        #    ``DataMerger`` truncated to ``min(partition_length)`` the row k
        #    of the head's training matrix contained predictions from four
        #    branches on four UNRELATED samples. Only column 0 correlated
        #    with the target (branch 0 was trained on partition 0 and on its
        #    own partition's rows predicted perfectly), and the head
        #    silently degenerated to "copy branch 0" -- which at inference
        #    collapsed to whatever branch 0's pipeline emitted (often a
        #    single-class constant when the branch itself under-fit).
        #
        #    Here, the same set of rows feeds every branch at both fit and
        #    predict time, so the head sees a properly-aligned feature
        #    matrix and can learn a real combination.
        stacked_train = self._stack_predictions(self._branches, train_data)

        # 3. fit the head on (stacked_train, full_target).
        self._head = self._fit_head(stacked_train, train_data.target)

        # ``current_pipeline`` is kept for downstream code / serialisers that
        # expect a single FEDOT Pipeline to inspect; we expose the head since
        # it owns the final decision boundary.
        self.current_pipeline = self._head

    def predict(self, test_data, output_mode: str = 'labels'):
        stacked_test = self._stack_predictions(self._branches, test_data)
        head_input = InputData(
            idx=np.arange(stacked_test.shape[0]),
            features=stacked_test,
            target=(np.asarray(test_data.target)
                    if getattr(test_data, 'target', None) is not None else None),
            task=self.task,
            data_type=DataTypesEnum.table)
        return self._head.predict(head_input, output_mode).predict

    def _fit_branches(self,
                      features_splits: List[np.ndarray],
                      target_splits: List[np.ndarray]) -> List[Pipeline]:
        """Build and fit one ``data_source/i -> atomized_automl`` pipeline
        per partition.  Each branch is an independent FEDOT composing run
        on its own slice of the training data; the TS-specific operations
        pool (InceptionTime, industrial_*_clf, quantile_extractor, ...) is
        fully available inside each branch — that is the point of RAF.

        The source node is kept because Industrial's atomized operations
        (``fedot_cls`` / ``fedot_regr``) are registered with FEDOT's
        repository only when routed through a ``data_source_*`` node via
        ``MultiModalData``.
        """
        branches: List[Pipeline] = []
        for idx, (features, target) in enumerate(zip(features_splits, target_splits)):
            source_name = f'{self.source_prefix}/{idx}'
            fold = InputData(
                idx=np.arange(len(features)),
                features=np.asarray(features),
                target=np.asarray(target),
                task=self.task,
                data_type=self.data_type)
            branch = (PipelineBuilder()
                      .add_node(operation_type=source_name, branch_idx=0)
                      .add_node(self.atomized_automl,
                                params=deepcopy(self.atomized_automl_params),
                                branch_idx=0)
                      .build())
            branch.fit(input_data=MultiModalData({source_name: fold}))
            branches.append(branch)
        return branches

    def _stack_predictions(self,
                           branches: List[Pipeline],
                           input_data: InputData) -> np.ndarray:
        """Run every branch on the same ``input_data`` and column-concatenate
        their outputs.  For classification we pull probabilities (one column
        per class per branch) so the head sees a continuous feature space;
        for regression the default scalar output is used.
        """
        branch_mode = 'full_probs' if self.problem == 'classification' else 'labels'
        target = (np.asarray(input_data.target)
                  if getattr(input_data, 'target', None) is not None else None)
        features = np.asarray(input_data.features)
        idx_arr = np.asarray(input_data.idx)

        columns: List[np.ndarray] = []
        for idx, branch in enumerate(branches):
            source_name = f'{self.source_prefix}/{idx}'
            fold = InputData(
                idx=idx_arr, features=features, target=target,
                task=self.task, data_type=self.data_type)
            mmd = MultiModalData({source_name: fold})
            try:
                raw = branch.predict(mmd, output_mode=branch_mode).predict
            except (TypeError, ValueError):
                raw = branch.predict(mmd).predict
            raw = np.asarray(raw)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            columns.append(raw)
        return np.concatenate(columns, axis=1).astype(np.float32, copy=False)

    def _fit_head(self, stacked: np.ndarray, target: np.ndarray) -> Pipeline:
        """Fit the meta-learner on the row-aligned stacked features.

        The head always operates on a plain ``(n_samples, n_stacked_features)``
        tabular matrix, regardless of the original ``data_type`` of the
        branches, so we pin its InputData to ``DataTypesEnum.table``.
        """
        head_input = InputData(
            idx=np.arange(stacked.shape[0]),
            features=stacked,
            target=np.asarray(target),
            task=self.task,
            data_type=DataTypesEnum.table)
        head = PipelineBuilder().add_node(self.head).build()
        head.fit(input_data=head_input)
        return head
