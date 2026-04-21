from copy import deepcopy

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.partitioning import FeatureSpacePartitioner
from fedot_ind.core.repository.constanst_repository import FEDOT_ATOMIZE_OPERATION, FEDOT_HEAD_ENSEMBLE, FEDOT_TASK
from fedot_ind.core.repository.model_repository import SKLEARN_CLF_MODELS, SKLEARN_REG_MODELS, default_industrial_availiable_operation


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
        self.problem = composing_params['problem']
        self.task = FEDOT_TASK[composing_params['problem']]
        self.atomized_automl = FEDOT_ATOMIZE_OPERATION[composing_params['problem']]
        self.head = FEDOT_HEAD_ENSEMBLE[composing_params['problem']]

        self.ensemble_method = self._raf_ensemble
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

        new_features, new_target = partitioner.partition(
            train_data.features, train_data.target)

        self.n_splits = len(new_features)

        self.current_pipeline = self.ensemble_method(new_features,
                                                     new_target,
                                                     n_splits=self.n_splits)

    def predict(self, test_data, output_mode: str = 'labels'):
        test_multimodal = self._to_multimodal(test_data)
        return self.current_pipeline.predict(test_multimodal, output_mode).predict

    def _to_multimodal(self, input_data):
        """Convert InputData to MultiModalData matching the training format."""
        data_dict = {}
        for i in range(self.n_splits):
            fold_data = InputData(idx=input_data.idx,
                                  features=input_data.features,
                                  target=input_data.target,
                                  task=self.task,
                                  data_type=self.data_type)
            data_dict[f'{self.source_prefix}/{i}'] = fold_data
        return MultiModalData(data_dict)

    def _raf_ensemble(self, features, target, n_splits):
        raf_ensemble = PipelineBuilder()
        data_dict = {}
        for i, data_fold_features, data_fold_target in zip(range(n_splits), features, target):

            train_fold = InputData(idx=np.arange(0, len(data_fold_features)),
                                   features=data_fold_features,
                                   target=data_fold_target,
                                   task=self.task,
                                   data_type=self.data_type)

            raf_ensemble.add_node(operation_type=f'{self.source_prefix}/{i}',
                                  branch_idx=i)\
                .add_node(self.atomized_automl,
                          params=self.atomized_automl_params,
                          branch_idx=i)

            data_dict.update({f'{self.source_prefix}/{i}': train_fold})
        train_multimodal = MultiModalData(data_dict)
        head_automl_params = deepcopy(self.atomized_automl_params)

        head_automl_params['available_operations'] = [
            operation for operation in head_automl_params['available_operations'] if operation in list(
                SKLEARN_CLF_MODELS.keys()) or operation in list(
                SKLEARN_REG_MODELS.keys())]

        raf_ensemble = raf_ensemble.join_branches(self.head).build()
        raf_ensemble.fit(input_data=train_multimodal)
        return raf_ensemble
