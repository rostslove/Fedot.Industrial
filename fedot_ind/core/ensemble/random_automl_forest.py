from copy import deepcopy
from typing import List

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.ensembling import EnsembleStrategy
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
    its own best pipeline.  The predictions of those workers are then
    combined by a configurable *ensembling strategy*:

    * ``'stacking'`` (default) -- a meta-learner trained on the row-aligned
      stacked feature matrix (original RAF behaviour).
    * ``'bagging'`` -- soft-/hard-vote averaging across branches.
    * ``'boosting'`` -- sequential AdaBoost-SAMME (classification) /
      gradient-boosting on residuals (regression). Ignores partitioning
      and reweights the full training set between rounds.

    Args:
        composing_params: dict with parameters for ensemble.  Recognised
            keys include ``problem``, ``timeout``, ``available_operations``,
            ``partitioning_method`` / ``partitioning_params`` for the
            data split, and ``ensembling_method`` / ``ensembling_params``
            for the prediction combiner.
        n_splits: number of data partitions / boosting rounds.
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

    Ensembling methods (``composing_params['ensembling_method']``):

    * ``'stacking'`` (default) -- meta-learner head over branch
      predictions. Tunable via ``ensembling_params['head']`` (operation
      name; defaults to ``FEDOT_HEAD_ENSEMBLE[problem]``).
    * ``'bagging'`` -- average / vote across branches.
      ``ensembling_params['voting']`` is ``'soft'`` (default) or
      ``'hard'``.
    * ``'boosting'`` -- sequential boosting of whole AutoML pipelines.
      ``ensembling_params['learning_rate']`` controls the regression
      shrinkage; ``early_stop_eps`` and ``random_state`` are also
      accepted.

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
        self.ensembling_method = self.atomized_automl_params.pop(
            'ensembling_method', 'stacking')
        self.ensembling_params = self.atomized_automl_params.pop(
            'ensembling_params', {})

        self.n_splits = n_splits
        self.batch_size = batch_size
        self._n_classes = None
        self.ensembler = None

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

        # Cache the classification class count so ``_normalize_branch_output``
        # can one-hot-expand any branch that silently returns labels instead
        # of full probabilities (see its docstring for why this happens).
        # ``None`` for regression — normaliser skips the expansion then.
        if self.problem == 'classification' and train_data.target is not None:
            self._n_classes = int(np.unique(np.asarray(train_data.target)).size)
        else:
            self._n_classes = None

        partitioner = FeatureSpacePartitioner.create(
            method=self.partitioning_method,
            n_splits=self.n_splits,
            params=self.partitioning_params)

        self.ensembler = EnsembleStrategy.create(
            method=self.ensembling_method,
            n_splits=self.n_splits,
            params=self.ensembling_params)

        # Each ensembler decides whether to use the partitioner (stacking
        # / bagging) or to ignore it (boosting reweights the full set).
        # On return ``self._branches`` is populated; stacking additionally
        # populates ``self._head``.
        self.ensembler.fit(self, train_data, partitioner)

    def predict(self, test_data, output_mode: str = 'labels'):
        if self.ensembler is None:
            raise RuntimeError('RAFEnsembler.predict called before fit')
        return self.ensembler.predict(self, test_data, output_mode)

    def _fit_single_branch(self,
                           idx: int,
                           features: np.ndarray,
                           target: np.ndarray) -> Pipeline:
        """Build and fit one ``data_source/i -> atomized_automl`` pipeline.

        The source name encodes the branch index so that
        :meth:`_collect_branch_predictions` can route the same input
        through the matching branch at inference time. Industrial's
        atomized operations (``fedot_cls`` / ``fedot_regr``) are
        registered with FEDOT's repository only when routed through a
        ``data_source_*`` node via :class:`MultiModalData`, which is why
        the source node is mandatory even when there is only one
        modality.
        """
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
        return branch

    def _collect_branch_predictions(self,
                                     branches: List[Pipeline],
                                     input_data,
                                     branch_indices=None) -> List[np.ndarray]:
        """Run every branch on the same ``input_data`` and return a list
        of normalised ``(N, k)`` matrices.

        For classification each matrix is ``(N, n_classes)`` of class
        probabilities; for regression it is ``(N, 1)`` of scalar
        predictions. Normalisation is delegated to
        :meth:`_normalize_branch_output` (see its docstring for the
        format-collision story).

        Args:
            branches: list of FEDOT pipelines built by
                :meth:`_fit_single_branch`.
            input_data: row-aligned ``InputData``-shaped duck.
            branch_indices: optional list of integer indices that maps
                each branch to the source name it was fitted with
                (``f'{prefix}/{idx}'``). Required when ``branches`` is
                a sub-slice of ``self._branches`` (e.g. boosting calls
                this with a single branch from round ``m``); without
                it the helper assumes the branches are passed in their
                original 0..N-1 order. Mismatches manifest as
                ``KeyError: 'data_source_<type>/0'`` from FEDOT's
                preprocessor, since the source name in the MultiModal
                envelope no longer matches the one the branch was fit
                with.
        """
        branch_mode = 'full_probs' if self.problem == 'classification' else 'labels'
        target = (np.asarray(input_data.target)
                  if getattr(input_data, 'target', None) is not None else None)
        features = np.asarray(input_data.features)
        idx_arr = np.asarray(getattr(input_data, 'idx', np.arange(features.shape[0])))

        if branch_indices is None:
            branch_indices = list(range(len(branches)))
        elif len(branch_indices) != len(branches):
            raise ValueError(
                f'branch_indices ({len(branch_indices)}) and branches '
                f'({len(branches)}) must have the same length')

        columns: List[np.ndarray] = []
        for idx, branch in zip(branch_indices, branches):
            source_name = f'{self.source_prefix}/{idx}'
            fold = InputData(
                idx=idx_arr, features=features, target=target,
                task=self.task, data_type=self.data_type)
            mmd = MultiModalData({source_name: fold})
            try:
                raw = branch.predict(mmd, output_mode=branch_mode).predict
            except (TypeError, ValueError):
                raw = branch.predict(mmd).predict
            columns.append(self._normalize_branch_output(np.asarray(raw)))
        return columns

    def _stack_predictions(self,
                           branches: List[Pipeline],
                           input_data) -> np.ndarray:
        """Column-concatenate every branch's normalised prediction.

        This is the row-aligned stacked feature matrix consumed by
        :class:`PEAStackEnsembler`'s head model. The key point is that
        every branch sees the SAME rows during stacking (whether at fit
        or predict time), so the head learns a real combination instead
        of degenerating to "copy branch 0" -- which was the failure mode
        of the original ``MultiModalData`` + ``join_branches`` path.
        """
        columns = self._collect_branch_predictions(branches, input_data)
        return np.concatenate(columns, axis=1).astype(np.float32, copy=False)

    def _normalize_branch_output(self, raw: np.ndarray) -> np.ndarray:
        """Coerce a branch's ``predict`` output into a 2-D ``(N, k)`` matrix
        so that column-concatenation across branches always succeeds.

        Each branch runs an independent FEDOT AutoML composer and the final
        pipeline it picks is not under our control — ``inception_model`` and
        ``cnn`` return ``(N, 1, n_classes)`` probabilities with an
        Industrial multi_dimensional "channel" axis; ``industrial_stat_clf``
        sometimes silently ignores ``output_mode='full_probs'`` and returns
        an ``(N,)`` label vector; classical sklearn heads return
        ``(N, n_classes)``. Without normalisation those formats collide at
        ``np.concatenate(axis=1)``.

        The normaliser does three things:

        1. Squeeze singleton middle axes (``(N, 1, k) -> (N, k)``); fall
           back to a flat reshape for any residual higher-rank output.
        2. Promote 1-D vectors to ``(N, 1)``.
        3. For classification only, when a branch returned a single column
           of integer-valued labels despite ``output_mode='full_probs'``
           being requested, expand it to a one-hot ``(N, n_classes)``
           matrix using ``self._n_classes`` inferred at fit time. This
           keeps every branch's contribution in a shared probability-like
           feature space for the head model.
        """
        if raw.ndim > 2:
            squeeze_axes = tuple(
                i for i in range(1, raw.ndim - 1) if raw.shape[i] == 1)
            if squeeze_axes:
                raw = raw.squeeze(axis=squeeze_axes)
            if raw.ndim > 2:
                raw = raw.reshape(raw.shape[0], -1)

        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)

        n_classes = getattr(self, '_n_classes', None)
        if (self.problem == 'classification'
                and n_classes is not None
                and n_classes > 1
                and raw.shape[1] == 1):
            vals = raw[:, 0]
            is_int_like = (np.issubdtype(vals.dtype, np.integer)
                           or bool(np.all(np.mod(vals, 1) == 0)))
            if is_int_like:
                labels = vals.astype(int)
                if labels.min() >= 0 and labels.max() < n_classes:
                    one_hot = np.zeros(
                        (labels.shape[0], n_classes), dtype=np.float32)
                    one_hot[np.arange(labels.shape[0]), labels] = 1.0
                    raw = one_hot
        return raw
