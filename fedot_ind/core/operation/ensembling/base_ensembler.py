"""Ensembling strategies for :class:`RAFEnsembler`.

Three Pipeline-level Ensembling Architectures (PEA) that mirror the
partitioning factory in :mod:`.partitioning`:

* :class:`PEABagEnsembler` -- soft-/hard-vote averaging across
  independently-fitted AutoML branches.
* :class:`PEABoostEnsembler` -- sequential boosting of whole AutoML
  pipelines (SAMME for classification, gradient-boosting on residuals
  for regression). Each next AutoML focuses on the current ensemble's
  errors.
* :class:`PEAStackEnsembler` -- branch predictions are used as features
  for a meta-learner (XGBoost / logistic regression / etc.). This is
  the original RAF behaviour, factored out of :class:`RAFEnsembler` so
  every ensembling method shares the same plug-in surface.

Every ensembler implements two methods and operates on a
:class:`RAFEnsembler` host:

``fit(raf, train_data, partitioner)``
    Use ``raf._fit_single_branch`` to train one AutoML pipeline at a
    time. Stacking & bagging consume the partitioner; boosting ignores
    it (sequential weighted resampling on the full set).

``predict(raf, test_data, output_mode)``
    Combine branch predictions into the final tensor. ``output_mode``
    matches FEDOT's convention: ``'labels'`` for hard outputs,
    ``'probs'`` / ``'full_probs'`` / ``'default'`` for soft outputs.

The :class:`EnsembleStrategy` factory builds a concrete ensembler from
``ensembling_method`` + ``ensembling_params``, the same way
``FeatureSpacePartitioner`` builds partitioners.
"""
import inspect
import logging
from abc import abstractmethod
from copy import deepcopy
from typing import List, Optional, Tuple

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot_ind.core.architecture.settings.computational import backend_methods as np


class BaseEnsembler:
    """Abstract base class for branch-prediction ensembling strategies.

    Args:
        n_splits: desired number of branches (== AutoML pipelines) in
            the ensemble. For partition-based methods this matches the
            partitioner's ``n_splits``; for boosting it is the number
            of sequential rounds.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fit(self, raf, train_data, partitioner) -> None:
        """Train the ensemble; populate ``raf._branches`` and any extra
        per-strategy state."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, raf, test_data, output_mode: str = 'labels') -> np.ndarray:
        """Return the combined ensemble prediction."""
        raise NotImplementedError

    @staticmethod
    def _wants_labels(output_mode: str) -> bool:
        """FEDOT's ``output_mode`` semantics: ``'labels'`` => integer
        class indices; everything else (``'probs'``, ``'full_probs'``,
        ``'default'``) => probabilities."""
        return str(output_mode).lower() == 'labels'

    @staticmethod
    def _probs_to_labels(probs: np.ndarray) -> np.ndarray:
        """Argmax classes, returned as a 1-D int vector (matches the
        shape FEDOT's ``predict`` would emit for labels)."""
        if probs.ndim == 1:
            return (probs > 0.5).astype(int)
        if probs.shape[1] == 1:
            return (probs[:, 0] > 0.5).astype(int)
        return np.argmax(probs, axis=1).astype(int)


class PEABagEnsembler(BaseEnsembler):
    """PEA-Bag: parallel AutoML pipelines combined by averaging / voting.

    Every branch is fitted on its own partition (independence is the
    point); at predict time the per-branch outputs are aggregated with
    a single rule:

    * ``voting='soft'`` -- average per-class probabilities (default,
      best calibrated).
    * ``voting='hard'`` -- majority vote on argmax labels (broken ties
      resolve to the lowest class id, sklearn-style).

    For regression both modes collapse to the arithmetic mean of the
    branch outputs.

    Args:
        n_splits: number of branches.
        voting: ``'soft'`` (default) or ``'hard'``.
    """

    def __init__(self, n_splits: int = 5, voting: str = 'soft'):
        super().__init__(n_splits)
        if voting not in ('soft', 'hard'):
            raise ValueError(f"voting must be 'soft' or 'hard', got {voting!r}")
        self.voting = voting

    def fit(self, raf, train_data, partitioner) -> None:
        features_splits, target_splits = partitioner.partition(
            train_data.features, train_data.target)
        raf.n_splits = len(features_splits)
        raf._branches = [
            raf._fit_single_branch(idx, f, t)
            for idx, (f, t) in enumerate(zip(features_splits, target_splits))
        ]
        # No head model -- aggregation is parameter-free at predict time.
        raf._head = None
        raf.current_pipeline = raf._branches[0]

    def predict(self, raf, test_data, output_mode: str = 'labels') -> np.ndarray:
        columns = raf._collect_branch_predictions(raf._branches, test_data)
        # ``columns`` is a list of (N, k) matrices; for classification
        # k == n_classes, for regression k == 1.
        if raf.problem == 'classification':
            return self._aggregate_classification(columns, output_mode)
        return self._aggregate_regression(columns)

    def _aggregate_classification(self,
                                   columns: List[np.ndarray],
                                   output_mode: str) -> np.ndarray:
        if self.voting == 'soft':
            probs = np.mean(np.stack(columns, axis=0), axis=0)
            if self._wants_labels(output_mode):
                return self._probs_to_labels(probs)
            return probs
        # hard voting: stack labels (N, n_branches) and pick the modal
        # class per row; for soft output we expand the modal vote into
        # a one-hot proxy (sufficient for downstream metrics that only
        # need a probability surface).
        labels_per_branch = np.column_stack(
            [self._probs_to_labels(c) for c in columns])
        n_classes = columns[0].shape[1] if columns[0].ndim == 2 else 2
        labels = np.array([
            np.bincount(row, minlength=n_classes).argmax()
            for row in labels_per_branch
        ], dtype=int)
        if self._wants_labels(output_mode):
            return labels
        votes = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        for cls in range(n_classes):
            votes[:, cls] = (labels_per_branch == cls).mean(axis=1)
        return votes

    @staticmethod
    def _aggregate_regression(columns: List[np.ndarray]) -> np.ndarray:
        # collapse trailing singleton axes so the output mirrors what
        # a single FEDOT regressor would emit.
        stacked = np.stack([c.reshape(c.shape[0], -1) for c in columns], axis=0)
        mean = stacked.mean(axis=0)
        if mean.shape[1] == 1:
            return mean[:, 0]
        return mean


class PEABoostEnsembler(BaseEnsembler):
    """PEA-Boost: sequential boosting of whole AutoML pipelines.

    The ``partitioner`` argument is ignored on purpose -- boosting
    rewrites the training distribution between rounds, so a fixed
    partitioning would defeat the strategy. ``n_splits`` is reused as
    the number of boosting rounds.

    * Classification: AdaBoost-SAMME on class labels.
      Each round resamples the full training set with replacement
      using the current weight vector; the branch is then fit on the
      resampled bootstrap and evaluated on the full set. Round-mass
      ``alpha_m = log((1 - err_m) / err_m) + log(K - 1)`` is the
      standard SAMME estimator weight.
    * Regression: gradient boosting with squared-error residuals.
      Round 0 stores the training-target mean; round ``m`` fits the
      branch on the residuals of the current ensemble and contributes
      ``learning_rate * h_m(x)`` to the output.

    The hypothesis behind PEA-Boost is that boosting *whole AutoML
    pipelines* yields stronger committees than boosting plain stumps
    -- each "weak learner" is itself an optimised pipeline, so the
    ensemble corrects errors at a coarser level than classic boosting.

    Args:
        n_splits: number of boosting rounds.
        learning_rate: shrinkage applied to each round's contribution
            (regression only; classification uses SAMME's natural
            ``alpha`` weights).
        early_stop_eps: classification rounds with ``err >= 1 - 1/K - eps``
            (worse than random) are dropped and the loop exits.
        random_state: seed for the weighted bootstrap RNG.
    """

    def __init__(self,
                 n_splits: int = 5,
                 learning_rate: float = 0.1,
                 early_stop_eps: float = 1e-3,
                 random_state: int = 42):
        super().__init__(n_splits)
        if learning_rate <= 0:
            raise ValueError(f'learning_rate must be > 0, got {learning_rate}')
        self.learning_rate = float(learning_rate)
        self.early_stop_eps = float(early_stop_eps)
        self.random_state = int(random_state)
        self.alphas_: List[float] = []
        self.init_value_: Optional[float] = None
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, raf, train_data, partitioner) -> None: 
        features = np.asarray(train_data.features)
        target = np.asarray(train_data.target).ravel()
        n_samples = features.shape[0]

        rng = np.random.default_rng(self.random_state)
        if raf.problem == 'classification':
            self._fit_classification(raf, features, target, rng, n_samples)
        else:
            self._fit_regression(raf, features, target)

        raf.n_splits = len(raf._branches)
        raf._head = None
        raf.current_pipeline = raf._branches[0] if raf._branches else None

    def _fit_classification(self, raf, features, target, rng, n_samples):
        self.classes_ = np.unique(target)
        self.n_classes_ = self.classes_.size
        # SAMME's "worse-than-random" threshold for K classes:
        random_err = 1.0 - 1.0 / max(self.n_classes_, 2)

        weights = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
        target_int = np.searchsorted(self.classes_, target).astype(int)
        raf._branches = []
        self.alphas_ = []

        for m in range(self.n_splits):
            sampled_idx = rng.choice(n_samples, size=n_samples,
                                     replace=True, p=weights)

            present = set(target_int[sampled_idx].tolist())
            missing = [c for c in range(self.n_classes_) if c not in present]
            if missing:
                replace_positions = rng.choice(
                    n_samples, size=len(missing), replace=False)
                for pos, cls in zip(replace_positions, missing):
                    candidates = np.where(target_int == cls)[0]
                    sampled_idx[pos] = int(rng.choice(candidates))
            features_m = features[sampled_idx]
            target_m = target[sampled_idx]
            try:
                branch = raf._fit_single_branch(m, features_m, target_m)
            except Exception as err:  # noqa: BLE001
                self.logger.warning(
                    f'PEA-Boost round {m}: branch fit failed ({err!r}); '
                    f'stopping early')
                break

            preds_int = self._branch_labels(raf, branch, m, features, target)
            miss = (preds_int != target_int).astype(np.float64)
            err_m = float(np.average(miss, weights=weights))
            err_m = float(np.clip(err_m, 1e-12, 1.0 - 1e-12))

            if err_m >= random_err - self.early_stop_eps:
                self.logger.info(
                    f'PEA-Boost round {m}: err={err_m:.4f} >= random '
                    f'{random_err:.4f}; stopping')
                break

            alpha = float(np.log((1.0 - err_m) / err_m)
                          + np.log(self.n_classes_ - 1))
            raf._branches.append(branch)
            self.alphas_.append(alpha)
            self.logger.info(
                f'PEA-Boost round {m}: err={err_m:.4f} alpha={alpha:.4f}')

            weights = weights * np.exp(alpha * miss)
            total = float(weights.sum())
            if total <= 0 or not np.isfinite(total):
                self.logger.warning(
                    f'PEA-Boost round {m}: degenerate weights; stopping')
                break
            weights = weights / total

        if not raf._branches:
            raise RuntimeError(
                'PEA-Boost: no branch produced a better-than-random fit; '
                'try a larger timeout per AutoML round or fewer rounds.')

    def _fit_regression(self, raf, features, target):
        target = target.astype(float)
        self.init_value_ = float(np.mean(target))
        residuals = target - self.init_value_
        raf._branches = []
        self.alphas_ = []

        for m in range(self.n_splits):
            try:
                branch = raf._fit_single_branch(m, features, residuals)
            except Exception as err:  # noqa: BLE001
                self.logger.warning(
                    f'PEA-Boost round {m}: branch fit failed ({err!r}); '
                    f'stopping early')
                break
            pred = self._branch_regression(raf, branch, m, features)
            raf._branches.append(branch)
            self.alphas_.append(self.learning_rate)
            residuals = residuals - self.learning_rate * pred
            self.logger.info(
                f'PEA-Boost regression round {m}: '
                f'residual std={residuals.std():.4f}')

        if not raf._branches:
            raise RuntimeError(
                'PEA-Boost regression: no branch was fitted successfully.')

    def predict(self, raf, test_data, output_mode: str = 'labels') -> np.ndarray:
        if not raf._branches:
            raise RuntimeError('PEA-Boost: ensemble is empty; call fit first.')
        if raf.problem == 'classification':
            return self._predict_classification(raf, test_data, output_mode)
        return self._predict_regression(raf, test_data)

    def _predict_classification(self, raf, test_data, output_mode):
        n_classes = self.n_classes_ or raf._n_classes or 2
        votes = np.zeros((self._n_test_rows(test_data), n_classes), dtype=np.float64)
        for idx, (branch, alpha) in enumerate(zip(raf._branches, self.alphas_)):
            preds_int = self._branch_labels(raf, branch, idx,
                                             test_data.features,
                                             getattr(test_data, 'target', None))
            for cls in range(n_classes):
                votes[preds_int == cls, cls] += alpha

        if self._wants_labels(output_mode):
            labels_int = votes.argmax(axis=1)
            return self.classes_[labels_int] if self.classes_ is not None else labels_int

        # Normalise to a probability surface; SAMME's natural mapping
        # exp((1/K) * alpha * 1[pred==k]) followed by row-normalisation
        # is monotone-equivalent to softmax over `votes` and avoids the
        # exp overflow when alphas accumulate.
        votes -= votes.max(axis=1, keepdims=True)
        exp_votes = np.exp(votes)
        probs = exp_votes / exp_votes.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def _predict_regression(self, raf, test_data):
        features = np.asarray(test_data.features)
        out = np.full(features.shape[0], self.init_value_, dtype=np.float64)
        for idx, (branch, alpha) in enumerate(zip(raf._branches, self.alphas_)):
            pred = self._branch_regression(raf, branch, idx, features)
            out = out + alpha * pred
        return out

    def _branch_labels(self, raf, branch, branch_idx, features,
                        target) -> np.ndarray:
        """Run a single classification branch on ``features`` and return
        a 1-D vector of class indices into ``self.classes_``.
        """
        td = _SimpleData(
            features=np.asarray(features),
            target=(np.asarray(target) if target is not None else None))
        cols = raf._collect_branch_predictions(
            [branch], td, branch_indices=[branch_idx])
        probs = cols[0]
        labels = self._probs_to_labels(probs)
        if self.classes_ is None:
            return labels.astype(int)
        # Map argmax-column-index back to class id, then to compact
        # 0..K-1 indices used by the vote matrix.
        if probs.ndim == 2 and probs.shape[1] == self.classes_.size:
            return labels.astype(int)
        # branch returned hard labels in original class space
        idx = np.searchsorted(self.classes_, labels)
        return idx.astype(int)

    @staticmethod
    def _branch_regression(raf, branch, branch_idx, features) -> np.ndarray:
        """Run a single regression branch and return a 1-D prediction
        vector. ``branch_idx`` selects the matching FEDOT source name;
        see :meth:`_branch_labels` for the rationale."""
        td = _SimpleData(features=np.asarray(features), target=None)
        cols = raf._collect_branch_predictions(
            [branch], td, branch_indices=[branch_idx])
        out = cols[0]
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out.ravel()

    @staticmethod
    def _n_test_rows(test_data) -> int:
        return int(np.asarray(test_data.features).shape[0])


class PEAStackEnsembler(BaseEnsembler):
    """PEA-Stack: meta-learner over branch predictions.

    Branches are trained on partitions; their predictions on the FULL
    training set are column-concatenated into a feature matrix that
    feeds a meta-learner (the "head"). This is the original
    :class:`RAFEnsembler` behaviour, factored out so partitioning and
    ensembling can be swapped independently.

    Args:
        n_splits: number of branches / partitions.
        head: FEDOT operation name for the meta-learner (e.g.
            ``'xgboost'``, ``'logit'``, ``'rf'``). When ``None`` the
            problem-default from ``FEDOT_HEAD_ENSEMBLE`` is used.
    """

    def __init__(self, n_splits: int = 5, head: Optional[str] = None):
        super().__init__(n_splits)
        self.head = head

    def fit(self, raf, train_data, partitioner) -> None:
        features_splits, target_splits = partitioner.partition(
            train_data.features, train_data.target)
        raf.n_splits = len(features_splits)

        raf._branches = [
            raf._fit_single_branch(idx, f, t)
            for idx, (f, t) in enumerate(zip(features_splits, target_splits))
        ]

        stacked_train = raf._stack_predictions(raf._branches, train_data)

        head_op = self.head or raf.head
        raf._head = self._fit_head(raf, head_op, stacked_train, train_data.target)
        raf.current_pipeline = raf._head

    def predict(self, raf, test_data, output_mode: str = 'labels') -> np.ndarray:
        if raf._head is None:
            raise RuntimeError('PEA-Stack: head model is not fitted.')
        stacked_test = raf._stack_predictions(raf._branches, test_data)
        head_input = InputData(
            idx=np.arange(stacked_test.shape[0]),
            features=stacked_test,
            target=(np.asarray(test_data.target)
                    if getattr(test_data, 'target', None) is not None else None),
            task=raf.task,
            data_type=DataTypesEnum.table)
        return raf._head.predict(head_input, output_mode).predict

    @staticmethod
    def _fit_head(raf, head_op: str, stacked: np.ndarray,
                  target: np.ndarray) -> Pipeline:
        head_input = InputData(
            idx=np.arange(stacked.shape[0]),
            features=stacked,
            target=np.asarray(target),
            task=raf.task,
            data_type=DataTypesEnum.table)
        head = PipelineBuilder().add_node(head_op).build()
        head.fit(input_data=head_input)
        return head


class _SimpleData:
    """Lightweight ``InputData``-shaped duck for boosting helpers.

    :meth:`RAFEnsembler._collect_branch_predictions` only looks at
    ``features`` / ``target`` / ``idx`` -- this avoids constructing a
    full ``InputData`` per inner call.
    """

    __slots__ = ('features', 'target', 'idx')

    def __init__(self, features: np.ndarray, target: Optional[np.ndarray]):
        self.features = features
        self.target = target
        self.idx = np.arange(features.shape[0])


class EnsembleStrategy:
    """Factory that instantiates an ensembler by name.

    Supported methods:

    * ``'stacking'`` -- :class:`PEAStackEnsembler` (default; original
      RAF behaviour).
    * ``'bagging'`` -- :class:`PEABagEnsembler` (soft / hard voting).
    * ``'boosting'`` -- :class:`PEABoostEnsembler` (SAMME for
      classification, gradient-boosting on residuals for regression).

    The aliases ``'pea_stack'`` / ``'pea_bag'`` / ``'pea_boost'`` are
    accepted as well, matching the names used in the literature.
    """

    ENSEMBLER_REGISTRY = {
        'stacking': PEAStackEnsembler,
        'pea_stack': PEAStackEnsembler,
        'bagging': PEABagEnsembler,
        'pea_bag': PEABagEnsembler,
        'boosting': PEABoostEnsembler,
        'pea_boost': PEABoostEnsembler,
    }

    @classmethod
    def create(cls,
               method: str = 'stacking',
               n_splits: int = 5,
               params: Optional[dict] = None) -> BaseEnsembler:
        params = dict(params or {})
        method = (method or 'stacking').lower()
        if method not in cls.ENSEMBLER_REGISTRY:
            raise ValueError(
                f"Unknown ensembling method '{method}'. "
                f"Available: {sorted(set(cls.ENSEMBLER_REGISTRY.keys()))}")
        ensembler_cls = cls.ENSEMBLER_REGISTRY[method]
        effective_n_splits = params.pop('n_splits', n_splits)
        # filter to kwargs the concrete ensembler actually accepts so
        # that one shared ``ensembling_params`` dict can be reused.
        accepted = set(inspect.signature(ensembler_cls.__init__).parameters)
        kwargs = {k: v for k, v in params.items() if k in accepted}
        return ensembler_cls(n_splits=effective_n_splits, **kwargs)
