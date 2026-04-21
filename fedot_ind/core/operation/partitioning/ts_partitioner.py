"""Time-series partitioning strategies for :class:`RAFEnsembler`.

Three strategies that complement the tabular ones in
:mod:`.cluster_partitioner`:

* :class:`TemporalSplitPartitioner` (TS1) -- contiguous blocks along the
  time/sample axis. Respects the original ordering (``_finalize`` does
  not shuffle within a TS partition).
* :class:`TSFeatureClusteringPartitioner` (TS2) -- per-series descriptor
  vectors (mean / std / trend / lag-1 AC / dominant FFT amplitude /
  entropy) clustered with K-Means. Groups series with similar patterns
  (trend, seasonality, volatility).
* :class:`TSModelDifficultyPartitioner` (TS3) -- a teacher model
  (``Ridge`` by default) is fit once on the full training set; samples
  are sorted by absolute residual and bucketed into ``n_splits``
  difficulty bands. No cross-validation, which would reshuffle time.

TS2 and TS3 expect each sample (row of ``features``) to be one series
or one windowed sub-series; they raise :class:`ValueError` on 1-D input
and point the caller at the M4 example for windowing.
"""
from typing import List, Optional, Sequence, Tuple

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.partitioning.cluster_partitioner import BasePartitioner


class TemporalSplitPartitioner(BasePartitioner):
    """TS1: split along the temporal / sample axis into contiguous blocks.

    The input is treated as already being in temporal order (which is
    the loader's contract for M4 / Monash and the row-order contract
    for multi-series TS benchmarks). The partitioner slices ``features``
    into ``n_splits`` contiguous chunks and slices ``target`` in
    lockstep when it is the same length, or replicates it across every
    partition when it is a short forecast horizon tail (the M4 loader's
    case: ``features`` is the full series, ``target`` is the last
    ``forecast_length`` points).

    Args:
        n_splits: number of contiguous temporal blocks.
        overlap: fraction ``[0, 0.9)`` of chunk length that each block
            reuses from the preceding one. Default 0 (disjoint blocks).
            Useful when every chunk needs extra context to train a
            forecaster.
        min_chunk_size: minimum samples per chunk. If ``n_splits`` would
            produce chunks smaller than this, ``n_splits`` is reduced.
    """

    _shuffle_within_partitions = False

    def __init__(self,
                 n_splits: int = 5,
                 overlap: float = 0.0,
                 min_chunk_size: int = 2):
        super().__init__(n_splits)
        if not 0.0 <= overlap < 0.9:
            raise ValueError(f'overlap must be in [0, 0.9), got {overlap}')
        self.overlap = overlap
        self.min_chunk_size = max(1, int(min_chunk_size))

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = np.asarray(features)
        target = np.asarray(target)
        n = features.shape[0]

        effective_n_splits = max(1, min(self.n_splits, n // self.min_chunk_size or 1))
        if effective_n_splits < self.n_splits:
            self.logger.warning(
                f'n_splits={self.n_splits} too large for n_samples={n} '
                f'and min_chunk_size={self.min_chunk_size}; '
                f'reduced to {effective_n_splits}')

        chunk = max(self.min_chunk_size, n // effective_n_splits)
        overlap_len = int(chunk * self.overlap)

        features_splits: List[np.ndarray] = []
        target_splits: List[np.ndarray] = []
        sync_target = target.shape[0] == n

        for i in range(effective_n_splits):
            start = max(0, i * chunk - overlap_len)
            end = n if i == effective_n_splits - 1 else (i + 1) * chunk
            features_splits.append(features[start:end])
            if sync_target:
                target_splits.append(target[start:end])
            else:
                # short forecast-horizon target: every worker sees it
                target_splits.append(target.copy())

        return self._finalize(features_splits, target_splits)


class TSFeatureClusteringPartitioner(BasePartitioner):
    """TS2: cluster series by hand-crafted statistical/spectral descriptors.

    Each row of ``features`` is treated as one 1-D time series and a
    compact descriptor vector is extracted from it. K-Means then groups
    series with similar trend / seasonality / volatility into the same
    partition.

    Default descriptors (all scale-sensitive; ``scale_features``
    z-scores the descriptor matrix before clustering):

    * ``mean``, ``std`` -- level and volatility.
    * ``trend`` -- slope of a linear fit against a time index.
    * ``lag1_ac`` -- lag-1 autocorrelation (seasonality proxy).
    * ``dom_fft`` -- amplitude of the dominant non-DC FFT bin.
    * ``entropy`` -- Shannon entropy of a 10-bin value histogram.

    Args:
        n_splits: number of clusters / partitions.
        features_to_use: subset of the default descriptors, or a
            callable ``series -> 1-D np.ndarray`` for a fully custom
            extractor.
        random_state: K-Means seed.
        scale_features: z-score descriptors before clustering.
        oversample_factor: unused for 2-D input. Kept for a future 1-D
            mode; see docstring of :meth:`partition` for why 1-D is
            unsupported today.
    """

    _AVAILABLE_FEATURES = ('mean', 'std', 'trend', 'lag1_ac', 'dom_fft', 'entropy')
    DEFAULT_FEATURES = _AVAILABLE_FEATURES

    def __init__(self,
                 n_splits: int = 5,
                 features_to_use=DEFAULT_FEATURES,
                 random_state: int = 42,
                 scale_features: bool = True,
                 oversample_factor: int = 4):
        super().__init__(n_splits)
        self.features_to_use = features_to_use
        self.random_state = random_state
        self.scale_features = scale_features
        self.oversample_factor = oversample_factor
        self.scaler: Optional[StandardScaler] = None
        self.clusterer: Optional[KMeans] = None
        self.descriptors_: Optional[np.ndarray] = None

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = np.asarray(features)
        target = np.asarray(target)
        self._check_2d(features)

        flat = self._flatten_features(features)
        descriptors = self._extract_descriptors(flat)
        self.descriptors_ = descriptors

        clustering_input = descriptors
        if self.scale_features:
            self.scaler = StandardScaler()
            clustering_input = self.scaler.fit_transform(descriptors)

        n_samples = flat.shape[0]
        effective_n_splits = min(self.n_splits, n_samples)
        if effective_n_splits < self.n_splits:
            self.logger.warning(
                f'n_splits={self.n_splits} exceeds n_samples={n_samples}. '
                f'Reduced to {effective_n_splits}')

        self.clusterer = KMeans(
            n_clusters=effective_n_splits,
            random_state=self.random_state,
            n_init='auto')
        cluster_labels = self.clusterer.fit_predict(clustering_input)

        features_splits, target_splits = [], []
        for cid in range(effective_n_splits):
            mask = cluster_labels == cid
            features_splits.append(features[mask])
            target_splits.append(target[mask])

        return self._finalize(features_splits, target_splits)

    def _check_2d(self, features: np.ndarray) -> None:
        if features.ndim < 2:
            raise ValueError(
                f'{type(self).__name__} expects 2-D features '
                f'(n_series, series_length); got shape {features.shape}.')

    def _extract_descriptors(self, x: np.ndarray) -> np.ndarray:
        """Compute the descriptor matrix.

        If ``features_to_use`` is callable, it is invoked once per row.
        Otherwise ``features_to_use`` is a subset of
        :attr:`_AVAILABLE_FEATURES` and the rows are vectorised where
        possible.
        """
        if callable(self.features_to_use):
            rows = [np.asarray(self.features_to_use(series), dtype=float).ravel()
                    for series in x]
            return np.vstack(rows)

        names: Sequence[str] = self.features_to_use
        unknown = set(names) - set(self._AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(
                f'Unknown TS descriptors {sorted(unknown)}. '
                f'Available: {self._AVAILABLE_FEATURES}')

        x = x.astype(float)
        columns: List[np.ndarray] = []
        for name in names:
            columns.append(self._descriptor(name, x))
        return np.column_stack(columns)

    @staticmethod
    def _descriptor(name: str, x: np.ndarray) -> np.ndarray:
        """Return a 1-D descriptor of length ``n_samples``."""
        if name == 'mean':
            return x.mean(axis=1)
        if name == 'std':
            return x.std(axis=1)
        if name == 'trend':
            t = np.arange(x.shape[1], dtype=float)
            # slope via closed-form least squares: cov(t, x) / var(t)
            t_centered = t - t.mean()
            var_t = max(float((t_centered ** 2).sum()), 1e-12)
            return ((x - x.mean(axis=1, keepdims=True)) * t_centered).sum(axis=1) / var_t
        if name == 'lag1_ac':
            return np.array([TSFeatureClusteringPartitioner._lag1_autocorr(row)
                             for row in x])
        if name == 'dom_fft':
            mags = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1))
            # skip DC (column 0); guard against all-zero series
            if mags.shape[1] <= 1:
                return np.zeros(x.shape[0])
            return mags[:, 1:].max(axis=1)
        if name == 'entropy':
            return np.array([TSFeatureClusteringPartitioner._hist_entropy(row)
                             for row in x])
        raise ValueError(f'Unknown descriptor {name!r}')

    @staticmethod
    def _lag1_autocorr(row: np.ndarray) -> float:
        if row.size < 2:
            return 0.0
        a, b = row[:-1], row[1:]
        a_c, b_c = a - a.mean(), b - b.mean()
        denom = float(np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum()))
        if denom < 1e-12:
            return 0.0
        return float((a_c * b_c).sum() / denom)

    @staticmethod
    def _hist_entropy(row: np.ndarray, bins: int = 10) -> float:
        if row.size == 0:
            return 0.0
        hist, _ = np.histogram(row, bins=bins)
        probs = hist.astype(float) / max(hist.sum(), 1)
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))


class TSModelDifficultyPartitioner(BasePartitioner):
    """TS3: bucket samples by teacher-model error.

    A single lightweight teacher is fit once on the whole training set;
    the per-sample error is the difficulty score. Samples are sorted by
    score and split into ``n_splits`` contiguous difficulty buckets.

    The default teacher is task-aware:

    * regression / forecasting -- ``Ridge(alpha=1.0)``; difficulty is
      the absolute residual ``|y - y_pred|``.
    * classification -- a shallow ``DecisionTreeClassifier``;
      difficulty is ``1 - p(y_true)`` when ``predict_proba`` is
      available, else the 0/1 mis-classification indicator.

    Cross-validated residuals -- as used by
    :class:`DifficultyPartitioner` -- are not used here because
    standard K-Fold CV reshuffles time, which is invalid for a TS
    curriculum. Teacher residuals are in-sample, which is acceptable
    for a partitioning signal (the teacher is deliberately weak and
    the split is coarse).

    Expects 2-D ``features`` (one row per sample / window). 1-D
    single-series input is rejected -- window the series first.

    Args:
        n_splits: number of difficulty buckets.
        teacher: pre-configured estimator implementing ``fit`` /
            ``predict`` (and ``predict_proba`` for classification).
            If ``None``, a task-appropriate default is picked.
        task_type: ``'classification'`` or ``'regression'``. If
            ``None``, inferred from ``target``.
        random_state: seed for the default teacher.
        scale_features: z-score features before fitting the teacher.
            Helpful for Ridge; harmless for trees.
    """

    _shuffle_within_partitions = False

    def __init__(self,
                 n_splits: int = 5,
                 teacher=None,
                 task_type: Optional[str] = None,
                 random_state: int = 42,
                 scale_features: bool = True):
        super().__init__(n_splits)
        self.teacher = teacher
        self.task_type = task_type
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler: Optional[StandardScaler] = None
        self.difficulty_: Optional[np.ndarray] = None

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features = np.asarray(features)
        target = np.asarray(target)
        self._check_2d(features)

        x = self._flatten_features(features)
        y = target.ravel()
        if self.scale_features:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x)

        task_type = self._infer_task_type(y, self.task_type)
        teacher = self._resolve_teacher(task_type)
        teacher.fit(x, y)
        difficulty = self._difficulty(teacher, x, y, task_type)
        self.difficulty_ = difficulty

        n_samples = len(y)
        effective_n_splits = min(self.n_splits, n_samples)
        if effective_n_splits < self.n_splits:
            self.logger.warning(
                f'n_splits={self.n_splits} exceeds n_samples={n_samples}. '
                f'Reduced to {effective_n_splits}')

        order = np.argsort(difficulty)
        buckets = np.array_split(order, effective_n_splits)
        features_splits = [features[idx] for idx in buckets]
        target_splits = [target[idx] for idx in buckets]

        self.logger.info(
            f'TSModelDifficultyPartitioner ({task_type}): difficulty '
            f'mean={difficulty.mean():.4f} '
            f'std={difficulty.std():.4f} '
            f'min={difficulty.min():.4f} max={difficulty.max():.4f}')
        return self._finalize(features_splits, target_splits)

    def _resolve_teacher(self, task_type: str):
        if self.teacher is not None:
            return self.teacher
        if task_type == 'classification':
            return DecisionTreeClassifier(
                max_depth=5, random_state=self.random_state)
        return Ridge(alpha=1.0, random_state=self.random_state)

    @staticmethod
    def _difficulty(teacher, x: np.ndarray, y: np.ndarray, task_type: str) -> np.ndarray:
        if task_type == 'classification' and hasattr(teacher, 'predict_proba'):
            proba = teacher.predict_proba(x)
            classes = teacher.classes_
            cols = np.searchsorted(classes, y)
            true_proba = proba[np.arange(len(y)), cols]
            return 1.0 - true_proba
        if task_type == 'classification':
            return (teacher.predict(x) != y).astype(float)
        y_pred = teacher.predict(x)
        return np.abs(y.astype(float) - y_pred.astype(float))

    def _check_2d(self, features: np.ndarray) -> None:
        if features.ndim < 2:
            raise ValueError(
                f'{type(self).__name__} expects 2-D features '
                f'(n_samples, window_size); got shape {features.shape}.')
