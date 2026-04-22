import inspect
import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot_ind.core.architecture.settings.computational import backend_methods as np


class BasePartitioner:
    """Abstract base class for dataset partitioning strategies.

    Partitioners split a feature matrix and its target into a list of
    disjoint subsets. Each subset is intended to be processed by an
    independent AutoML worker in the RAF ensemble.

    Args:
        n_splits: desired number of partitions.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split features and target into disjoint partitions."""
        raise NotImplementedError

    @staticmethod
    def _flatten_features(features: np.ndarray) -> np.ndarray:
        """Reshape features to 2-D ``(n_samples, n_features)`` for clustering."""
        if features.ndim > 2:
            return features.reshape(features.shape[0], -1)
        return features

    @staticmethod
    def _infer_task_type(target: np.ndarray,
                         explicit: Optional[str] = None) -> str:
        """Return ``'classification'`` or ``'regression'``.

        If ``explicit`` is provided it wins; otherwise a simple heuristic
        on ``target`` decides: integer dtype or "few distinct values"
        relative to the sample count ⇒ classification.
        """
        if explicit is not None:
            return explicit
        y = np.asarray(target).ravel()
        unique = np.unique(y)
        if np.issubdtype(y.dtype, np.integer) or \
                unique.size <= max(20, int(0.05 * len(y))):
            return 'classification'
        return 'regression'

    #: target fraction of the minority class after donation.  Set to
    #: 0.5 so that the first (most-balanced) partition reaches a true
    #: 50/50 split — this is important because ``RAFEnsembler`` uses
    #: the first partition as ``main_target`` for the downstream head
    #: model, and a skewed main target collapses the head into a
    #: constant predictor.
    _MINORITY_TARGET_RATIO = 0.5
    #: absolute floor on donation size, independent of partition size.
    _MIN_DONATION = 10

    def _finalize(self,
                  features_splits: List[np.ndarray],
                  target_splits: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Drop empty partitions, ensure class diversity, shuffle, and
        promote the most class-balanced partition to index 0.

        The ordering step matters for :class:`RAFEnsembler`, which treats
        the first partition as ``main_target`` for the downstream FEDOT
        ``DataMerger``. If the first partition is class-skewed, the head
        model silently collapses to a constant predictor.

        Every partition is ALWAYS shuffled in-place because
        ``DataMerger`` truncates every branch's output to the
        ``min(partition_length)`` prefix — an unshuffled prefix would
        feed the head model a set of rows with no cross-branch
        alignment (each branch's prefix is its own contiguous block of
        samples, not a matching slice) and collapse the head into a
        constant predictor.  This invariant is required by every
        partitioner subclass; TS partitioners that group by time,
        cluster, or difficulty still keep the *partition assignment*
        intact, only the row order inside a partition is randomised.

        For continuous (regression / TS forecasting) targets the
        class-diversity donation and the balance-based reordering are
        both skipped: every unique float value would otherwise be
        treated as its own "class", turning the donation loop into a
        no-op that still scans every sample.
        """
        features_splits, target_splits = zip(*[
            (f, t) for f, t in zip(features_splits, target_splits) if len(f) > 0
        ]) if any(len(f) > 0 for f in features_splits) else ([], [])
        features_splits, target_splits = list(features_splits), list(target_splits)

        is_classification = self._targets_are_classification(target_splits)
        if is_classification:
            features_splits, target_splits = self._ensure_class_diversity(
                features_splits, target_splits)

        rng = np.random.default_rng(1)
        for idx in range(len(features_splits)):
            order = rng.permutation(len(features_splits[idx]))
            features_splits[idx] = features_splits[idx][order]
            target_splits[idx] = target_splits[idx][order]

        if is_classification:
            features_splits, target_splits = self._reorder_by_balance(
                features_splits, target_splits)
        return features_splits, target_splits

    @classmethod
    def _targets_are_classification(cls, target_splits: List[np.ndarray]) -> bool:
        """Cheap check: concatenate all targets and ask :meth:`_infer_task_type`."""
        if not target_splits:
            return False
        concatenated = np.concatenate([t.ravel() for t in target_splits])
        return cls._infer_task_type(concatenated) == 'classification'

    @staticmethod
    def _reorder_by_balance(features_splits: List[np.ndarray],
                            target_splits: List[np.ndarray]
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Reorder partitions so that the most class-balanced one comes
        first.  Only meaningful for classification (≥2 global classes);
        for single-class / regression problems the original order is
        preserved.

        The score is the L1 distance between the partition's class
        distribution and the uniform distribution over global classes —
        smaller is better. Perfect 50/50 scores 0.
        """
        if len(target_splits) < 2:
            return features_splits, target_splits

        global_classes = np.unique(np.concatenate(
            [t.ravel() for t in target_splits]))
        if global_classes.size < 2:
            return features_splits, target_splits

        uniform = 1.0 / global_classes.size

        def imbalance(target):
            flat = target.ravel()
            n = len(flat)
            if n == 0:
                return float('inf')
            return sum(abs((np.sum(flat == c) / n) - uniform)
                       for c in global_classes)

        order = sorted(range(len(target_splits)),
                       key=lambda i: imbalance(target_splits[i]))
        return ([features_splits[i] for i in order],
                [target_splits[i] for i in order])

    def _ensure_class_diversity(self,
                                features_splits: List[np.ndarray],
                                target_splits: List[np.ndarray]
                                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Bring every class in every partition up to a viable ratio.

        For every (partition, class) pair where the class is absent or
        under-represented (below ``_MINORITY_TARGET_RATIO``), copy
        samples of that class from the partition that has the most of
        them. FEDOT runs non-stratified train/test splits and CV inside
        each worker and the head, so a class with only a handful of
        samples is effectively missing — its samples can all land in
        the test fold and disappear from the training fold.
        """
        target_flat = [t.ravel() for t in target_splits]
        global_classes = np.unique(np.concatenate(target_flat))
        if global_classes.size < 2:
            return features_splits, target_splits

        rng = np.random.default_rng(0)
        ratio = self._MINORITY_TARGET_RATIO

        for idx in range(len(target_splits)):
            for cls in global_classes.tolist():
                current_size = len(target_flat[idx])
                present_count = int(np.sum(target_flat[idx] == cls))
                # We want the post-donation partition to contain at
                # least ``ratio`` of class ``cls``, so that after FEDOT's
                # ~30% test split each fold still has enough samples.
                # Solving (present + d) / (current + d) >= ratio gives
                #     d >= (ratio*current - present) / (1 - ratio)
                ratio_target = int(np.ceil(
                    max(0.0, ratio * current_size - present_count) / (1.0 - ratio)))
                desired_min = max(self._MIN_DONATION, ratio_target + present_count)
                if present_count >= desired_min:
                    continue

                deficit = desired_min - present_count
                donor_idx = self._find_donor_with_class(
                    idx, cls, target_flat)
                if donor_idx is None:
                    continue

                donor_cls_indices = np.where(target_flat[donor_idx] == cls)[0]
                n_take = min(deficit, len(donor_cls_indices))
                if n_take <= 0:
                    continue
                chosen = rng.choice(
                    donor_cls_indices, size=n_take, replace=False)

                features_splits[idx] = np.concatenate(
                    [features_splits[idx], features_splits[donor_idx][chosen]],
                    axis=0)
                target_splits[idx] = np.concatenate(
                    [target_splits[idx], target_splits[donor_idx][chosen]],
                    axis=0)
                target_flat[idx] = target_splits[idx].ravel()
                self.logger.info(
                    f'Partition {idx}: donated {n_take} samples of class '
                    f'{cls} from partition {donor_idx}')

        return features_splits, target_splits

    @staticmethod
    def _find_donor_with_class(deficit_idx: int,
                               cls,
                               target_flat: List[np.ndarray]) -> Optional[int]:
        """Return the partition (other than ``deficit_idx``) that holds
        the largest number of samples of ``cls``."""
        best_idx = None
        best_count = 0
        for candidate, t in enumerate(target_flat):
            if candidate == deficit_idx:
                continue
            count = int(np.sum(t == cls))
            if count > best_count:
                best_count = count
                best_idx = candidate
        return best_idx


class SequentialPartitioner(BasePartitioner):
    """Split data into equal-sized sequential chunks.

    Mirrors the original ``np.array_split`` behaviour used by
    :class:`RAFEnsembler` before cluster-aware partitioning was added,
    and then routes the result through :meth:`_finalize` so that
    class-sorted inputs still satisfy the per-partition diversity
    contract.
    """

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        features_splits = list(np.array_split(features, self.n_splits))
        target_splits = list(np.array_split(target, self.n_splits))
        return self._finalize(features_splits, target_splits)


class KMeansPartitioner(BasePartitioner):
    """Partition data by clustering samples in feature space via K-Means.

    Args:
        n_splits: number of clusters / partitions.
        random_state: seed for reproducibility.
        scale_features: whether to z-score features before clustering.
    """

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42,
                 scale_features: bool = True):
        super().__init__(n_splits)
        self.random_state = random_state
        self.scale_features = scale_features
        self.clusterer: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        flat_features = self._flatten_features(features)
        clustering_input = self._scale(flat_features)

        n_samples = flat_features.shape[0]
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
        for cluster_id in range(effective_n_splits):
            mask = cluster_labels == cluster_id
            features_splits.append(features[mask])
            target_splits.append(target[mask])

        return self._finalize(features_splits, target_splits)

    def _scale(self, flat_features: np.ndarray) -> np.ndarray:
        if not self.scale_features:
            return flat_features
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(flat_features)


class DBSCANPartitioner(BasePartitioner):
    """Partition data by density-based clustering (DBSCAN).

    DBSCAN discovers clusters of arbitrary shape without a predefined
    count. Noise points are assigned to the nearest cluster, and the
    number of partitions is balanced towards ``n_splits`` by merging
    the smallest clusters or splitting the largest ones.

    Args:
        n_splits: target number of partitions (approximate for DBSCAN).
        eps: maximum neighbourhood distance.
        min_samples: minimum samples in a neighbourhood for a core point.
        scale_features: whether to z-score features before clustering.
    """

    def __init__(self,
                 n_splits: int = 5,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 scale_features: bool = True):
        super().__init__(n_splits)
        self.eps = eps
        self.min_samples = min_samples
        self.scale_features = scale_features
        self.clusterer: Optional[DBSCAN] = None
        self.scaler: Optional[StandardScaler] = None

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        flat_features = self._flatten_features(features)
        clustering_input = self._scale(flat_features)

        self.clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = self.clusterer.fit_predict(clustering_input)

        cluster_ids = sorted(cid for cid in set(cluster_labels) if cid != -1)
        if not cluster_ids:
            self.logger.warning(
                'DBSCAN found no clusters (all noise). '
                'Falling back to sequential partitioning')
            return SequentialPartitioner(self.n_splits).partition(features, target)

        features_by_cluster = {cid: features[cluster_labels == cid] for cid in cluster_ids}
        target_by_cluster = {cid: target[cluster_labels == cid] for cid in cluster_ids}

        self._absorb_noise(
            flat_features, features, target, cluster_labels,
            cluster_ids, features_by_cluster, target_by_cluster)

        # anything smaller than ``min_partition_size`` cannot reliably
        # survive FEDOT's internal CV, so we balance towards the average
        # partition size
        min_partition_size = max(1, len(features) // (self.n_splits * 2))
        features_splits, target_splits = self._balance_partitions(
            features_by_cluster, target_by_cluster, min_partition_size)

        return self._finalize(features_splits, target_splits)

    def _scale(self, flat_features: np.ndarray) -> np.ndarray:
        if not self.scale_features:
            return flat_features
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(flat_features)

    @staticmethod
    def _absorb_noise(flat_features: np.ndarray,
                      features: np.ndarray,
                      target: np.ndarray,
                      cluster_labels: np.ndarray,
                      cluster_ids: List[int],
                      features_by_cluster: Dict[int, np.ndarray],
                      target_by_cluster: Dict[int, np.ndarray]) -> None:
        """Assign DBSCAN noise samples to the nearest cluster centroid."""
        noise_mask = cluster_labels == -1
        if not np.any(noise_mask):
            return

        centroids = np.array([
            flat_features[cluster_labels == cid].mean(axis=0) for cid in cluster_ids
        ])
        noise_features = flat_features[noise_mask]
        distances = np.linalg.norm(
            noise_features[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2)
        nearest_clusters = np.array(cluster_ids)[np.argmin(distances, axis=1)]
        for idx, cid in zip(np.where(noise_mask)[0], nearest_clusters):
            features_by_cluster[cid] = np.concatenate(
                [features_by_cluster[cid], features[idx:idx + 1]], axis=0)
            target_by_cluster[cid] = np.concatenate(
                [target_by_cluster[cid], target[idx:idx + 1]], axis=0)

    def _balance_partitions(self,
                            features_by_cluster: Dict[int, np.ndarray],
                            target_by_cluster: Dict[int, np.ndarray],
                            min_partition_size: int = 1
                            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Merge or split clusters to approximate ``self.n_splits`` partitions.

        Partitions smaller than ``min_partition_size`` are absorbed into
        their nearest siblings before the split/merge step so that
        downstream CV always sees reasonably-sized folds.
        """
        self._absorb_tiny_clusters(
            features_by_cluster, target_by_cluster, min_partition_size)

        n_found = len(features_by_cluster)
        if n_found == self.n_splits:
            cluster_ids = sorted(features_by_cluster.keys())
            return ([features_by_cluster[cid] for cid in cluster_ids],
                    [target_by_cluster[cid] for cid in cluster_ids])

        if n_found > self.n_splits:
            return self._merge_smallest(features_by_cluster, target_by_cluster)
        return self._split_largest(
            features_by_cluster, target_by_cluster, min_partition_size)

    @staticmethod
    def _absorb_tiny_clusters(features_by_cluster: Dict[int, np.ndarray],
                              target_by_cluster: Dict[int, np.ndarray],
                              min_partition_size: int) -> None:
        """Merge every cluster smaller than ``min_partition_size`` into
        the next-smallest cluster. Operates in-place on the dicts."""
        if len(features_by_cluster) <= 1:
            return

        while True:
            sizes = {cid: len(f) for cid, f in features_by_cluster.items()}
            small = [cid for cid, s in sizes.items() if s < min_partition_size]
            if not small or len(sizes) <= 1:
                return
            # merge the smallest cluster into the next smallest so that
            # large clusters do not get even larger
            small.sort(key=lambda c: sizes[c])
            src = small[0]
            remaining = [cid for cid in sizes if cid != src]
            dst = min(remaining, key=lambda c: sizes[c])
            features_by_cluster[dst] = np.concatenate(
                [features_by_cluster[dst], features_by_cluster[src]], axis=0)
            target_by_cluster[dst] = np.concatenate(
                [target_by_cluster[dst], target_by_cluster[src]], axis=0)
            del features_by_cluster[src], target_by_cluster[src]

    def _merge_smallest(self,
                        features_by_cluster: Dict[int, np.ndarray],
                        target_by_cluster: Dict[int, np.ndarray]
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Iteratively merge the two smallest clusters until the target count is reached."""
        sizes = {cid: len(f) for cid, f in features_by_cluster.items()}
        while len(sizes) > self.n_splits:
            order = sorted(sizes, key=sizes.get)
            src, dst = order[0], order[1]
            features_by_cluster[dst] = np.concatenate(
                [features_by_cluster[dst], features_by_cluster[src]], axis=0)
            target_by_cluster[dst] = np.concatenate(
                [target_by_cluster[dst], target_by_cluster[src]], axis=0)
            sizes[dst] += sizes[src]
            del features_by_cluster[src], target_by_cluster[src], sizes[src]

        cluster_ids = sorted(features_by_cluster.keys())
        return ([features_by_cluster[cid] for cid in cluster_ids],
                [target_by_cluster[cid] for cid in cluster_ids])

    def _split_largest(self,
                       features_by_cluster: Dict[int, np.ndarray],
                       target_by_cluster: Dict[int, np.ndarray],
                       min_partition_size: int = 1
                       ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split the largest clusters in half until the target count is
        reached, without producing halves smaller than
        ``min_partition_size``.
        """
        features_list = list(features_by_cluster.values())
        target_list = list(target_by_cluster.values())

        while len(features_list) < self.n_splits:
            sizes = [len(f) for f in features_list]
            largest_idx = int(np.argmax(sizes))
            # refuse to split if either half would be too small
            if sizes[largest_idx] < max(2, 2 * min_partition_size):
                self.logger.warning(
                    f'Cannot split further: largest partition has '
                    f'{sizes[largest_idx]} samples (min={min_partition_size})')
                break
            feat = features_list.pop(largest_idx)
            tgt = target_list.pop(largest_idx)
            mid = len(feat) // 2
            features_list.extend([feat[:mid], feat[mid:]])
            target_list.extend([tgt[:mid], tgt[mid:]])

        return features_list, target_list


class DifficultyPartitioner(BasePartitioner):
    """Partition data by sample difficulty estimated from a weak model (T2).

    A lightweight "easy" model (shallow decision tree by default) is fit
    via cross-validation on the full training set; per-sample *difficulty*
    scores are derived from the resulting error signal:

    * classification: ``difficulty = 1 - p(y_true)`` using
      ``cross_val_predict(method='predict_proba')`` -- higher when the
      weak model is uncertain about the true label. Falls back to
      ``1[y_pred != y_true]`` when the model has no ``predict_proba``.
    * regression: absolute residual ``|y_true - y_pred|``.

    Samples are then sorted by difficulty and split into ``n_splits``
    contiguous buckets.  Each RAF worker therefore specialises on a
    different difficulty band.  The ordering of the resulting partitions
    is finalised by :meth:`BasePartitioner._finalize`, which promotes the
    most class-balanced partition to index 0 (required by
    :class:`RAFEnsembler`'s ``main_target`` contract).

    Args:
        n_splits: number of difficulty buckets.
        task_type: ``'classification'`` or ``'regression'``. If ``None``
            the type is inferred from ``target`` at partition time.
        weak_model: optional pre-configured estimator implementing
            ``fit`` / ``predict`` (and ``predict_proba`` for
            classification). Defaults to a shallow
            ``DecisionTreeClassifier`` / ``DecisionTreeRegressor`` --
            both tree-based models are scale-invariant, so no feature
            scaling is applied by this partitioner.
        cv: number of CV folds for ``cross_val_predict``.
        random_state: seed for reproducibility of the default weak model.
    """

    def __init__(self,
                 n_splits: int = 5,
                 task_type: Optional[str] = None,
                 weak_model=None,
                 cv: int = 3,
                 random_state: int = 42):
        super().__init__(n_splits)
        self.task_type = task_type
        self.weak_model = weak_model
        self.cv = cv
        self.random_state = random_state

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        x = self._flatten_features(features)
        y = np.asarray(target).ravel()

        task_type = self._infer_task_type(y, self.task_type)
        difficulty = self._compute_difficulty(x, y, task_type)

        n_samples = len(y)
        effective_n_splits = min(self.n_splits, n_samples)
        if effective_n_splits < self.n_splits:
            self.logger.warning(
                f'n_splits={self.n_splits} exceeds n_samples={n_samples}. '
                f'Reduced to {effective_n_splits}')

        order = np.argsort(difficulty)
        feature_buckets = np.array_split(order, effective_n_splits)
        features_splits = [features[idx] for idx in feature_buckets]
        target_splits = [target[idx] for idx in feature_buckets]

        self.logger.info(
            f'DifficultyPartitioner: difficulty mean={difficulty.mean():.4f}, '
            f'std={difficulty.std():.4f}, '
            f'min={difficulty.min():.4f}, max={difficulty.max():.4f}')
        return self._finalize(features_splits, target_splits)

    def _default_weak_model(self, task_type: str):
        if task_type == 'classification':
            return DecisionTreeClassifier(
                max_depth=5, random_state=self.random_state)
        return DecisionTreeRegressor(
            max_depth=5, random_state=self.random_state)

    def _compute_difficulty(self,
                            x: np.ndarray,
                            y: np.ndarray,
                            task_type: str) -> np.ndarray:
        """Return a 1-D difficulty score per sample.

        The CV path and the "fit on the full set" fallback share the same
        post-processing: we always end up with either (``proba``,
        ``classes``) for classification with ``predict_proba``, or
        ``y_pred`` otherwise. That pair then maps to a difficulty score
        the same way in both branches.
        """
        model = self.weak_model if self.weak_model is not None \
            else self._default_weak_model(task_type)
        has_proba = task_type == 'classification' and hasattr(model, 'predict_proba')
        effective_cv = self._effective_cv(y, task_type)

        proba = y_pred = classes = None
        try:
            if effective_cv < 2:
                raise ValueError(f'effective_cv={effective_cv} is too small for CV')
            if has_proba:
                proba = cross_val_predict(
                    model, x, y, cv=effective_cv, method='predict_proba')
                classes = np.unique(y)
            else:
                y_pred = cross_val_predict(model, x, y, cv=effective_cv)
        except Exception as err:  # noqa: BLE001
            self.logger.warning(
                f'cross_val_predict unavailable ({err!r}); '
                f'fitting weak model on the full set')
            model.fit(x, y)
            if has_proba:
                proba = model.predict_proba(x)
                classes = model.classes_
            else:
                y_pred = model.predict(x)

        if has_proba:
            return self._proba_to_difficulty(proba, y, classes)
        if task_type == 'classification':
            return (y_pred != y).astype(float)
        return np.abs(y.astype(float) - y_pred.astype(float))

    def _effective_cv(self, y: np.ndarray, task_type: str) -> int:
        """CV fold count actually usable for this dataset.

        For classification ``StratifiedKFold`` requires every class to
        have at least ``cv`` samples; for regression only the sample
        count matters.
        """
        effective_cv = min(self.cv, len(y))
        if task_type == 'classification':
            _, counts = np.unique(y, return_counts=True)
            effective_cv = min(effective_cv, int(counts.min()))
        return effective_cv

    @staticmethod
    def _proba_to_difficulty(proba: np.ndarray,
                             y: np.ndarray,
                             classes: np.ndarray) -> np.ndarray:
        """Difficulty = 1 - p(true_class); larger means more uncertain.

        ``classes`` is assumed to be sorted (it comes from ``np.unique``
        or sklearn's ``classes_``), so ``searchsorted`` maps each label
        to its column in ``proba`` in one vectorised call.
        """
        cols = np.searchsorted(classes, y)
        true_proba = proba[np.arange(len(y)), cols]
        return 1.0 - true_proba


class StratifiedPartitioner(BasePartitioner):
    """Partition data with stratified sampling to preserve target distribution (T3).

    For classification, :class:`sklearn.model_selection.StratifiedKFold`
    is used -- every partition keeps the original class proportions
    (within the usual integer-rounding slack).  For regression, the
    target is discretised into ``regression_bins`` quantile bins and the
    same stratified split is applied to those bins, so each partition
    covers the full target range.

    This is the cheapest partitioner that still respects the class
    balance invariant enforced by :meth:`BasePartitioner._finalize`,
    and it is the natural baseline for the cluster-based strategies.

    Args:
        n_splits: number of partitions (== number of stratified folds).
        task_type: ``'classification'`` or ``'regression'`` (auto-inferred
            if ``None``).
        regression_bins: number of quantile bins used for regression
            stratification. Ignored for classification.
        random_state: seed for reproducibility.
        shuffle: whether to shuffle before splitting.
    """

    def __init__(self,
                 n_splits: int = 5,
                 task_type: Optional[str] = None,
                 regression_bins: int = 10,
                 random_state: int = 42,
                 shuffle: bool = True):
        super().__init__(n_splits)
        self.task_type = task_type
        self.regression_bins = regression_bins
        self.random_state = random_state
        self.shuffle = shuffle

    def partition(self,
                  features: np.ndarray,
                  target: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        y = np.asarray(target).ravel()
        n_samples = len(y)

        effective_n_splits = min(self.n_splits, n_samples)
        if effective_n_splits < 2:
            # a single split == the whole dataset; defer to the finaliser
            return self._finalize([features], [target])
        if effective_n_splits < self.n_splits:
            self.logger.warning(
                f'n_splits={self.n_splits} exceeds n_samples={n_samples}. '
                f'Reduced to {effective_n_splits}')

        task_type = self._infer_task_type(y, self.task_type)
        strata = self._build_strata(y, task_type)
        effective_n_splits = self._clamp_to_min_class_size(
            strata, effective_n_splits)

        try:
            skf = StratifiedKFold(
                n_splits=effective_n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None)
            bucket_indices = [test_idx for _, test_idx in skf.split(
                np.zeros(n_samples), strata)]
        except ValueError as err:
            self.logger.warning(
                f'StratifiedKFold failed ({err!r}); falling back to shuffled '
                f'sequential splits')
            rng = np.random.default_rng(self.random_state)
            order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            bucket_indices = np.array_split(order, effective_n_splits)

        features_splits = [features[idx] for idx in bucket_indices]
        target_splits = [target[idx] for idx in bucket_indices]
        return self._finalize(features_splits, target_splits)

    def _build_strata(self, y: np.ndarray, task_type: str) -> np.ndarray:
        if task_type == 'classification':
            return y
        n_bins = max(2, min(self.regression_bins, len(np.unique(y))))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.quantile(y, quantiles)
        # np.digitize returns integers in [0, n_bins]
        return np.digitize(y, edges)

    @staticmethod
    def _clamp_to_min_class_size(strata: np.ndarray, n_splits: int) -> int:
        _, counts = np.unique(strata, return_counts=True)
        min_count = int(counts.min()) if counts.size else 1
        return max(2, min(n_splits, min_count))


class FeatureSpacePartitioner:
    """Factory that instantiates a partitioner by name.

    Supported methods:

    * Tabular:

      * ``'sequential'`` -- equal-sized sequential chunks (default).
      * ``'kmeans'`` -- K-Means clustering in feature space.
      * ``'dbscan'`` -- DBSCAN density-based clustering.
      * ``'difficulty'`` -- T2: bucket samples by a weak model's
        per-sample error / uncertainty.
      * ``'stratified'`` -- T3: stratified splits preserving target
        distribution.

    * Time series (registered lazily from
      :mod:`.ts_partitioner` to avoid circular imports):

      * ``'temporal'`` -- TS1: contiguous blocks along the time axis.
      * ``'ts_feature_clustering'`` -- TS2: cluster series by
        hand-crafted statistical / spectral descriptors.
      * ``'ts_difficulty'`` -- TS3: bucket samples by residuals from a
        teacher model fit on the whole set (no CV, TS-safe).
    """

    PARTITIONER_REGISTRY = {
        'sequential': SequentialPartitioner,
        'kmeans': KMeansPartitioner,
        'dbscan': DBSCANPartitioner,
        'difficulty': DifficultyPartitioner,
        'stratified': StratifiedPartitioner,
    }

    @classmethod
    def create(cls,
               method: str = 'sequential',
               n_splits: int = 5,
               params: Optional[dict] = None) -> BasePartitioner:
        params = dict(params or {})
        method = method.lower()
        if method not in cls.PARTITIONER_REGISTRY:
            raise ValueError(
                f"Unknown partitioning method '{method}'. "
                f"Available: {list(cls.PARTITIONER_REGISTRY.keys())}")
        partitioner_cls = cls.PARTITIONER_REGISTRY[method]
        effective_n_splits = params.pop('n_splits', n_splits)
        # keep only kwargs accepted by the concrete partitioner so that a
        # shared ``partitioning_params`` dict can be reused across methods
        accepted = set(inspect.signature(partitioner_cls.__init__).parameters)
        kwargs = {k: v for k, v in params.items() if k in accepted}
        return partitioner_cls(n_splits=effective_n_splits, **kwargs)
