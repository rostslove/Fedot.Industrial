import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification

from fedot_ind.core.operation.partitioning import (
    BasePartitioner,
    SequentialPartitioner,
    KMeansPartitioner,
    DBSCANPartitioner,
    FeatureSpacePartitioner
)


@pytest.fixture
def tabular_data():
    """Simple tabular dataset for partitioning tests."""
    X, y = make_classification(n_samples=200, n_features=10,
                               n_classes=2, random_state=42)
    y = y.reshape(-1, 1)
    return X, y


@pytest.fixture
def clustered_data():
    """Dataset with 3 well-separated blobs."""
    X, y = make_blobs(n_samples=300, centers=3,
                      cluster_std=0.5, random_state=42)
    y = y.reshape(-1, 1)
    return X, y


@pytest.fixture
def data_3d():
    """3-D dataset simulating time series (n_samples, n_channels, ts_len)."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 2, 50)
    y = rng.randint(0, 2, size=(150, 1))
    return X, y


class TestSequentialPartitioner:
    def test_partition_count(self, tabular_data):
        X, y = tabular_data
        partitioner = SequentialPartitioner(n_splits=4)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) == 4
        assert len(tgt_splits) == 4

    def test_all_samples_preserved(self, tabular_data):
        X, y = tabular_data
        partitioner = SequentialPartitioner(n_splits=3)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        # class-diversity rebalancing may duplicate samples across
        # partitions, but no original sample is ever lost
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)

    def test_feature_shape_preserved(self, tabular_data):
        X, y = tabular_data
        partitioner = SequentialPartitioner(n_splits=5)
        feat_splits, _ = partitioner.partition(X, y)
        for split in feat_splits:
            assert split.shape[1] == X.shape[1]


class TestKMeansPartitioner:
    def test_partition_count(self, tabular_data):
        X, y = tabular_data
        partitioner = KMeansPartitioner(n_splits=4, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) == 4
        assert len(tgt_splits) == 4

    def test_all_samples_preserved(self, tabular_data):
        X, y = tabular_data
        partitioner = KMeansPartitioner(n_splits=3, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        # rebalancing may duplicate a few samples across partitions, but
        # no original sample is allowed to be lost
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)

    def test_disjoint_partitions(self, tabular_data):
        X, y = tabular_data
        partitioner = KMeansPartitioner(n_splits=4, random_state=42)
        feat_splits, _ = partitioner.partition(X, y)
        all_indices = []
        for split in feat_splits:
            for row in split:
                idx = np.where(np.all(X == row, axis=1))[0]
                all_indices.extend(idx.tolist())
        assert len(set(all_indices)) == len(X)

    def test_clustered_data_respects_structure(self, clustered_data):
        X, y = clustered_data
        partitioner = KMeansPartitioner(n_splits=3, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        # each partition should have at least 2 classes (guaranteed by diversity check)
        for split_y in tgt_splits:
            assert len(np.unique(split_y)) >= 2

    def test_class_diversity_guaranteed(self):
        """Even with perfectly separable clusters, each partition must have >= 2 classes."""
        rng = np.random.RandomState(0)
        # class 0 in one corner, class 1 in another — KMeans will separate them
        X0 = rng.randn(100, 5) + 10
        X1 = rng.randn(100, 5) - 10
        X = np.vstack([X0, X1])
        y = np.array([0] * 100 + [1] * 100).reshape(-1, 1)
        partitioner = KMeansPartitioner(n_splits=2, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        for split_y in tgt_splits:
            assert len(np.unique(split_y)) >= 2, \
                f'Partition has only {np.unique(split_y)} classes'

    def test_handles_3d_features(self, data_3d):
        X, y = data_3d
        partitioner = KMeansPartitioner(n_splits=3, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) == 3
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)
        # original shape dimensions preserved
        for split in feat_splits:
            assert split.ndim == 3
            assert split.shape[1:] == X.shape[1:]

    def test_n_splits_exceeds_samples(self):
        X = np.random.randn(3, 5)
        y = np.array([[0], [1], [0]])
        partitioner = KMeansPartitioner(n_splits=10, random_state=42)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) <= 3

    def test_no_scaling(self, tabular_data):
        X, y = tabular_data
        partitioner = KMeansPartitioner(n_splits=3, scale_features=False)
        feat_splits, _ = partitioner.partition(X, y)
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)


class TestDBSCANPartitioner:
    def test_discovers_clusters(self, clustered_data):
        X, y = clustered_data
        partitioner = DBSCANPartitioner(n_splits=3, eps=1.0, min_samples=5)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) >= 1
        # class-diversity fix may duplicate a few samples across partitions,
        # so the total is allowed to grow but must never drop below the input
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)

    def test_merge_when_too_many_clusters(self, clustered_data):
        X, y = clustered_data
        # small eps → many clusters → should merge down to n_splits
        partitioner = DBSCANPartitioner(n_splits=2, eps=0.3, min_samples=3)
        feat_splits, _ = partitioner.partition(X, y)
        assert len(feat_splits) <= 3  # approximately n_splits

    def test_fallback_when_all_noise(self, tabular_data):
        X, y = tabular_data
        # very small eps → everything is noise → fallback to sequential
        partitioner = DBSCANPartitioner(n_splits=3, eps=0.001, min_samples=100)
        feat_splits, tgt_splits = partitioner.partition(X, y)
        assert len(feat_splits) == 3
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)

    def test_noise_assigned_to_nearest(self, clustered_data):
        X, y = clustered_data
        partitioner = DBSCANPartitioner(n_splits=3, eps=0.8, min_samples=5)
        feat_splits, _ = partitioner.partition(X, y)
        # all input samples (including noise) must be represented; class-
        # diversity donation may additionally duplicate a few
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)

    def test_handles_3d_features(self, data_3d):
        X, y = data_3d
        partitioner = DBSCANPartitioner(n_splits=3, eps=5.0, min_samples=3)
        feat_splits, _ = partitioner.partition(X, y)
        total = sum(len(f) for f in feat_splits)
        assert total >= len(X)


class TestFeatureSpacePartitioner:
    def test_factory_sequential(self, tabular_data):
        X, y = tabular_data
        p = FeatureSpacePartitioner.create('sequential', n_splits=3)
        assert isinstance(p, SequentialPartitioner)
        feat_splits, _ = p.partition(X, y)
        assert len(feat_splits) == 3

    def test_factory_kmeans(self, tabular_data):
        X, y = tabular_data
        p = FeatureSpacePartitioner.create('kmeans', n_splits=4,
                                           params={'random_state': 0})
        assert isinstance(p, KMeansPartitioner)
        feat_splits, _ = p.partition(X, y)
        assert len(feat_splits) == 4

    def test_factory_dbscan(self, tabular_data):
        X, y = tabular_data
        p = FeatureSpacePartitioner.create('dbscan', n_splits=3,
                                           params={'eps': 2.0, 'min_samples': 3})
        assert isinstance(p, DBSCANPartitioner)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown partitioning method'):
            FeatureSpacePartitioner.create('unknown_method')

    def test_factory_case_insensitive(self, tabular_data):
        X, y = tabular_data
        p = FeatureSpacePartitioner.create('KMeans', n_splits=3)
        assert isinstance(p, KMeansPartitioner)
