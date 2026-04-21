from fedot_ind.core.operation.partitioning.cluster_partitioner import (
    BasePartitioner,
    SequentialPartitioner,
    KMeansPartitioner,
    DBSCANPartitioner,
    DifficultyPartitioner,
    StratifiedPartitioner,
    FeatureSpacePartitioner
)
from fedot_ind.core.operation.partitioning.ts_partitioner import (
    TemporalSplitPartitioner,
    TSFeatureClusteringPartitioner,
    TSModelDifficultyPartitioner,
)

# Register TS partitioners on the shared factory. Doing it here (rather
# than at the bottom of cluster_partitioner.py) keeps cluster_partitioner
# free of imports from ts_partitioner, which inherits from it.
FeatureSpacePartitioner.PARTITIONER_REGISTRY.update({
    'temporal': TemporalSplitPartitioner,
    'ts_feature_clustering': TSFeatureClusteringPartitioner,
    'ts_difficulty': TSModelDifficultyPartitioner,
})

__all__ = [
    'BasePartitioner',
    'SequentialPartitioner',
    'KMeansPartitioner',
    'DBSCANPartitioner',
    'DifficultyPartitioner',
    'StratifiedPartitioner',
    'TemporalSplitPartitioner',
    'TSFeatureClusteringPartitioner',
    'TSModelDifficultyPartitioner',
    'FeatureSpacePartitioner',
]
