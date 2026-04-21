import numpy as np
from mdpocketclustering.clustering import cluster_states


def test_kmeans_runs():

    X = np.random.rand(100, 1)

    meta = [{"replicate": 1, "frame": i, "binding_energy": -10} for i in range(100)]

    df, model = cluster_states(X, meta, n_clusters=3)

    assert "cluster" in df.columns
    assert df["cluster"].nunique() <= 3
