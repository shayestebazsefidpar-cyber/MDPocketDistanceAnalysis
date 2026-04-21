import pandas as pd
from mdpocketclustering.analysis import cluster_summary, occupancy


def test_summary():

    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "binding_energy": [-10, -20, -30, -40],
        "replicate": [1, 1, 2, 2]
    })

    summary = cluster_summary(df, None)

    assert "energy_mean" in summary.columns


def test_occupancy():

    df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "replicate": [1, 1, 2, 2]
    })

    occ = occupancy(df)

    assert abs(occ.sum().mean() - 100) < 1e-6
