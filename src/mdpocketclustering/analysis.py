import pandas as pd


def aggregate_replicates(
    registry,
    value: str,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean/std/count across replicates for any numeric field.
    """

    df = registry.to_dataframe()

    if value not in df.columns:
        raise ValueError(f"Column '{value}' not found in registry dataframe")

    if by is None:
        by = ["mutation"]

    missing = [c for c in by if c not in df.columns]
    if missing:
        raise ValueError(f"Grouping columns not found: {missing}")

    df = df.dropna(subset=[value]).copy()

    return (
        df.groupby(by)[value].agg(mean="mean", std="std", count="count").reset_index()
    )


def cluster_summary(df, X=None):
    return (
        df.groupby("cluster")
        .agg(
            energy_mean=("binding_energy", "mean"),
            energy_std=("binding_energy", "std"),
            count=("cluster", "size"),
        )
        .reset_index()
    )


def occupancy(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["replicate"], df["cluster"], normalize="index") * 100
