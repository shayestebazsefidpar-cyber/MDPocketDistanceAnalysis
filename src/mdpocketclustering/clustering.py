import pandas as pd
from sklearn.cluster import KMeans


def cluster_states(X, meta, n_clusters=3):
    """
    Run KMeans clustering on feature matrix X and attach labels to meta.

    Parameters
    ----------
    X : array-like (n_samples, n_features)
    meta : list[dict]
    n_clusters : int

    Returns
    -------
    df : pd.DataFrame
    model : KMeans
    """

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X)

    df = pd.DataFrame(meta)
    df["cluster"] = labels

    return df, km
