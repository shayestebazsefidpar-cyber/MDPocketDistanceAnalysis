import pandas as pd
from sklearn.cluster import KMeans


def cluster_states(X, meta, n_clusters=3):

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X)

    df = pd.DataFrame(meta)
    df["cluster"] = labels

    return df, km
