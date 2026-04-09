import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

K = 2
REGIME_NAMES = {
    0: "Alta",
    1: "Baixa",
}


def fit_kmeans(X: np.ndarray, seed: int = 42) -> KMeans:
    km = KMeans(n_clusters=K, random_state=seed, n_init=20, max_iter=500)
    km.fit(X)
    return km


def map_clusters_to_regimes(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    tmp = df[["log_return"]].copy()
    tmp["cluster"] = labels
    stats = tmp.groupby("cluster").agg(ret=("log_return", "mean"))
    mapping = {stats["ret"].idxmax(): 0, stats["ret"].idxmin(): 1}
    return np.array([mapping[l] for l in labels])


def run_clustering(datasets: dict, df: pd.DataFrame) -> dict:
    results = {}
    for name, X in datasets.items():
        km = fit_kmeans(X)
        raw_labels = km.labels_
        regimes = map_clusters_to_regimes(df, raw_labels)
        results[name] = {"kmeans": km, "raw_labels": raw_labels, "regimes": regimes}
        print(f"[{name}] clusters: {np.bincount(raw_labels)}")
    return results
