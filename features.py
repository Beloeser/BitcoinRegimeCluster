import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(20).std()
    df["amplitude"] = (df["high"] - df["low"]) / df["close"]
    df["volume_norm"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    df["momentum"] = df["close"] / df["close"].shift(10) - 1
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma_ratio"] = df["ma20"] / df["ma50"] - 1

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    return df.dropna().reset_index(drop=True)


FEATURE_COLS = ["log_return", "volatility", "amplitude", "volume_norm",
                "momentum", "ma_ratio", "rsi"]


def build_datasets(df: pd.DataFrame, n_components: int = 3) -> dict:
    X = df[FEATURE_COLS].values
    Xs = StandardScaler().fit_transform(X)

    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(Xs)
    X_ica = FastICA(n_components=n_components, random_state=42, max_iter=1000).fit_transform(Xs)
    X_pca_ica = FastICA(n_components=n_components, random_state=42, max_iter=1000).fit_transform(X_pca)

    return {
        "raw": Xs,
        "pca": X_pca,
        "ica": X_ica,
        "pca_ica": X_pca_ica,
    }, PCA(n_components=n_components, random_state=42).fit(Xs)
