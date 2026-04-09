import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

from clustering import K, REGIME_NAMES

COLORS = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6", "#e67e22"]


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": silhouette_score(X, labels, sample_size=5000, random_state=42),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def temporal_stability(regimes: np.ndarray, window: int = 48) -> float:
    total = len(regimes) // window
    if total == 0:
        return 0.0
    stable = sum(
        1 for i in range(total)
        if np.bincount(regimes[i * window:(i + 1) * window], minlength=K).max() / window > 0.5
    )
    return stable / total


def evaluate_all(datasets: dict, results: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in datasets:
        X = datasets[name]
        regimes = results[name]["regimes"]
        m = compute_metrics(X, regimes)
        m["stability"] = temporal_stability(regimes)
        m["dataset"] = name
        for i, s in enumerate(np.bincount(regimes, minlength=K)):
            m[f"size_r{i}"] = s
        rows.append(m)
    return pd.DataFrame(rows).set_index("dataset")


def plot_regimes_over_time(df: pd.DataFrame, results: dict, out_dir: str = "."):
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for ax, (name, res) in zip(axes, results.items()):
        regimes = res["regimes"]
        ax.plot(df["timestamp"], df["close"], color="black", lw=0.5, alpha=0.6)
        for r in range(K):
            mask = regimes == r
            ax.scatter(df["timestamp"][mask], df["close"][mask],
                       c=COLORS[r], s=2, label=REGIME_NAMES[r], zorder=3)
        ax.set_title(f"Regimes — {name.upper()}", fontsize=10)
        ax.set_ylabel("BTC/USDT")
        if name == list(results.keys())[-1]:
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=COLORS[i], markersize=8,
                                  label=REGIME_NAMES[i]) for i in range(K)]
            ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=K)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/regimes_over_time.png", dpi=150)
    plt.close()


def plot_scatter_2d(datasets: dict, results: dict, out_dir: str = "."):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, results.items()):
        X = datasets[name]
        regimes = res["regimes"]
        coords = PCA(n_components=2, random_state=42).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
        for r in range(K):
            mask = regimes == r
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=COLORS[r], s=3, alpha=0.5, label=REGIME_NAMES[r])
        ax.set_title(f"PCA 2D — {name.upper()}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLORS[i], markersize=8,
                          label=REGIME_NAMES[i]) for i in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=K, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{out_dir}/scatter_2d.png", dpi=150)
    plt.close()


def plot_return_vs_volatility(df: pd.DataFrame, results: dict, out_dir: str = "."):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, results.items()):
        regimes = res["regimes"]
        for r in range(K):
            mask = regimes == r
            ax.scatter(df["log_return"][mask], df["volatility"][mask],
                       c=COLORS[r], s=3, alpha=0.4, label=REGIME_NAMES[r])
        ax.set_title(f"Retorno vs Volatilidade — {name.upper()}", fontsize=10)
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Volatilidade")
        ax.set_xlim(-0.05, 0.05)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLORS[i], markersize=8,
                          label=REGIME_NAMES[i]) for i in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=K, fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"{out_dir}/return_vs_volatility.png", dpi=150)
    plt.close()


def plot_correlation_heatmaps(df: pd.DataFrame, results: dict,
                               feature_cols: list, out_dir: str = "."):
    regimes = results["raw"]["regimes"]
    ncols = min(K, 3)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for r in range(K):
        mask = regimes == r
        sub = df.loc[mask, feature_cols]
        if len(sub) < 5:
            axes[r].set_visible(False)
            continue
        sns.heatmap(sub.corr(), ax=axes[r], cmap="RdBu_r", center=0,
                    annot=True, fmt=".2f", linewidths=0.3,
                    annot_kws={"size": 6}, cbar=False)
        axes[r].set_title(f"Regime {r}: {REGIME_NAMES[r]}", fontsize=9)
        axes[r].tick_params(labelsize=7)
    for r in range(K, len(axes)):
        axes[r].set_visible(False)
    plt.suptitle("Correlação por Regime (Dataset Bruto)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmaps.png", dpi=150)
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, out_dir: str = "."):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cols = ["silhouette", "calinski_harabasz", "davies_bouldin", "stability"]
    titles = ["Silhouette ↑", "Calinski-Harabasz ↑", "Davies-Bouldin ↓", "Estabilidade ↑"]
    for ax, col, title in zip(axes, cols, titles):
        vals = metrics_df[col]
        bars = ax.bar(range(len(vals)), vals.values,
                      color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=20, fontsize=8)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/metrics_comparison.png", dpi=150)
    plt.close()


def run_evaluation(datasets: dict, results: dict, df: pd.DataFrame,
                   feature_cols: list, out_dir: str = ".") -> pd.DataFrame:
    metrics = evaluate_all(datasets, results, df)
    print("\n=== Métricas ===")
    print(metrics[["silhouette", "calinski_harabasz", "davies_bouldin", "stability"]].to_string())

    plot_regimes_over_time(df, results, out_dir)
    plot_scatter_2d(datasets, results, out_dir)
    plot_return_vs_volatility(df, results, out_dir)
    plot_correlation_heatmaps(df, results, feature_cols, out_dir)
    plot_metrics_comparison(metrics, out_dir)
    print(f"\nGráficos salvos em '{out_dir}/'")
    return metrics
