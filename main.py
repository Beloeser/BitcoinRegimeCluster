import os
import pandas as pd

from data_collection import collect_data
from features import add_features, build_datasets, FEATURE_COLS
from clustering import run_clustering
from evaluation import run_evaluation

OUT_DIR = "outputs"
DATA_FILE = "btc_30m.csv"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print(f"Carregando dados de '{DATA_FILE}'...")
        df_raw = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    else:
        df_raw = collect_data(years_back=5)
        df_raw.to_csv(DATA_FILE, index=False)

    df = add_features(df_raw)
    print(f"Dataset: {len(df):,} candles, {df['timestamp'].min()} → {df['timestamp'].max()}")

    datasets, _ = build_datasets(df, n_components=3)
    results = run_clustering(datasets, df)

    metrics = run_evaluation(datasets, results, df, FEATURE_COLS, out_dir=OUT_DIR)
    metrics.to_csv(f"{OUT_DIR}/metrics.csv")

    print("\n✓ Concluído. Arquivos em:", OUT_DIR)
    print(metrics[["silhouette", "calinski_harabasz", "davies_bouldin", "stability"]].to_string())


if __name__ == "__main__":
    main()
