# Bitcoin Regime Clustering

Automatic identification of market regimes in BTC/USDT using 30-minute candle data from the Binance public API.

---

## Structure

```
├── data_collection.py   # paginated data collection from Binance API
├── features.py          # feature engineering and dimensionality reduction
├── clustering.py        # KMeans + regime mapping
├── evaluation.py        # metrics and charts
├── main.py              # main pipeline
├── requirements.txt
└── outputs/             # generated charts and CSVs
```

---

## How to run

```bash
pip install -r requirements.txt
python3 main.py
```

On the first run, data is downloaded from Binance and saved to `btc_30m.csv`. Subsequent runs reuse the cached file. To force a fresh download, delete `btc_30m.csv`.

---

## Data

- **Pair:** BTCUSDT
- **Interval:** 30 minutes
- **Period:** August 2017 → April 2026
- **Total candles:** 151,227
- **Source:** `GET https://api.binance.com/api/v3/klines`

---

## Features

| Feature | Description |
|---|---|
| `log_return` | ln(close_t / close_t-1) |
| `volatility` | rolling(20) std of log_return |
| `amplitude` | (high - low) / close |
| `volume_norm` | rolling(20) z-score of volume |
| `momentum` | close_t / close_t-10 - 1 |
| `ma_ratio` | MA20 / MA50 - 1 |
| `rsi` | RSI(14) |

---

## Regimes (K=2)

| Regime | Name | N | Mean Return | Ret. Std | Mean Vol. |
|---|---|---|---|---|---|
| 0 | Bull | 124,447 | +0.000633 | 0.003718 | 0.003646 |
| 1 | Bear | 26,780 | -0.002839 | 0.010188 | 0.007596 |

The **Bear** regime concentrates the highest volatility and negative returns. The **Bull** regime covers most of the time with positive returns and contained volatility.

---

## Compared approaches

Four dataset versions are tested in parallel:

| Approach | Description |
|---|---|
| `raw` | 7 standardized features used directly |
| `pca` | Reduced to 3 principal components |
| `ica` | Reduced to 3 independent components |
| `pca_ica` | ICA applied on top of PCA components |

---

## Results

| Dataset | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Stability ↑ |
|---|---|---|---|---|
| raw | 0.3637 | 29314.40 | 1.7382 | 0.9924 |
| **pca** | **0.4460** | **46270.42** | **1.3327** | **0.9943** |
| ica | 0.3724 | 40234.92 | 1.4979 | 0.9908 |
| pca_ica | 0.3724 | 40234.92 | 1.4979 | 0.9908 |

PCA wins across all metrics. Stability above 0.99 means each 24h window is dominated by a single regime in over 99% of cases.

---

## Generated charts

| File | Description |
|---|---|
| `regimes_over_time.png` | price colored by regime across all 4 approaches |
| `scatter_2d.png` | PCA 2D projection of clusters |
| `return_vs_volatility.png` | return vs volatility colored by regime |
| `correlation_heatmaps.png` | feature correlation heatmap per regime |
| `metrics_comparison.png` | metrics comparison across approaches |
