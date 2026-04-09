# Bitcoin Regime Clustering

Identificação automática de regimes de mercado no BTC/USDT usando dados de 30 minutos da API pública da Binance.

---

## Estrutura

```
├── data_collection.py   # coleta paginada da API Binance
├── features.py          # feature engineering e redução dimensional
├── clustering.py        # KMeans + mapeamento de regimes
├── evaluation.py        # métricas e gráficos
├── main.py              # pipeline principal
├── requirements.txt
└── outputs/             # gráficos e CSVs gerados
```

---

## Como rodar

```bash
pip install -r requirements.txt
python3 main.py
```

Na primeira execução os dados são baixados da Binance e salvos em `btc_30m.csv`. Nas execuções seguintes o CSV é reutilizado. Para forçar nova coleta, apague o `btc_30m.csv`.

---

## Dados

- **Par:** BTCUSDT
- **Intervalo:** 30 minutos
- **Período:** agosto/2017 → abril/2026
- **Total de candles:** 151.227
- **Fonte:** `GET https://api.binance.com/api/v3/klines`

---

## Features

| Feature | Descrição |
|---|---|
| `log_return` | ln(close_t / close_t-1) |
| `volatility` | desvio padrão rolling(20) do log_return |
| `amplitude` | (high - low) / close |
| `volume_norm` | z-score rolling(20) do volume |
| `momentum` | close_t / close_t-10 - 1 |
| `ma_ratio` | MA20 / MA50 - 1 |
| `rsi` | RSI(14) |

---

## Regimes (K=2)

| Regime | Nome | N | Retorno Médio | Ret. Std | Vol. Média |
|---|---|---|---|---|---|
| 0 | Alta | 124.447 | +0.000633 | 0.003718 | 0.003646 |
| 1 | Baixa | 26.780 | -0.002839 | 0.010188 | 0.007596 |

O regime de **Baixa** concentra os movimentos de maior volatilidade e retornos negativos. O regime de **Alta** cobre a maior parte do tempo com retornos positivos e volatilidade contida.

---

## Abordagens comparadas

Quatro versões do dataset são testadas em paralelo:

| Abordagem | Descrição |
|---|---|
| `raw` | 7 features padronizadas diretamente |
| `pca` | Redução para 3 componentes principais |
| `ica` | Redução para 3 componentes independentes |
| `pca_ica` | ICA aplicado sobre os componentes do PCA |

---

## Resultados

| Dataset | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Estabilidade ↑ |
|---|---|---|---|---|
| raw | 0.3637 | 29314.40 | 1.7382 | 0.9924 |
| **pca** | **0.4460** | **46270.42** | **1.3327** | **0.9943** |
| ica | 0.3724 | 40234.92 | 1.4979 | 0.9908 |
| pca_ica | 0.3724 | 40234.92 | 1.4979 | 0.9908 |

PCA vence em todas as métricas. A estabilidade acima de 0.99 indica que cada janela de 24h é dominada por um único regime em mais de 99% dos casos.

---

## Gráficos gerados

| Arquivo | Descrição |
|---|---|
| `regimes_over_time.png` | preço colorido por regime nas 4 abordagens |
| `scatter_2d.png` | projeção PCA 2D dos clusters |
| `return_vs_volatility.png` | retorno vs volatilidade por regime |
| `correlation_heatmaps.png` | correlação entre features por regime |
| `metrics_comparison.png` | comparação das métricas entre abordagens |
# BitcoinRegimeCluster
