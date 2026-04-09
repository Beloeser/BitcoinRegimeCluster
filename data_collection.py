import requests
import pandas as pd
import time

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "30m"
LIMIT = 1000


def fetch_klines(start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT,
        "startTime": start_ms,
        "endTime": end_ms,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_data(years_back: int = 5) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - years_back * 365 * 24 * 3600 * 1000
    interval_ms = LIMIT * 30 * 60 * 1000

    all_rows = []
    current = start_ms
    print(f"Coletando {SYMBOL} {INTERVAL}...")

    while current < end_ms:
        batch_end = min(current + interval_ms, end_ms)
        rows = fetch_klines(current, batch_end)
        if not rows:
            current = batch_end + 1
            continue
        all_rows.extend(rows)
        current = rows[-1][0] + 1
        print(f"  {len(all_rows)} candles...", end="\r")
        time.sleep(0.2)

    print(f"\nTotal: {len(all_rows)} candles")

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df
