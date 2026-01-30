import pandas as pd

INPUT_CSV = "1m.csv"   # your raw 1m file
OUTPUT_PREFIX = "output_"  # prefix for output files

# -----------------------------
# 1. Load raw 1m data
# -----------------------------
df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])

df = df.sort_values("timestamp")
df = df.set_index("timestamp")

# -----------------------------
# 2. Resample function (ML-safe)
# -----------------------------
def resample_ohlcv(df, timeframe):
    out = pd.DataFrame()
    out["open"] = df["open"].resample(
        timeframe, label="left", closed="left"
    ).first()
    out["high"] = df["high"].resample(
        timeframe, label="left", closed="left"
    ).max()
    out["low"] = df["low"].resample(
        timeframe, label="left", closed="left"
    ).min()
    out["close"] = df["close"].resample(
        timeframe, label="left", closed="left"
    ).last()
    out["volume"] = df["volume"].resample(
        timeframe, label="left", closed="left"
    ).sum()
    out["amount"] = df["amount"].resample(
        timeframe, label="left", closed="left"
    ).sum()

    # drop incomplete candles
    out = out.dropna()

    return out

# -----------------------------
# 3. Generate timeframes
# -----------------------------
timeframes = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h"
}

for name, tf in timeframes.items():
    resampled = resample_ohlcv(df, tf)
    resampled.to_csv(f"{OUTPUT_PREFIX}{name}.csv")
    print(f"Saved {len(resampled)} rows → {OUTPUT_PREFIX}{name}.csv")
