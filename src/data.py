"""
src/data.py
Data loading, merging, and resampling pipeline.

Public API
----------
load_kaggle_csv(path)           -> pd.DataFrame  (minute-level, 2019-present)
fetch_binance_klines(start, end) -> pd.DataFrame  (minute-level)
merge_sources(df1, df2)         -> pd.DataFrame  (deduplicated, sorted)
resample_ohlcv(df, timeframes)  -> dict[str, pd.DataFrame]
save_processed(frames, out_dir) -> None
run_pipeline(kaggle_path, out_dir, binance_start, binance_end) -> dict
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAGGLE_START = "2019-01-01"
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
# Binance returns at most 1000 candles per request
BINANCE_LIMIT = 1000

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
}

# Aggregation rules for resampling OHLCV data
RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


# ---------------------------------------------------------------------------
# 1. Load Kaggle CSV
# ---------------------------------------------------------------------------

def load_kaggle_csv(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle Bitcoin historical CSV and return a clean minute-level
    DataFrame starting from 2019-01-01.

    The Kaggle dataset (bitstampUSD or similar) typically has columns:
        Timestamp, Open, High, Low, Close, Volume_(BTC), Volume_(Currency), Weighted_Price
    or:
        unix, date, symbol, open, high, low, close, Volume BTC, Volume USD

    The function handles both formats and normalises to a common schema.

    Parameters
    ----------
    path : str or Path
        Path to the raw CSV file inside data/raw/.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume — indexed by a UTC DatetimeIndex
        named 'timestamp'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found at {path}. "
            "Download it from Kaggle and place it in data/raw/."
        )

    print(f"[data] Loading Kaggle CSV from {path} …")
    df = pd.read_csv(path)

    # ---- normalise column names ----
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ---- parse timestamp ----
    if "timestamp" in df.columns:
        # Unix seconds (Bitstamp format)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    elif "unix" in df.columns:
        df["timestamp"] = pd.to_datetime(df["unix"], unit="ms", utc=True)
        if df["timestamp"].dt.year.max() < 2000:
            # unit was seconds, not ms
            df["timestamp"] = pd.to_datetime(df["unix"], unit="s", utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    else:
        raise ValueError(
            "Cannot identify a timestamp column in the Kaggle CSV. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.set_index("timestamp").sort_index()

    # ---- select OHLCV columns (handle alternate volume column names) ----
    col_map = {}
    for std in ["open", "high", "low", "close"]:
        if std in df.columns:
            col_map[std] = std
        else:
            raise ValueError(f"Expected column '{std}' not found in Kaggle CSV.")

    # Volume column varies across Kaggle datasets
    for candidate in ["volume", "volume_(btc)", "volume_btc", "volume_currency"]:
        if candidate in df.columns:
            col_map[candidate] = "volume"
            break
    else:
        raise ValueError("Cannot identify a volume column in the Kaggle CSV.")

    df = df.rename(columns=col_map)[OHLCV_COLS]

    # ---- coerce to float, drop rows where price is NaN or 0 ----
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df["close"] > 0].dropna(subset=OHLCV_COLS)

    # ---- filter to 2019-present ----
    df = df[df.index >= KAGGLE_START]

    print(f"[data] Kaggle CSV: {len(df):,} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# 2. Fetch Binance klines
# ---------------------------------------------------------------------------

def fetch_binance_klines(
    start: str = "2021-01-01",
    end: str | None = None,
    symbol: str = BINANCE_SYMBOL,
) -> pd.DataFrame:
    """Fetch minute-level OHLCV from the Binance public REST API (no key needed).

    Paginates automatically with a 1-second courtesy sleep between requests
    to stay well within rate limits.

    Parameters
    ----------
    start : str
        ISO date string, e.g. "2021-01-01".
    end : str or None
        ISO date string.  Defaults to today (UTC).
    symbol : str
        Binance trading pair, default "BTCUSDT".

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume — indexed by UTC DatetimeIndex
        named 'timestamp'.
    """
    try:
        from binance.client import Client
    except ImportError as exc:
        raise ImportError(
            "python-binance is not installed. Run: pip install python-binance"
        ) from exc

    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"[data] Fetching Binance {symbol} {BINANCE_INTERVAL} klines "
          f"{start} → {end} …")

    # Public client — no API key required for historical klines
    client = Client()

    rows = []
    current_start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    total_fetched = 0

    while current_start_ms < end_ms:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=BINANCE_INTERVAL,
                startTime=current_start_ms,
                endTime=end_ms,
                limit=BINANCE_LIMIT,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Binance API request failed: {exc}. "
                "Check your internet connection."
            ) from exc

        if not klines:
            break

        rows.extend(klines)
        total_fetched += len(klines)

        # Advance cursor past the last returned candle
        current_start_ms = klines[-1][0] + 60_000  # +1 minute in ms

        print(f"[data]   … fetched {total_fetched:,} candles so far", end="\r")
        time.sleep(0.3)  # be polite to the public endpoint

    print()  # newline after the \r progress line

    if not rows:
        raise RuntimeError(
            f"No klines returned from Binance for {symbol} between {start} and {end}."
        )

    # Binance kline columns:
    # 0:open_time 1:open 2:high 3:low 4:close 5:volume
    # 6:close_time 7:quote_asset_vol 8:num_trades 9-11: taker fields
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[OHLCV_COLS].apply(pd.to_numeric, errors="coerce")
    df = df[df["close"] > 0].dropna(subset=OHLCV_COLS)
    df = df.sort_index()

    print(f"[data] Binance: {len(df):,} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# 3. Merge sources
# ---------------------------------------------------------------------------

def merge_sources(kaggle: pd.DataFrame, binance: pd.DataFrame) -> pd.DataFrame:
    """Concatenate Kaggle and Binance DataFrames, deduplicate, and sort.

    Where timestamps overlap, Binance data takes priority (it is kept and the
    Kaggle row for the same minute is dropped).

    Parameters
    ----------
    kaggle, binance : pd.DataFrame
        Minute-level OHLCV DataFrames with a UTC DatetimeIndex named 'timestamp'.

    Returns
    -------
    pd.DataFrame
        Merged, deduplicated, and sorted DataFrame.
    """
    print("[data] Merging sources …")

    kaggle = kaggle.copy()
    binance = binance.copy()

    # Tag source so we can resolve duplicates deterministically
    kaggle["_source"] = 0   # lower priority
    binance["_source"] = 1  # higher priority

    combined = pd.concat([kaggle, binance])
    combined = combined.sort_index()

    # Keep the Binance row when timestamps clash
    combined = (
        combined
        .reset_index()
        .sort_values(["timestamp", "_source"], ascending=[True, False])
        .drop_duplicates(subset="timestamp", keep="first")
        .set_index("timestamp")
        .drop(columns="_source")
        .sort_index()
    )

    n_kaggle = len(kaggle)
    n_binance = len(binance)
    n_overlap = n_kaggle + n_binance - len(combined)

    print(f"[data] Merged: {len(combined):,} rows  "
          f"(Kaggle {n_kaggle:,} + Binance {n_binance:,}, "
          f"overlap removed {n_overlap:,})  "
          f"({combined.index[0].date()} → {combined.index[-1].date()})")
    return combined


# ---------------------------------------------------------------------------
# 4. Resample
# ---------------------------------------------------------------------------

def resample_ohlcv(
    df: pd.DataFrame,
    timeframes: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Resample a minute-level OHLCV DataFrame to multiple timeframes.

    Parameters
    ----------
    df : pd.DataFrame
        Minute-level OHLCV with UTC DatetimeIndex.
    timeframes : dict, optional
        Mapping of label → pandas offset alias, e.g. {"5m": "5min", "1h": "1h"}.
        Defaults to the module-level TIMEFRAMES constant.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the labels from `timeframes`; values are resampled DataFrames.
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    print("[data] Resampling …")
    frames = {}
    for label, freq in timeframes.items():
        resampled = (
            df[OHLCV_COLS]
            .resample(freq, closed="left", label="left")
            .agg(RESAMPLE_AGG)
            .dropna(subset=OHLCV_COLS)
        )
        # Drop any candles where close is 0 (gaps in data)
        resampled = resampled[resampled["close"] > 0]
        frames[label] = resampled
        print(f"[data]   {label:>4s}: {len(resampled):,} candles  "
              f"({resampled.index[0].date()} → {resampled.index[-1].date()})")

    return frames


# ---------------------------------------------------------------------------
# 5. Save processed files
# ---------------------------------------------------------------------------

def save_processed(
    frames: dict[str, pd.DataFrame],
    out_dir: str | Path = "data/processed",
) -> None:
    """Save each resampled DataFrame to a CSV in `out_dir`.

    Files are named  btc_5m.csv, btc_30m.csv, btc_1h.csv.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Output of :func:`resample_ohlcv`.
    out_dir : str or Path
        Destination directory.  Created if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] Saving processed files to {out_dir.resolve()} …")
    for label, df in frames.items():
        out_path = out_dir / f"btc_{label}.csv"
        df.to_csv(out_path)
        print(f"[data]   Saved {out_path.name}  ({len(df):,} rows)")

    print("[data] Done.")


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    kaggle_path: str | Path = "data/raw/btcusd_1-min_data.csv",
    out_dir: str | Path = "data/processed",
    binance_start: str = "2021-01-01",
    binance_end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the complete data pipeline end-to-end.

    Steps
    -----
    1. Load Kaggle CSV
    2. Fetch Binance klines
    3. Merge sources
    4. Resample to 5m, 30m, 1h
    5. Save to data/processed/

    Parameters
    ----------
    kaggle_path : str or Path
        Path to the raw Kaggle CSV.
    out_dir : str or Path
        Output directory for processed CSVs.
    binance_start : str
        Start date for Binance fetch (should overlap slightly with Kaggle end).
    binance_end : str or None
        End date for Binance fetch.  None = today.

    Returns
    -------
    dict[str, pd.DataFrame]
        Resampled frames keyed by timeframe label ("5m", "30m", "1h").
    """
    print("=" * 60)
    print("Bitcoin Price Data Pipeline")
    print("=" * 60)

    kaggle_df = load_kaggle_csv(kaggle_path)
    binance_df = fetch_binance_klines(start=binance_start, end=binance_end)
    merged_df = merge_sources(kaggle_df, binance_df)
    frames = resample_ohlcv(merged_df)
    save_processed(frames, out_dir)

    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return frames


# ---------------------------------------------------------------------------
# 7. Taker buy ratio enrichment
# ---------------------------------------------------------------------------

_BINANCE_INTERVAL_MAP = {
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
}


def fetch_taker_buy_ratio(
    interval: str,
    start: str = "2021-01-01",
    end: str | None = None,
    symbol: str = BINANCE_SYMBOL,
) -> pd.DataFrame:
    """Fetch taker_buy_ratio at a given candle interval directly from Binance.

    taker_buy_ratio = taker_buy_base_volume / total_volume
      ≈ 1.0  → aggressive buyers dominating (bullish pressure)
      ≈ 0.0  → aggressive sellers dominating (bearish pressure)
      ≈ 0.5  → balanced

    This is an order-flow signal not available in OHLCV alone.

    Parameters
    ----------
    interval : str
        One of "5m", "15m", "30m", "1h".
    start : str
        ISO date string, e.g. "2021-01-01".
    end : str or None
        ISO date string. Defaults to today (UTC).
    symbol : str
        Binance symbol, default "BTCUSDT".

    Returns
    -------
    pd.DataFrame
        Single column 'taker_buy_ratio' with UTC DatetimeIndex.
    """
    try:
        from binance.client import Client
    except ImportError as exc:
        raise ImportError("Run: pip install python-binance") from exc

    if interval not in _BINANCE_INTERVAL_MAP:
        raise ValueError(f"interval must be one of {list(_BINANCE_INTERVAL_MAP)}")

    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    client = Client()
    rows = []
    current_ms = int(start_dt.timestamp() * 1000)
    end_ms     = int(end_dt.timestamp()   * 1000)
    total      = 0

    print(f"[data] Fetching taker_buy_ratio  {interval}  {start} → {end} …")

    while current_ms < end_ms:
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=_BINANCE_INTERVAL_MAP[interval],
                startTime=current_ms,
                endTime=end_ms,
                limit=BINANCE_LIMIT,
            )
        except Exception as exc:
            raise RuntimeError(f"Binance API failed: {exc}") from exc

        if not klines:
            break

        rows.extend(klines)
        total += len(klines)
        current_ms = klines[-1][0] + 1
        print(f"[data]   … {total:,} candles", end="\r")
        time.sleep(0.25)

    print()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"]       = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["taker_buy_base"]  = pd.to_numeric(df["taker_buy_base"],  errors="coerce")
    df["volume"]          = pd.to_numeric(df["volume"],          errors="coerce")
    df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
    df = df.set_index("timestamp")[["taker_buy_ratio"]].dropna().sort_index()

    print(f"[data] taker_buy_ratio {interval}: {len(df):,} candles  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def enrich_with_taker_buy_ratio(
    frames: dict[str, pd.DataFrame],
    binance_start: str = "2021-01-01",
    neutral_fill: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Add taker_buy_ratio to each processed frame by fetching from Binance.

    Rows outside Binance coverage (e.g. Kaggle 2019-2021 period) are filled
    with `neutral_fill` (0.5 = balanced buy/sell pressure).

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Output of resample_ohlcv or loaded from data/processed/.
    binance_start : str
        Start date for Binance fetch (should match the original pipeline).
    neutral_fill : float
        Value used for rows without Binance coverage (default 0.5).

    Returns
    -------
    dict[str, pd.DataFrame]
        Same frames with an added 'taker_buy_ratio' column.
    """
    enriched = {}
    for label, df in frames.items():
        tbr = fetch_taker_buy_ratio(label, start=binance_start)
        df  = df.copy().join(tbr, how="left")

        n_missing = df["taker_buy_ratio"].isna().sum()
        if n_missing > 0:
            df["taker_buy_ratio"] = df["taker_buy_ratio"].fillna(neutral_fill)
            print(f"[data] {label}: filled {n_missing:,} pre-Binance rows with {neutral_fill}")

        enriched[label] = df
        rng = df["taker_buy_ratio"]
        print(f"[data] {label}: taker_buy_ratio  "
              f"mean={rng.mean():.3f}  min={rng.min():.3f}  max={rng.max():.3f}")

    return enriched


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the BTC data pipeline.")
    parser.add_argument(
        "--kaggle", default="data/raw/btcusd_1-min_data.csv",
        help="Path to Kaggle CSV (default: data/raw/btcusd_1-min_data.csv)",
    )
    parser.add_argument(
        "--out", default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--binance-start", default="2021-01-01",
        help="Binance fetch start date (default: 2021-01-01)",
    )
    parser.add_argument(
        "--binance-end", default=None,
        help="Binance fetch end date (default: today)",
    )
    args = parser.parse_args()

    run_pipeline(
        kaggle_path=args.kaggle,
        out_dir=args.out,
        binance_start=args.binance_start,
        binance_end=args.binance_end,
    )
