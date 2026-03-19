"""
src/features.py
Feature engineering pipeline for all three time-horizon datasets.

Public API
----------
add_lagged_returns(df, lags)          -> pd.DataFrame
add_technical_indicators(df)          -> pd.DataFrame
add_volume_zscore(df, window)         -> pd.DataFrame
add_time_features(df)                 -> pd.DataFrame
create_labels(df, horizon)            -> pd.DataFrame
build_features(df)                    -> pd.DataFrame   (full pipeline)
"""

import numpy as np
import pandas as pd

try:
    import ta as ta_lib
except ImportError as exc:
    raise ImportError(
        "ta is not installed. Run: pip install ta"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LAGS = [1, 2, 4, 6, 12, 24]

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOL_ZSCORE_WINDOW = 24

FEATURE_COLS = None  # populated by build_features; exposed for downstream use


# ---------------------------------------------------------------------------
# 1. Lagged returns
# ---------------------------------------------------------------------------

def add_lagged_returns(
    df: pd.DataFrame,
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add log-return columns for each lag in `lags`.

    Column names:  ret_1, ret_2, ret_4, …

    Log return at lag k:  ln(close_t / close_{t-k})

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    lags : list[int]
        Number of candles to look back.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new return columns appended (in-place copy).
    """
    df = df.copy()
    for lag in lags:
        df[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
    return df


# ---------------------------------------------------------------------------
# 2. Technical indicators via ta
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Band width, and ATR using the `ta` library.

    Columns added
    -------------
    rsi_14          — Relative Strength Index (14-period)
    macd            — MACD line
    macd_signal     — Signal line
    macd_hist       — Histogram (MACD − signal)
    bb_width        — Bollinger Band width = (upper − lower) / middle
    atr_14          — Average True Range (14-period)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with indicator columns appended.
    """
    df = df.copy()

    # RSI
    df["rsi_14"] = ta_lib.momentum.RSIIndicator(
        close=df["close"], window=RSI_PERIOD
    ).rsi()

    # MACD
    macd = ta_lib.trend.MACD(
        close=df["close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # Bollinger Bands — width = (upper − lower) / middle
    bb = ta_lib.volatility.BollingerBands(
        close=df["close"], window=BB_PERIOD, window_dev=BB_STD
    )
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    # ATR
    df["atr_14"] = ta_lib.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_PERIOD
    ).average_true_range()

    return df


# ---------------------------------------------------------------------------
# 3. Rolling volume z-score
# ---------------------------------------------------------------------------

def add_volume_zscore(
    df: pd.DataFrame,
    window: int = VOL_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Add a rolling z-score of volume over `window` candles.

    z = (volume − rolling_mean) / rolling_std

    Column added:  vol_zscore

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'volume' column.
    window : int
        Rolling window size (default 24 candles).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 'vol_zscore' appended.
    """
    df = df.copy()
    roll = df["volume"].rolling(window, min_periods=window)
    df["vol_zscore"] = (df["volume"] - roll.mean()) / (roll.std() + 1e-9)
    return df


# ---------------------------------------------------------------------------
# 4. Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week as integer features.

    Columns added
    -------------
    hour        — 0–23  (UTC)
    day_of_week — 0=Monday … 6=Sunday

    Parameters
    ----------
    df : pd.DataFrame
        Must have a UTC DatetimeIndex named 'timestamp'.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with time feature columns appended.
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    return df


# ---------------------------------------------------------------------------
# 5. Labels
# ---------------------------------------------------------------------------

def create_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create a binary classification target.

    target = 1  if  close_{t + horizon} > close_t
    target = 0  otherwise

    The last `horizon` rows will have NaN labels and should be dropped
    before training.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    horizon : int
        Number of candles ahead to predict (default 1 = next candle).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a 'target' column appended.
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype(float)
    # Mark the last `horizon` rows as NaN (no future close available)
    df.loc[df.index[-horizon:], "target"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 6. Full feature pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    lags: list[int] = DEFAULT_LAGS,
    vol_window: int = VOL_ZSCORE_WINDOW,
    horizon: int = 1,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Apply the complete feature engineering pipeline to a resampled OHLCV DataFrame.

    Steps
    -----
    1. Lagged log-returns
    2. RSI, MACD, Bollinger Band width, ATR
    3. Rolling volume z-score
    4. Hour of day, day of week
    5. Binary target label (next-candle direction)
    6. Drop NaN rows (warm-up period from indicators + label shift)

    Parameters
    ----------
    df : pd.DataFrame
        Resampled OHLCV DataFrame (output of resample_ohlcv).
    lags : list[int]
        Lag periods for return features.
    vol_window : int
        Rolling window for volume z-score.
    horizon : int
        Prediction horizon in candles.
    drop_na : bool
        Whether to drop rows containing NaN after feature creation (default True).

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame ready for modelling.  OHLCV columns are
        retained so the DataFrame can still be used for further processing.
    """
    print(f"[features] Building features  (rows before: {len(df):,}) …")

    df = add_lagged_returns(df, lags=lags)
    df = add_technical_indicators(df)
    df = add_volume_zscore(df, window=vol_window)
    df = add_time_features(df)
    df = create_labels(df, horizon=horizon)

    if drop_na:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        print(f"[features] Dropped {dropped:,} NaN rows (indicator warm-up + label shift)")

    print(f"[features] Done  (rows after: {len(df):,},  features: {len(get_feature_cols(df))})")
    return df


# ---------------------------------------------------------------------------
# Helper — feature column names (everything except OHLCV and target)
# ---------------------------------------------------------------------------

_OHLCV = {"open", "high", "low", "close", "volume"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes OHLCV and 'target').

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`build_features`.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    return [c for c in df.columns if c not in _OHLCV and c != "target"]
