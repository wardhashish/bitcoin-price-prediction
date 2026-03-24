"""
src/features.py
Feature engineering pipeline for all four time-horizon datasets.

Public API
----------
add_lagged_returns(df, lags)          -> pd.DataFrame
add_technical_indicators(df)          -> pd.DataFrame
add_tier1_features(df)                -> pd.DataFrame
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

RSI_PERIOD    = 14
MACD_FAST     = 12
MACD_SLOW     = 26
MACD_SIGNAL   = 9
BB_PERIOD     = 20
BB_STD        = 2
ATR_PERIOD    = 14
VOL_ZSCORE_WINDOW = 24

# Tier-1 extra indicators
STOCH_PERIOD  = 14
STOCH_SMOOTH  = 3
MFI_PERIOD    = 14
ADX_PERIOD    = 14
OBV_ROC_PERIOD = 10   # rate-of-change window for OBV
EMA_FAST      = 9
EMA_SLOW      = 21


# ---------------------------------------------------------------------------
# 1. Lagged returns
# ---------------------------------------------------------------------------

def add_lagged_returns(
    df: pd.DataFrame,
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add log-return columns for each lag in `lags`.

    Log return at lag k:  ln(close_t / close_{t-k})
    """
    df = df.copy()
    for lag in lags:
        df[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
    return df


# ---------------------------------------------------------------------------
# 2. Original technical indicators
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Band width, and ATR.

    Columns added
    -------------
    rsi_14       — Relative Strength Index (14-period)
    macd         — MACD line
    macd_signal  — Signal line
    macd_hist    — Histogram (MACD − signal)
    bb_width     — Bollinger Band width = (upper − lower) / middle
    atr_14       — Average True Range (14-period)
    """
    df = df.copy()

    df["rsi_14"] = ta_lib.momentum.RSIIndicator(
        close=df["close"], window=RSI_PERIOD
    ).rsi()

    macd = ta_lib.trend.MACD(
        close=df["close"],
        window_slow=MACD_SLOW,
        window_fast=MACD_FAST,
        window_sign=MACD_SIGNAL,
    )
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    bb = ta_lib.volatility.BollingerBands(
        close=df["close"], window=BB_PERIOD, window_dev=BB_STD
    )
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    df["atr_14"] = ta_lib.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=ATR_PERIOD
    ).average_true_range()

    return df


# ---------------------------------------------------------------------------
# 3. Tier-1 advanced features
# ---------------------------------------------------------------------------

def add_tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six Tier-1 advanced features used by professional technical analysts.

    Columns added
    -------------
    candle_body     — (close − open) / (high − low + ε)
                      Positive = bullish candle, negative = bearish.
                      Magnitude = how much of the range was body (commitment).
    upper_wick      — (high − max(open,close)) / (high − low + ε)
                      Large upper wick = sellers rejected higher prices.
    lower_wick      — (min(open,close) − low) / (high − low + ε)
                      Large lower wick = buyers rejected lower prices.
    adx_14          — Average Directional Index (14-period).
                      0–25 = no trend, 25–50 = trending, 50+ = strong trend.
                      High ADX = momentum strategy works; low ADX = mean-revert.
    stoch_k         — Stochastic %K (14-period): where is close in recent range?
                      (close − lowest_low) / (highest_high − lowest_low) × 100
    stoch_d         — Stochastic %D: 3-period SMA of %K (signal line).
    mfi_14          — Money Flow Index (14-period): RSI weighted by volume×price.
                      >80 = overbought, <20 = oversold (with volume confirmation).
    obv_roc         — Rate of change of On-Balance Volume over OBV_ROC_PERIOD.
                      Rising OBV_ROC = accumulation, falling = distribution.
    ema_ratio_fast  — close / EMA(9): distance of price from fast moving average.
    ema_ratio_slow  — close / EMA(21): distance of price from slow moving average.
    ema_cross       — EMA(9) / EMA(21): >1 means fast above slow (bullish cross).
    """
    df = df.copy()
    hl_range = df["high"] - df["low"] + 1e-9

    # ---- Candle structure ----
    df["candle_body"] = (df["close"] - df["open"]) / hl_range
    df["upper_wick"]  = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range
    df["lower_wick"]  = (df[["open", "close"]].min(axis=1) - df["low"]) / hl_range

    # ---- ADX ----
    adx = ta_lib.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=ADX_PERIOD
    )
    df["adx_14"] = adx.adx()

    # ---- Stochastic ----
    stoch = ta_lib.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=STOCH_PERIOD, smooth_window=STOCH_SMOOTH,
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ---- MFI ----
    df["mfi_14"] = ta_lib.volume.MFIIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        volume=df["volume"], window=MFI_PERIOD,
    ).money_flow_index()

    # ---- OBV rate of change ----
    obv = ta_lib.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    df["obv_roc"] = obv.pct_change(periods=OBV_ROC_PERIOD)

    # ---- EMA ratios ----
    ema_fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_ratio_fast"] = df["close"] / ema_fast
    df["ema_ratio_slow"] = df["close"] / ema_slow
    df["ema_cross"]      = ema_fast / ema_slow

    return df


# ---------------------------------------------------------------------------
# 4. Rolling volume z-score
# ---------------------------------------------------------------------------

def add_volume_zscore(
    df: pd.DataFrame,
    window: int = VOL_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Add a rolling z-score of volume over `window` candles."""
    df = df.copy()
    roll = df["volume"].rolling(window, min_periods=window)
    df["vol_zscore"] = (df["volume"] - roll.mean()) / (roll.std() + 1e-9)
    return df


# ---------------------------------------------------------------------------
# 5. Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day and day-of-week as integer features."""
    df = df.copy()
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    return df


# ---------------------------------------------------------------------------
# 6. Labels
# ---------------------------------------------------------------------------

def create_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create a binary classification target.

    target = 1  if  close_{t + horizon} > close_t,  else 0.
    Last `horizon` rows are set to NaN (no future close available).
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = (future_close > df["close"]).astype(float)
    df.loc[df.index[-horizon:], "target"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 7. Full feature pipeline
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
    1. Lagged log-returns (ret_1, ret_2, ret_4, ret_6, ret_12, ret_24)
    2. RSI, MACD, Bollinger Band width, ATR
    3. Tier-1 advanced features (candle structure, ADX, Stochastic, MFI, OBV ROC, EMA ratios)
    4. Rolling volume z-score
    5. Hour of day, day of week
    6. Binary target label
    7. Drop NaN rows
    """
    print(f"[features] Building features  (rows before: {len(df):,}) …")

    df = add_lagged_returns(df, lags=lags)
    df = add_technical_indicators(df)
    df = add_tier1_features(df)
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
# Helper — feature column names
# ---------------------------------------------------------------------------

_OHLCV = {"open", "high", "low", "close", "volume"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes OHLCV and 'target')."""
    return [c for c in df.columns if c not in _OHLCV and c != "target"]
