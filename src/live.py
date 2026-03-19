"""
src/live.py
Live BTC price direction inference via Binance WebSocket.

Behaviour
---------
1. Fetch the last `window_size` closed candles from Binance REST API.
2. Open a WebSocket to stream the current (incomplete) candle in real-time.
3. Construct the live candle on-the-fly:
       open, high_so_far, low_so_far, last_price (close proxy), volume_so_far
4. Append the live candle to the history, build the full feature vector.
5. Every 60 seconds: run inference and print prediction, confidence, candle %.
6. On candle close: rotate history and start tracking the new live candle.

Usage
-----
    python -m src.live \\
        --model-path models/lgbm_5m.pkl \\
        --interval 5m
"""

import argparse
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_features, get_feature_cols

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
INFER_INTERVAL_S = 60          # run inference every N seconds
HISTORY_BUFFER = 200           # extra closed candles to fetch (for indicator warmup)

_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
}


# ---------------------------------------------------------------------------
# Live candle state (mutable, updated by WebSocket thread)
# ---------------------------------------------------------------------------

@dataclass
class LiveCandle:
    open_time_ms: int = 0
    open:         float = 0.0
    high:         float = 0.0
    low:          float = 0.0
    close:        float = 0.0   # last traded price
    volume:       float = 0.0
    is_closed:    bool = False

    def to_series(self, timestamp: pd.Timestamp) -> pd.Series:
        return pd.Series(
            {
                "open":   self.open,
                "high":   self.high,
                "low":    self.low,
                "close":  self.close,
                "volume": self.volume,
            },
            name=timestamp,
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LiveInference:
    """Stream BTC klines and run periodic direction predictions.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved LightGBM model (.pkl).
    interval : str
        Candle interval: "1m", "5m", "30m", or "1h".
    window_size : int
        Number of candles to keep in history buffer.
    symbol : str
        Binance symbol, default "BTCUSDT".
    infer_every : int
        Inference interval in seconds, default 60.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "lgbm",
        interval: str = "5m",
        window_size: int = 50,
        symbol: str = SYMBOL,
        infer_every: int = INFER_INTERVAL_S,
    ):
        if interval not in _INTERVAL_MS:
            raise ValueError(f"interval must be one of {list(_INTERVAL_MS)}")

        self.interval = interval
        self.interval_ms = _INTERVAL_MS[interval]
        self.window_size = window_size
        self.symbol = symbol
        self.infer_every = infer_every
        self.model_type = model_type.lower()

        self._live = LiveCandle()
        self._lock = threading.Lock()
        self._history: pd.DataFrame | None = None  # closed candles, OHLCV only

        self.model = self._load_model(model_path, model_type)
        print(f"[live] Model loaded  type={model_type}  interval={interval}  "
              f"window={window_size}")

    # ---- model loading ----

    @staticmethod
    def _load_model(path: str | Path, model_type: str):
        from src.models import get_model

        m = get_model(model_type)
        m.load(path)
        return m

    # ---- REST: fetch closed candles ----

    def _fetch_closed_candles(self, n: int) -> pd.DataFrame:
        """Fetch the last `n` closed minute-resolution klines and resample."""
        try:
            from binance.client import Client
        except ImportError as exc:
            raise ImportError("Run: pip install python-binance") from exc

        client = Client()

        # We always fetch at minute level then resample to avoid fetching
        # too many candles for longer intervals
        minutes_per_candle = self.interval_ms // 60_000
        n_minutes = (n + HISTORY_BUFFER) * minutes_per_candle

        print(f"[live] Fetching {n} closed {self.interval} candles from Binance …")
        klines = client.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=min(n + HISTORY_BUFFER + 1, 1000),
        )

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[OHLCV_COLS].astype(float)

        # Drop the last candle — it may be the still-open one
        df = df.iloc[:-1]

        # Keep only the last `n` complete candles
        df = df.iloc[-n:]
        print(f"[live] History: {len(df)} closed candles  "
              f"({df.index[0]} → {df.index[-1]})")
        return df

    # ---- WebSocket callback ----

    def _on_kline_message(self, msg: dict) -> None:
        """Called by the Binance WebSocket thread on every tick."""
        if msg.get("e") == "error":
            print(f"[live][ws] Error: {msg}")
            return

        k = msg.get("k", {})
        with self._lock:
            self._live.open_time_ms = k["t"]
            self._live.open   = float(k["o"])
            self._live.high   = float(k["h"])
            self._live.low    = float(k["l"])
            self._live.close  = float(k["c"])
            self._live.volume = float(k["v"])
            self._live.is_closed = k["x"]

            if self._live.is_closed:
                # Rotate: append closed candle to history, reset live state
                self._rotate_candle()

    def _rotate_candle(self) -> None:
        """Append the just-closed live candle to history (called under lock)."""
        ts = pd.Timestamp(self._live.open_time_ms, unit="ms", tz="UTC")
        new_row = self._live.to_series(ts).to_frame().T
        new_row.index.name = "timestamp"
        self._history = pd.concat([self._history, new_row]).iloc[-(self.window_size + HISTORY_BUFFER):]
        self._live = LiveCandle()

    # ---- candle completion ----

    def _candle_completion_pct(self) -> float:
        """Estimate how far through the current candle we are (0–100 %)."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        elapsed = now_ms - self._live.open_time_ms
        return min(100.0, elapsed / self.interval_ms * 100)

    # ---- build feature vector ----

    def _build_feature_df(self) -> pd.DataFrame | None:
        """Append the live candle to history and run feature engineering."""
        with self._lock:
            if self._history is None or len(self._history) < self.window_size:
                return None

            history = self._history.copy()
            live_candle = self._live

        # Build a live candle row using current price as close proxy
        ts = pd.Timestamp(live_candle.open_time_ms, unit="ms", tz="UTC")
        live_row = pd.DataFrame(
            [{
                "open":   live_candle.open   or history["close"].iloc[-1],
                "high":   live_candle.high   or live_candle.close,
                "low":    live_candle.low    or live_candle.close,
                "close":  live_candle.close  or history["close"].iloc[-1],
                "volume": live_candle.volume,
            }],
            index=pd.DatetimeIndex([ts], name="timestamp"),
        )

        combined = pd.concat([history, live_row])

        try:
            featured = build_features(combined, drop_na=False)
        except Exception as exc:
            print(f"[live] Feature engineering error: {exc}")
            return None

        # Drop rows that still have NaN after indicator warmup
        featured = featured.dropna(subset=get_feature_cols(featured))
        if len(featured) == 0:
            return None

        return featured

    # ---- inference ----

    def _infer(self) -> None:
        """Build features, run model, print result."""
        featured = self._build_feature_df()
        if featured is None:
            print("[live] Waiting for enough history …")
            return

        feat_cols = get_feature_cols(featured)
        completion = self._candle_completion_pct()

        try:
            X = featured[feat_cols].values[-1:].astype(np.float32)
            prob = float(self.model.predict_proba(X)[-1])

        except Exception as exc:
            print(f"[live] Inference error: {exc}")
            return

        direction = "UP  ▲" if prob >= 0.5 else "DOWN ▼"
        confidence = max(prob, 1 - prob) * 100
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(
            f"[{now_str}]  {self.symbol} {self.interval}  "
            f"→ {direction}  "
            f"confidence={confidence:.1f}%  "
            f"candle={completion:.1f}% complete  "
            f"last_price={self._live.close:.2f}"
        )

    # ---- main loop ----

    def run(self) -> None:
        """Start WebSocket and run inference on a fixed interval."""
        try:
            from binance import ThreadedWebsocketManager
        except ImportError as exc:
            raise ImportError("Run: pip install python-binance") from exc

        # Pre-fetch history before opening WebSocket
        n_candles = self.window_size + HISTORY_BUFFER
        self._history = self._fetch_closed_candles(n_candles)

        # Start WebSocket in background thread
        twm = ThreadedWebsocketManager()
        twm.start()

        stream_name = twm.start_kline_socket(
            callback=self._on_kline_message,
            symbol=self.symbol,
            interval=self.interval,
        )
        print(f"[live] WebSocket open  stream={stream_name}")
        print(f"[live] Running inference every {self.infer_every}s  (Ctrl-C to stop)\n")

        try:
            while True:
                self._infer()
                time.sleep(self.infer_every)
        except KeyboardInterrupt:
            print("\n[live] Stopping …")
        finally:
            twm.stop_socket(stream_name)
            twm.join()
            print("[live] Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live BTC price direction inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to saved model (without extension for LSTM/TFT).",
    )
    parser.add_argument(
        "--model-type", default="lgbm", choices=["lgbm"],
        help="Model type.",
    )
    parser.add_argument(
        "--interval", default="5m", choices=list(_INTERVAL_MS),
        help="Candle interval.",
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Lookback window size (candles).",
    )
    parser.add_argument(
        "--symbol", default=SYMBOL,
        help="Binance trading pair.",
    )
    parser.add_argument(
        "--every", type=int, default=INFER_INTERVAL_S,
        help="Inference interval in seconds.",
    )
    args = parser.parse_args()

    LiveInference(
        model_path=args.model_path,
        model_type=args.model_type,
        interval=args.interval,
        window_size=args.window,
        symbol=args.symbol,
        infer_every=args.every,
    ).run()


if __name__ == "__main__":
    main()
