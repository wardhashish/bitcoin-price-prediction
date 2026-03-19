# Bitcoin Price Direction Prediction

**CMPS 261 — Machine Learning Course Project**

Predicts Bitcoin (BTC/USDT) price direction (up/down) across three time horizons: **5 minutes**, **30 minutes**, and **1 hour**.

## Models

| Model | Role |
|---|---|
| Logistic Regression | Sanity baseline |
| LightGBM | Tabular benchmark |
| LSTM | Sequential deep learning |
| Temporal Fusion Transformer | Attention-based forecasting |

## Data Sources

- **Kaggle Bitcoin Historical Data** (2019–2021, minute-level) → `data/raw/`
- **Binance Public API** (2021–present, minute-level, no API key required)
- Combined window: 2019 to present

## Features

Lagged returns, RSI, MACD, Bollinger Bands, ATR, volume z-score, hour of day, day of week.

## Project Structure

```
data/
  raw/          # raw CSVs (gitignored)
  processed/    # resampled & feature-engineered CSVs (gitignored)
notebooks/
  01_data_pipeline.ipynb
  02_eda.ipynb
  03_features.ipynb
  04_models.ipynb
  05_evaluation.ipynb
src/
  data.py       # data loading, merging, resampling
  features.py   # feature engineering
  models.py     # model definitions and training
  evaluate.py   # metrics and evaluation utilities
  live.py       # live inference via Binance WebSocket
```

## Live Inference

`src/live.py` connects to the Binance WebSocket, constructs the current incomplete candle in real-time, and runs inference every 60 seconds, outputting prediction (up/down), confidence score, and candle completion %.

## Split Strategy

Strict chronological train/val/test split — no shuffling, no data leakage.
