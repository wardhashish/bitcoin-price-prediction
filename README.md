# Bitcoin Price Direction Prediction

**CMPS 261 — Machine Learning Course Project**

Predicts Bitcoin (BTC/USDT) price direction (up/down) across three time horizons: **5 minutes**, **30 minutes**, and **1 hour** using LightGBM.

## Model

**LightGBM** — gradient boosted decision tree classifier, chosen for its strong performance on tabular financial data.

| Horizon | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| 5m | 51.36% | 0.489 | 0.522 |
| 30m | 52.33% | 0.540 | 0.531 |
| **1h** | **53.19%** | **0.544** | **0.538** |

## Data Sources

- **Kaggle Bitcoin Historical Data** (2019–2021, minute-level) → `data/raw/`
- **Binance Public API** (2021–present, minute-level, no API key required)
- Combined window: 2019 to present

## Features

Lagged returns (ret_1, ret_2, ret_4, ret_6, ret_12, ret_24), RSI-14, MACD, Bollinger Band width, ATR-14, volume z-score, hour of day, day of week — 15 features total.

## Project Structure

```
data/
  raw/          # raw CSVs (gitignored)
  processed/    # resampled CSVs (gitignored)
models/
  lgbm_5m.pkl   # trained LightGBM — 5m horizon
  lgbm_30m.pkl  # trained LightGBM — 30m horizon
  lgbm_1h.pkl   # trained LightGBM — 1h horizon
notebooks/
  01_data_pipeline.ipynb  # fetch, merge, resample data
  02_eda.ipynb            # exploratory data analysis
  03_features.ipynb       # feature engineering
  04_models.ipynb         # LightGBM training
  05_evaluation.ipynb     # test set evaluation & plots
src/
  data.py       # data loading, merging, resampling
  features.py   # feature engineering pipeline
  models.py     # LightGBM model class
  evaluate.py   # metrics and evaluation utilities
  live.py       # live inference via Binance WebSocket
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
```

**Option A — Notebook (no terminal needed)**

Open `notebooks/05_evaluation.ipynb` and run the last cell ("Live Prediction (Single-Shot)").
It fetches the latest 1h candles from Binance, builds features, and prints the prediction inline.

**Option B — Terminal (live WebSocket stream)**

```bash
python3 -m src.live --model-path models/lgbm_1h.pkl --interval 1h
```

Expected output every 60 seconds:
```
[2026-03-19 03:10:05 UTC / 06:10:05 AST]  BTCUSDT 1h
  Predicting candle: 06:00 → 07:00 (local)
  Direction:         UP  ▲
  Confidence:        58.3%
  Current candle:    45.2% complete
  Last price:        $83,500.00
```

## Split Strategy

Strict chronological train / val / test split — no shuffling, no data leakage.

```
2019 ──────────────── 2023 │ 2023──2024 │ 2024──2026
        Train (70%)        │  Val (15%) │ Test (15%)
```
