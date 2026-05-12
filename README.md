# AltFore: Stock Forecasting with Alternative Data

AltFore predicts next-day stock return direction using machine learning models
augmented with Reddit (WallStreetBets) mention data. The project emphasises
out-of-sample evaluation and economic significance via trading strategies.

## Project Goals

- Out-of-sample ML return forecasting on daily stock data
- Alternative data signals: Reddit mention volume, mention momentum, abnormal attention
- Economic significance: evaluate via long-short portfolios and quintile analysis, not just model metrics
- Expanding to additional signal sources (search trends, sentiment scoring)

---

## Data

### Price data
Daily OHLCV downloaded from Yahoo Finance via `yfinance` for 2021–2025.

### WSB mention data
Historical daily mention counts from WallStreetBets, stored in `wsb_data/` as one CSV per year
(2021–2025). Each file covers ~100 tickers ranked by annual mention volume.

### Ticker universe
20 curated tickers with at least 4 years of WSB coverage and meaningful mention volume.
ETFs, indexes, crypto proxies, and tickers with fewer than 2 years of WSB data were excluded.

| Ticker | Notes |
|--------|-------|
| AAPL, AMZN, MSFT | Large-cap anchors |
| TSLA, NVDA, AMD, INTC | Semiconductor / EV high-attention names |
| GME, AMC | Meme stock core |
| PLTR, COIN, SNAP | Retail-driven growth |
| NFLX, LULU | Earnings-driven volatility |
| F, GM, BA, JPM, TSM | Sector diversifiers |
| MU, AMZN | Cyclical / large-cap |

**Dataset: 25,031 rows, 20 tickers, 2021–2025.**

---

## Features

All features are constructed to be available at close of day T, predicting direction at close of day T+1.
No lookahead.

### Price features
| Feature | Description |
|---------|-------------|
| `return_1d` | Prior day return |
| `momentum_5d` | 5-day price momentum |
| `momentum_20d` | 20-day price momentum |
| `volatility_20d` | Rolling 20-day return std |
| `volume_ratio` | Volume / 20-day average volume |

### Mention features (within-ticker normalised)
| Feature | Description |
|---------|-------------|
| `mentions_log` | log1p of yesterday's raw mention count |
| `mentions_log_chg_5d` | Pct change of 7-day smoothed log-mentions vs 5 days ago |
| `mentions_abnormal` | Z-score of log-mentions vs 30-day rolling mean/std |
| `mentions_vol_scaled` | Abnormal mentions divided by realised volatility |
| `mentions_volume_scaled` | Raw mentions divided by 20-day average volume |

Within-ticker normalisation (log1p, rolling z-score) was introduced after initial features
showed zero importance — raw mention counts are dominated by cross-ticker scale differences
(GME peaks at 267,000 mentions/day vs JPM at 305).

### Target
`direction_fwd_1d` — binary: 1 if next-day close > today's close.

---

## Model

### Architecture
LightGBM binary classifier (primary), benchmarked against Random Forest, Extra Trees,
Logistic Regression (L1 and L2), and a majority-class dummy baseline.

### Walk-forward evaluation
Two expanding windows to avoid lookahead and measure consistency:

| Window | Train | Val | Test |
|--------|-------|-----|------|
| 1 | 2022 | 2023 | 2024 |
| 2 | 2022–2023 | 2024 | 2025 |

Val split is used only for early stopping (LightGBM) — not for model selection.

### Hyperparameters (LightGBM)
`n_estimators=300`, `learning_rate=0.03`, `num_leaves=15`, `min_child_samples=20`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, early stopping patience=30.

---

## Results

### Model comparison — test AUC (20-ticker universe)

| Model | 2024 | 2025 | Mean |
|-------|------|------|------|
| LightGBM | 0.504 | 0.509 | **0.507** |
| Extra Trees | 0.496 | 0.512 | 0.504 |
| Random Forest | 0.494 | 0.513 | 0.503 |
| Logistic L1 | 0.497 | 0.498 | 0.498 |
| Dummy baseline | 0.500 | 0.500 | 0.500 |

LightGBM is the strongest model overall. Tree models consistently beat linear models
out-of-sample, suggesting non-linear feature interactions are meaningful.

### Feature importance (Extra Trees, 2022–2023 → 2025 window)
`return_1d` dominates. All five mention features register non-zero importance after
within-ticker normalisation was introduced; `mentions_log`, `mentions_log_chg_5d`,
and `mentions_abnormal` each score comparably to the price momentum features.

### Signal quality (Extra Trees, test 2025)
| Metric | Value |
|--------|-------|
| Accuracy | 51.3% |
| AUC | 0.512 |
| Spearman IC | 0.021 (p=0.145) |
| Brier score | 0.250 (baseline 0.250) |

The IC is positive but not yet significant at the portfolio level. Monthly IC is 50%
positive in both test years — the signal is lumpy rather than consistent.

### Long-short portfolio (Extra Trees, top-2 / bottom-2 daily by predicted probability)
| Year | Ann. Return | Sharpe | Max Drawdown |
|------|-------------|--------|--------------|
| 2024 | -75% | -0.79 | -81% |
| 2025 | +28% | +0.58 | -28% |

2024 results show the model is actively harmful in that window. 2025 is positive but
t-stat of 0.57 is not significant. Quintile return spread is near-flat in both years,
meaning the probability ranking does not yet reliably separate return outcomes.

### Conditional IC (test 2025)
| Condition | IC | p |
|-----------|-----|---|
| High abnormal mentions | 0.008 | 0.70 |
| Low abnormal mentions | 0.031 | 0.12 |
| High volatility | 0.009 | 0.66 |
| Low volatility | 0.018 | 0.38 |

No stable regime conditioning effect detected yet.

---

## Known Limitations and Next Steps

| Area | Status |
|------|--------|
| Probability calibration | Not implemented — all predictions cluster near 0.50 |
| Ticker expansion | 20 tickers; WSB data has ~100, more expansion possible |
| Sentiment polarity | Only mention counts, no tone scoring |
| Google Trends signal | Not started |
| Viable trading strategy | Not yet — quintile spread is flat, L/S Sharpe near zero |

---

## Architecture

```
scripts/          — thin CLI wrappers only
src/altfore/
  ingest/         — Reddit RSS fetching and parsing
  pipeline/       — dataset construction (prices + WSB mentions)
  modeling/       — feature engineering, model training, evaluation metrics
  visualization/  — plotting utilities
wsb_data/         — raw WSB mention CSVs (2021–2025, not hand-edited)
dataset/          — generated outputs (not hand-edited)
```

### Key generated files
| File | Description |
|------|-------------|
| `dataset/wsb_model_dataset.csv` | 25,031-row price + mentions panel |
| `dataset/model_comparison.csv` | AUC / IC / accuracy per model × split |
| `dataset/deep_eval.csv` | Brier, quintile spread, L/S metrics per split |
| `dataset/feature_importance.csv` | Feature importances per model × split |
| `dataset/ls_daily_returns.csv` | Daily L/S portfolio returns per test year |

---

## Quick Start

```bash
pip install -r requirements.txt

# Rebuild the WSB price + mentions dataset
python scripts/build_wsb_dataset.py

# Train and evaluate all models
python scripts/train_model.py
```
