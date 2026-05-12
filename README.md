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

| Model | 2024 (raw) | 2025 (raw) | Mean | 2024 (cal) | 2025 (cal) |
|-------|-----------|-----------|------|-----------|-----------|
| LightGBM | 0.504 | 0.509 | **0.507** | 0.496 | 0.509 |
| Extra Trees | 0.496 | 0.512 | 0.504 | 0.496 | 0.512 |
| Random Forest | 0.494 | 0.513 | 0.503 | 0.494 | 0.513 |
| Logistic L1 | 0.497 | 0.498 | 0.498 | 0.497 | 0.498 |
| Dummy baseline | 0.500 | 0.500 | 0.500 | — | — |

LightGBM is the strongest raw model. Calibration does not materially change AUC
(ranking is preserved by a monotone transform), but dramatically improves probability
spread and calibration error (see below).

### Feature importance (LightGBM, 2022–2023 → 2025 window)
`return_1d` dominates. All five mention features register non-zero importance after
within-ticker normalisation was introduced; `mentions_log`, `mentions_log_chg_5d`,
and `mentions_abnormal` each score comparably to the price momentum features.

### Probability calibration (LightGBM)

Raw predictions cluster in an extremely narrow band around 0.47–0.50 — the model
can rank but produces no actionable confidence scores. Platt scaling (sigmoid
calibration fitted on the val split via `CalibratedClassifierCV(FrozenEstimator(...))`)
substantially corrects this.

| Metric | Raw (2024) | Calibrated (2024) | Raw (2025) | Calibrated (2025) |
|--------|-----------|------------------|-----------|------------------|
| Mean predicted prob | 0.468 | 0.530 | 0.500 | 0.513 |
| Std predicted prob | 0.002 | 0.011 | 0.008 | 0.013 |
| ECE | 0.045 | 0.020 | 0.011 | 0.002 |
| Brier score | 0.252 | 0.250 | 0.250 | 0.250 |
| % predictions > 0.55 | 0% | 7.8% | 0% | 0.2% |

After calibration the model produces predictions above 0.55 for the first time.
At the 0.55 threshold in 2025, hit rate is 90% on 10 predictions — too few for
statistical significance but directionally encouraging.

### Signal quality (LightGBM calibrated, test 2025)
| Metric | Value |
|--------|-------|
| Spearman IC | 0.016 (p=0.25) |
| Monthly IC mean | 0.010 |
| Months IC positive | 58% |
| Q5-Q1 return spread | +26.8 bps/day |

Monthly IC is lumpy — Dec 2025 (IC=0.120, p=0.012) drives most of the annual signal.

### Long-short portfolio (LightGBM calibrated, top-2 / bottom-2 daily)
| Year | Ann. Return | Sharpe | Max Drawdown | Win Rate | t-stat |
|------|-------------|--------|--------------|----------|--------|
| 2024 | -48% | -0.63 | -69% | 48.4% | -0.63 |
| 2025 | +30% | +0.66 | -36% | 52.4% | 0.66 |

2025 Sharpe improved to 0.66 after calibration (vs 0.58 raw) and the Q5-Q1 spread
turned positive (+26.8 bps). 2024 remains unprofitable — the model's 2024 rankings
actively hurt the long-short portfolio. Neither year is statistically significant.

### Conditional IC (LightGBM calibrated, test 2025)
| Condition | IC | p |
|-----------|-----|---|
| High abnormal mentions | +0.033 | 0.099 |
| Low abnormal mentions | -0.001 | 0.946 |
| High volatility | +0.030 | 0.137 |
| Low volatility | -0.006 | 0.775 |

The signal concentrates in high-mention and high-volatility regimes — consistent
with attention-driven price discovery, though not yet significant.

---

## Known Limitations and Next Steps

| Area | Status |
|------|--------|
| Probability calibration | Implemented (Platt scaling on val split) — spread improved but still narrow |
| Ticker expansion | 20 tickers; WSB data has ~100, more expansion possible |
| Sentiment polarity | Only mention counts, no tone scoring |
| Google Trends signal | Not started |
| Viable trading strategy | Not yet — 2024 L/S is deeply negative; 2025 is promising but t=0.66 |

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
