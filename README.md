# AltFore: Stock Forecasting with Alternative Data

AltFore predicts next-day stock return direction using machine learning models
augmented with Reddit (WallStreetBets) mention data and Google Trends search
interest. The project emphasises out-of-sample evaluation and economic
significance via trading strategies.

## Project Goals

- Out-of-sample ML return forecasting on daily stock data
- Alternative data signals: Reddit mention volume, mention momentum, abnormal attention, Google Trends search interest
- Economic significance: evaluate via long-short portfolios and quintile analysis, not just model metrics
- Expanding to additional signal sources (sentiment scoring)

---

## Data

### Price data
Daily OHLCV downloaded from Yahoo Finance via `yfinance` for 2021–2025.

### WSB mention data
Historical daily mention counts from WallStreetBets, stored in `wsb_data/` as one CSV per year
(2021–2025). Each file covers ~100 tickers ranked by annual mention volume.

### Google Trends data
Weekly US search interest (0–100, Google-normalised) fetched via the Google Trends API for
2021–2025. Requests are batched in groups of 5 tickers (API limit); each batch is normalised
independently, which is fine because all derived features are z-scored within-ticker anyway.
Weekly values are forward-filled to daily to align with the price panel.

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

### Google Trends features (within-ticker normalised)
| Feature | Description |
|---------|-------------|
| `trends_log` | log1p of forward-filled weekly search interest (lagged 1 day) |
| `trends_abnormal` | Z-score of `trends_log` vs 30-day rolling mean/std |
| `trends_chg_5d` | Pct change of 7-day smoothed `trends_log` vs 5 days ago (week-over-week search momentum) |
| `trends_mentions_divergence` | `trends_abnormal − mentions_abnormal`: positive = search spike without Reddit buzz; negative = Reddit buzz without search interest |

`trends_mentions_divergence` is the novel cross-signal feature. A large positive value
suggests quiet accumulation (search attention without social noise); a large negative
suggests an echo chamber confined to WSB.

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

### Model comparison — test AUC (20-ticker universe, 14 features)

| Model | 2024 (raw) | 2025 (raw) | Mean |
|-------|-----------|-----------|------|
| Extra Trees | 0.498 | **0.511** | **0.504** |
| LightGBM | 0.498 | 0.509 | 0.504 |
| Random Forest | 0.491 | 0.511 | 0.501 |
| Dummy baseline | 0.500 | 0.500 | 0.500 |
| Logistic L1 | 0.497 | 0.498 | 0.498 |

AUC is largely unchanged from the 10-feature baseline — consistent with the view that
directional accuracy near the efficient market boundary is hard to move. The portfolio-level
improvement (below) comes from better ranking, not higher raw accuracy.

### Feature importance (LightGBM, 2022–2023 → 2025 window)
`return_1d` dominates. All five mention features register non-zero importance after
within-ticker normalisation. Among the four new trends features, `trends_abnormal` and
`trends_mentions_divergence` score comparably to the mention momentum features;
`trends_log` and `trends_chg_5d` carry lower but non-zero weight.

### Probability calibration (LightGBM)

Raw predictions cluster in a narrow band — the model can rank but produces no actionable
confidence scores. Platt scaling (sigmoid calibration fitted on the val split via
`CalibratedClassifierCV(FrozenEstimator(...))`) corrects this.

| Metric | Raw (2024) | Calibrated (2024) | Raw (2025) | Calibrated (2025) |
|--------|-----------|------------------|-----------|------------------|
| Mean predicted prob | 0.467 | 0.529 | 0.500 | 0.513 |
| Std predicted prob | 0.004 | 0.005 | 0.008 | 0.011 |
| ECE | 0.046 | 0.016 | 0.011 | 0.007 |
| Brier score | 0.252 | 0.250 | 0.250 | 0.250 |
| % predictions > 0.55 | 0% | 0% | 0% | 0% |

Calibration spread is narrower with the expanded feature set — the model no longer
produces predictions above 0.55. Ranking quality improved (see portfolio metrics below)
but high-confidence point predictions are no longer available.

### Signal quality (LightGBM calibrated, test 2025)
| Metric | Value |
|--------|-------|
| Spearman IC | 0.016 (p=0.25) |
| Monthly IC mean | +0.014 |
| Months IC positive | 58% |
| Q5-Q1 return spread | +16.8 bps/day |

Monthly IC is more evenly distributed across H2 2025 compared to the prior setup
(which was concentrated in Dec 2025 alone).

### Long-short portfolio (LightGBM calibrated, top-2 / bottom-2 daily)

| Year | Ann. Return | Sharpe | Max Drawdown | Win Rate | t-stat |
|------|-------------|--------|--------------|----------|--------|
| 2024 | **+38%** | **+0.976** | −36% | 52.0% | 0.98 |
| 2025 | **+49%** | **+1.034** | −30% | 53.2% | 1.03 |

This is a material improvement over the 10-feature baseline (2024: −48% / Sharpe −0.63;
2025: +30% / Sharpe +0.66). 2024 flipped from deeply negative to solidly positive.
Both years now have Sharpe near 1.0 and t-statistics approaching 1.0. Neither clears
conventional statistical significance (t > 2), but both years now point in the same
direction — an important consistency check.

### Conditional IC (LightGBM calibrated, test 2025)
| Condition | IC | p |
|-----------|-----|---|
| High abnormal mentions | +0.032 | 0.109 |
| Low abnormal mentions | −0.000 | 0.989 |
| High volatility | +0.026 | 0.195 |
| Low volatility | −0.007 | 0.721 |

The signal concentration in high-mention and high-volatility regimes persists and is
slightly stronger than the pre-trends baseline, consistent with attention-driven price
discovery. Flat-to-negative IC in low-mention / low-volatility regimes confirms the
model has no edge in quiet periods.

---

## Known Limitations and Next Steps

| Area | Status |
|------|--------|
| Probability calibration | Implemented — ECE improved, but calibrated spread is now too narrow for threshold-based strategies (0% of predictions > 0.55) |
| Ticker expansion | 20 tickers; WSB data has ~100, more expansion possible |
| Sentiment polarity | Only mention counts, no tone scoring |
| Google Trends granularity | Weekly only for multi-year windows; daily data requires overlapping ~90-day chunks and normalisation stitching |
| Statistical significance | Best t-stat is 1.03 (2025) — directionally consistent but not significant at p < 0.05 |

---

## Architecture

```
scripts/          — thin CLI wrappers only
src/altfore/
  ingest/         — Reddit RSS fetching; Google Trends fetching
  pipeline/       — dataset construction (prices + WSB mentions + Trends)
  modeling/       — feature engineering, model training, evaluation metrics
  visualization/  — plotting utilities
wsb_data/         — raw WSB mention CSVs (2021–2025, not hand-edited)
dataset/          — generated outputs (not hand-edited)
```

### Key generated files
| File | Description |
|------|-------------|
| `dataset/trends_daily.csv` | Weekly Google Trends search interest, long format (date, ticker, trends_interest) |
| `dataset/wsb_model_dataset.csv` | 25,031-row price + mentions + trends panel |
| `dataset/model_comparison.csv` | AUC / IC / accuracy per model × split |
| `dataset/deep_eval.csv` | Brier, quintile spread, L/S metrics per split |
| `dataset/feature_importance.csv` | Feature importances per model × split |
| `dataset/ls_daily_returns.csv` | Daily L/S portfolio returns per test year |

---

## Quick Start

```bash
pip install -r requirements.txt

# Fetch Google Trends search interest (run once; ~30s, 4 API requests)
python scripts/build_trends_dataset.py

# Rebuild the WSB price + mentions + trends dataset
python scripts/build_wsb_dataset.py

# Train and evaluate all models
python scripts/train_model.py
```
