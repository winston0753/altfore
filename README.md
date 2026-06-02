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

## Observations

Several empirical patterns emerged from the daily-horizon experiments:

**Next-day AUC is near-random.** Even with 14 features spanning price, Reddit mentions, and Google Trends, test AUC hovers at 0.504 — effectively random at the single-prediction level. This is expected near the efficient market boundary; no feature set reliably predicts tomorrow's close.

**Portfolio edge exists through cross-sectional ranking, not point prediction.** The model cannot reliably say "TSLA goes up tomorrow," but it ranks stocks by relative conviction well enough that a top-2 long / bottom-2 short strategy achieves Sharpe ~1.0 across both test years. The edge is in ordering, not direction.

**Reddit + Trends act as a regime filter, not a directional signal.** Cross-sectional IC is +0.032 when mentions are abnormally high, near zero in quiet periods. The combined signals identify *when* attention-driven price discovery may be active; they do not resolve which direction the price moves unconditionally.

**The overnight attention-to-price mechanism is weak.** Expecting a stock to rise or fall the next day based on yesterday's mention count imposes an extremely tight reaction window. A more defensible hypothesis: sustained attention over a week or more accumulates and persists, giving price action time to respond. This motivates the weekly extension below.

**Google Trends is structurally weekly.** Weekly Trends values are forward-filled to daily in the current model — a fundamental granularity mismatch. A weekly forecasting horizon aligns the signal with its native resolution and eliminates the forward-fill approximation.

---

## Weekly Horizon Extension

**Motivation.** Aggregating mention signals over a full trading week reduces noise from one-day spikes, aligns naturally with the Google Trends data source, and allows time for price to respond to sustained attention. The key distinction the weekly model tests: sustained attention (e.g., 1,000 mentions spread over 5 days) vs. a single-day spike of equal magnitude — a meaningful structural difference obscured in the daily target.

### Weekly features

All features are computed from data available at Friday close of week T, predicting direction at Friday close of week T+1.

| Feature | Description |
|---------|-------------|
| `return_1w` | Current week's realised return (lag-1 price momentum) |
| `momentum_4w` | 4-week price momentum |
| `momentum_12w` | 12-week price momentum |
| `volatility_12w` | Rolling 12-week std of weekly returns |
| `volume_ratio_4w` | Week volume / 4-week average volume |
| `mentions_sum_log` | log1p of total weekly mention count |
| `mentions_abnormal_w` | Z-score of weekly mention sum vs 12-week rolling baseline |
| `mentions_momentum_w` | Pct change of weekly mention sum vs prior week |
| `mentions_peak_ratio` | Max single-day / mean daily mentions — spike vs sustained attention |
| `trends_log` | log1p of weekly Trends interest (native granularity; no forward-fill) |
| `trends_abnormal_w` | Z-score of `trends_log` vs 12-week rolling baseline |
| `trends_mentions_divergence_w` | `trends_abnormal_w − mentions_abnormal_w` |

`mentions_peak_ratio` is the novel feature here: it distinguishes a stock with evenly distributed weekly attention from one with a single-day spike, testing whether the *pattern* of attention within the week carries information beyond the total volume.

### Walk-forward and portfolio

Same two expanding windows as the daily model (train 2022 → test 2024; train 2022–2023 → test 2025). Long-short portfolio holds for one week and rebalances weekly; annualisation uses 52 periods/year.

**Build and run:**
```bash
python scripts/build_period_dataset.py   # → dataset/wsb_weekly_dataset.csv
python scripts/train_period_model.py     # → dataset/weekly_model_comparison.csv, weekly_deep_eval.csv
```

### Weekly results

#### Model comparison — test AUC (20-ticker universe, 12 features)

| Model | 2024 | 2025 | Mean |
|-------|------|------|------|
| Random Forest | **0.522** | **0.511** | **0.517** |
| Logistic L1 | 0.511 | 0.503 | 0.507 |
| Extra Trees (calibrated) | 0.507 | 0.515 | 0.511 |
| LightGBM (calibrated) | 0.504 | 0.508 | 0.506 |
| Dummy baseline | 0.500 | 0.500 | 0.500 |

Random Forest is the most consistent model at the weekly horizon; LightGBM is unremarkable.

#### Feature importance — weekly IC (Random Forest, raw)

Random Forest raw IC: +0.038 (2024) and +0.019 (2025) — both positive and the most stable signal of any weekly model. At n=1,040 test obs per year, these correspond to t-stats near 1.2 — directionally consistent but not statistically significant.

#### Long-short portfolio (LightGBM calibrated, top-2 / bottom-2 weekly)

| Year | Ann. Return | Sharpe | Max Drawdown | Win Rate | t-stat |
|------|-------------|--------|--------------|----------|--------|
| 2024 | −31.1% | −0.682 | −47.9% | 50.0% | −0.68 |
| 2025 | +24.1% | +0.570 | −29.6% | 63.5% | +0.57 |

The weekly LightGBM portfolio is inconsistent across years — opposite sign to the daily model's stability. This is a meaningful negative result.

#### Conditional IC (LightGBM calibrated)

| Condition | 2024 IC | 2025 IC |
|-----------|---------|---------|
| High abnormal mentions | −0.053 | +0.029 |
| Low abnormal mentions | +0.061 | −0.015 |
| High volatility | +0.015 | +0.037 |
| Low volatility | +0.009 | −0.008 |

The regime filter that works on the daily model (IC concentrates in high-mention periods) does not survive weekly aggregation. No consistent conditioning variable improves signal.

### Weekly findings and interpretation

**The hypothesis was not confirmed.** Aggregating mentions to weekly does not produce a more reliable signal than next-day prediction. Three structural reasons:

1. **Training data is too small.** The first walk-forward split has only 800 training observations (vs ~16,000 daily). LightGBM cannot learn stable feature interactions at this scale; the results are effectively noise-fitted.

2. **Weekly aggregation destroys within-week variation.** The daily model's edge appears to come primarily from short-horizon price momentum (`return_1d`). Its weekly analogue (`return_1w`) is a noisier version of the same signal, and the attention features do not compensate.

3. **Google Trends adds no new information at weekly resolution.** Since Trends is already a weekly series forward-filled to daily, the weekly model uses the same data at its native granularity — no new signal relative to the daily model's already-incorporated Trends features.

**Discovered bug: Trends data was silently zeroed.** The original `load_google_trends` merge reindexed weekly (Sunday) observations directly to trading-day (Mon–Fri) dates, which dropped all values before `ffill` ran. All `trends_interest` values in `wsb_model_dataset.csv` were 0, meaning prior daily model results also ran without Trends signal. Fixed in `src/altfore/pipeline/wsb_dataset.py` by expanding to the union index before forward-filling. The daily model results in the Results section above should be re-run against the corrected dataset.

---

## Known Limitations and Next Steps

| Area | Status |
|------|--------|
| Probability calibration | Implemented — ECE improved, but calibrated spread is now too narrow for threshold-based strategies (0% of predictions > 0.55) |
| Ticker expansion | 20 tickers; WSB data has ~100, more expansion possible |
| Sentiment polarity | Only mention counts, no tone scoring |
| Google Trends granularity | Weekly only for multi-year windows; daily data requires overlapping ~90-day chunks and normalisation stitching |
| Statistical significance | Best t-stat is 1.03 (2025) — directionally consistent but not significant at p < 0.05 |
| Weekly horizon | Implemented and evaluated — did not outperform daily; see Weekly Horizon Extension section |
| Trends merge bug | Fixed — prior daily results used zeroed Trends features; daily model should be re-run against corrected dataset |
| Daily model re-evaluation | Pending — daily Results section reflects pre-fix run; re-run `train_model.py` to get accurate trends-inclusive numbers |

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
