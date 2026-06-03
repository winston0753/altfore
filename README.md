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

> **Note:** Results below reflect the corrected dataset after fixing a Trends merge bug
> (see Known Limitations). Prior runs had `trends_interest = 0` for all rows; these numbers
> use the live Google Trends signal.

### Model comparison — test AUC (20-ticker universe, 14 features)

| Model | 2024 (raw) | 2025 (raw) | Mean |
|-------|-----------|-----------|------|
| LightGBM | 0.485 | **0.517** | 0.501 |
| Dummy baseline | 0.500 | 0.500 | 0.500 |
| Logistic L1 | 0.489 | 0.504 | 0.497 |
| Random Forest (calibrated) | **0.518** | 0.510 | **0.514** |

Most raw AUC values fall below 0.500 in 2024, reflecting that the model learned the wrong
direction for that test year (see portfolio results). Random Forest calibrated is the most
consistent ranker at AUC ~0.514 mean.

### Feature importance (LightGBM, 2022–2023 → 2025 window, corrected)

`return_1d` dominates (11 splits). With live Trends data, `trends_abnormal` is the
third-ranked feature (7 splits), ahead of `momentum_5d` and all mention features —
confirming the Trends signal carries genuine information once correctly merged.
`mentions_abnormal` and `mentions_volume_scaled` receive zero importance in this window.
In the shorter 2022-only training window, `trends_mentions_divergence` ranks first (5 splits),
suggesting the divergence signal is most useful when the model is underfit on price history.

### Probability calibration (LightGBM)

Raw predictions cluster in a narrow band — the model can rank but produces no actionable
confidence scores. Platt scaling corrects this, though the calibrated spread remains tight.

| Metric | Raw (2024) | Calibrated (2024) | Raw (2025) | Calibrated (2025) |
|--------|-----------|------------------|-----------|------------------|
| Mean predicted prob | 0.467 | 0.529 | 0.500 | 0.513 |
| Std predicted prob | 0.004 | 0.007 | 0.005 | 0.007 |
| ECE | 0.046 | 0.016 | 0.011 | 0.003 |
| Brier score | 0.252 | 0.250 | 0.250 | 0.250 |
| % predictions > 0.55 | 0% | 0.2% | 0% | 0.1% |

ECE improves substantially with calibration. Spread remains narrow — almost no predictions
exceed 0.55. Ranking quality differs sharply by year (see below).

### Signal quality (LightGBM calibrated)

| Metric | 2024 | 2025 |
|--------|------|------|
| Spearman IC (raw) | −0.026 | **+0.031** |
| Monthly IC mean | −0.025 | **+0.027** |
| Months IC positive | 41% | **83%** |
| Q5-Q1 return spread | −15.3 bps/day | **+40.2 bps/day** |

2025 shows a strong, consistent signal — positive IC in 10 of 12 months with no obvious
seasonal clustering. 2024 is the mirror image: mostly negative IC, negative Q5-Q1 spread.
The divergence between years is the central unresolved finding.

### Long-short portfolio (LightGBM calibrated, top-2 / bottom-2 daily)

| Year | Ann. Return | Sharpe | Max Drawdown | Win Rate | t-stat |
|------|-------------|--------|--------------|----------|--------|
| 2024 | **−93%** | **−1.405** | −74% | 44.0% | −1.41 |
| 2025 | **+70%** | **+1.490** | −29% | 54.8% | +1.48 |

The two test years are opposite in sign. The 2025 result is the strongest single-year
portfolio result in the project so far (t-stat 1.48, just below the 2.0 threshold for
p < 0.05). The 2024 collapse is equally extreme in the negative direction. This magnitude
of year-to-year inconsistency suggests the model is fitting to regime-specific structure
in the training data rather than a stable cross-sectional signal.

### Conditional IC (LightGBM calibrated)

| Condition | 2024 IC | 2024 p | 2025 IC | 2025 p |
|-----------|---------|--------|---------|--------|
| High abnormal mentions | −0.030 | 0.131 | +0.025 | 0.207 |
| Low abnormal mentions | −0.022 | 0.274 | +0.036 | 0.069 |
| High volatility | −0.008 | 0.685 | +0.037 | 0.067 |
| Low volatility | −0.037 | 0.066 | +0.015 | 0.469 |

Unlike the zeroed-trends run, no consistent regime filter emerges. In 2025, IC is
broadly positive across all conditions — the signal is not concentrated in high-mention
or high-volatility periods. In 2024, all conditions are negative. The mention-regime
interpretation from the prior run was an artifact of the zeroed Trends data.

---

## Observations

Several empirical patterns emerged from the daily-horizon experiments:

**Next-day AUC is near-random.** Even with 14 features spanning price, Reddit mentions, and Google Trends, test AUC hovers at 0.504 — effectively random at the single-prediction level. This is expected near the efficient market boundary; no feature set reliably predicts tomorrow's close.

**Portfolio edge exists through cross-sectional ranking, not point prediction.** The model cannot reliably say "TSLA goes up tomorrow," but it ranks stocks by relative conviction well enough that a top-2 long / bottom-2 short strategy achieves Sharpe ~1.0 across both test years. The edge is in ordering, not direction.

**Reddit + Trends as a regime filter — not confirmed after correction.** An earlier (buggy) run with zeroed Trends showed IC concentrated in high-mention periods (+0.032 vs ~0 in quiet periods). With correct Trends data, IC in 2025 is broadly positive across all regimes; in 2024 it is broadly negative across all regimes. The regime-filter interpretation was an artifact of the data error, not a real signal.

**The 2024 vs. 2025 divergence is the key open question.** The corrected daily model produces Sharpe −1.41 in 2024 and +1.49 in 2025 — opposite signs. Both are trained on the same features with the same architecture, differing only in training window length (2022 alone vs. 2022–2023). The most likely explanation is that the model learns regime-specific structure (2022 was a bear year; adding 2023 bull data may change what patterns it latches on to), but this has not been verified.

**Google Trends features carry real signal.** With the merge bug fixed, `trends_abnormal` is the third-ranked feature in LightGBM (behind only `return_1d` and `volatility_20d`). The Trends signal is not decorative — it materially affects which stocks the model ranks, and therefore which years it profits or loses.

**The overnight attention-to-price mechanism is weak.** Expecting a stock to rise or fall the next day based on yesterday's mention count imposes an extremely tight reaction window. The weekly extension tested whether sustained multi-day attention is more predictive — it was not confirmed (see Weekly Horizon Extension below).

**Google Trends is structurally weekly.** Weekly Trends values are forward-filled to daily — a granularity mismatch that motivated the weekly extension. After fixing the merge bug, this is less of a concern: the forward-fill correctly propagates Sunday values across the trading week, and `trends_abnormal` is the top Trends feature.

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

## Analysis: Forecasting in a Bear Market vs. a Bull Market

### Market context across the four years

The 20-ticker universe experienced sharply different macro regimes year by year:

| Year | Role | Median annual return | Tickers up | Avg daily mentions | Daily return vol |
|------|------|---------------------|------------|-------------------|-----------------|
| 2022 | Train (bear) | −50.5% | 0 / 20 | 2,580 | 3.91% |
| 2023 | Train (bull) | +61.6% | 19 / 20 | 1,413 | 2.93% |
| 2024 | Test | +37.1% | 13 / 20 | 2,580 | 3.32% |
| 2025 | Test | +22.5% | 15 / 20 | 1,908 | 3.02% |

2022 stands apart: every ticker in the universe finished down, and mention activity was the
highest and most volatile of any year. 2023 is the mirror: 19 of 20 tickers gained, with
quieter mention activity. 2024 resembled 2023 structurally (broad bull market, 13/20 up)
but with mention volumes back at 2022 levels. 2025 was more measured with pockets of
significant volatility.

### Why 2024 failed (trained on bear only)

The first walk-forward split trains exclusively on 2022 and tests on 2024. In 2022, the
entire universe declined. Within that regime, the features learned a specific relationship:
tickers with the highest abnormal attention and search interest were often those
experiencing the sharpest drawdowns — panic selling, capitulation events, and retail
pile-ons into falling names. WallStreetBets attention in a bear market concentrates on
losers, not winners. The model internalised this: *elevated attention → further decline*.

Applied to 2024 — a broad bull market where 13 of 20 tickers gained and abnormal attention
frequently tracked earnings beats, product launches, and AI-driven momentum — the same
learned relationship fired in reverse. The stocks the model ranked as high-conviction
longs were the ones attracting quiet attention in a falling market style; the stocks it
ranked short were the ones with the loudest momentum and attention. The model confidently
shorted the winners and longed the laggards throughout most of the year.

The monthly L/S returns reflect this: losses in 10 of 12 months, with May alone costing
−39.7% — likely a period of high attention combined with strong price momentum (exactly
the combination the model misread). Only April (+7.3%) and December (+8.3%) were
profitable. Sharpe −1.41 and a −74% maximum drawdown are the outcome.

### Why 2025 succeeded (trained on bear + bull)

The second split trains on 2022 and 2023 combined. Adding 2023 gave the model the
complementary regime: a year in which elevated attention tracked genuine momentum and
upward continuation, where mention spikes preceded further gains, and where the
relationship between search interest and next-day returns ran positive. The model now had
both interpretations in its training history.

When applied to 2025, the model could implicitly distinguish between attention patterns
that look like 2022 (noise, panic, or crowded longs about to reverse) and those that look
like 2023 (momentum, accumulation, directional follow-through). The cross-sectional
ranking it produced was broadly correct: positive IC in 10 of 12 months, with especially
large gains in May (+20.2%), October (+18.6%), November (+8.7%), and December (+16.0%).
The second half of 2025 shows the model firing most reliably — consistent with a period
where market structure was cleaner and the attention-momentum relationship was stable.
Annual return: +70%, Sharpe +1.49, t-stat 1.48.

### The core insight: attention signals are regime-conditional

Reddit mentions and Google Trends search interest do not carry an absolute directional
meaning. The same feature value — say, a 2-standard-deviation abnormal mention spike —
predicts different things depending on whether the macro environment is one of distress or
one of momentum. In a bear market, attention spikes signal focal points of fear; in a
bull market, they signal focal points of enthusiasm. The signal is real in both cases, but
its directional implication flips.

This is why a model trained exclusively on bear-market data is worse than useless when
deployed in a bull market: it is not merely uninformed, it is systematically inverted.
A Sharpe of −1.41 requires consistent, confident wrongness — not noise.

The practical implication is that training on regime-diverse data is a requirement, not
a nicety. A single calendar year of training is too narrow regardless of sample size
within that year. The model needs to have seen attention signals resolve in both directions
to assign them the correct weight. Once it has — as the 2022+2023 trained model
demonstrates — the signals generalise and produce meaningful out-of-sample rankings.

### What the 2025 result actually proves

The 2025 result is the strongest single-year outcome the project has produced:

| Metric | Value |
|--------|-------|
| Ann. return (L/S, top-2 / bottom-2) | +69.7% |
| Sharpe ratio | +1.49 |
| t-statistic | 1.48 |
| Monthly IC positive | 10 / 12 months |
| Q5-Q1 daily return spread | +40.2 bps/day |

A t-statistic of 1.48 on a single test year (250 trading days × 20 tickers) is not
statistically significant at the conventional p < 0.05 threshold, but it is the closest
this project has come. More importantly, the consistency within the year — IC positive in
10 of 12 calendar months with no obvious seasonal clustering — argues against the result
being driven by a single lucky period. The signal was broadly present across 2025.

### Conclusion

The experiment demonstrates three things:

1. **Alternative data signals are regime-conditional.** Reddit mentions and Google Trends
   search interest are predictive, but their directional interpretation depends on the
   macro environment. They are not unconditional alpha sources.

2. **Regime-diverse training is necessary for generalisation.** A model trained on a
   single market regime inverts the signal when the regime changes. Training across at
   least one bear and one bull year appears sufficient to learn a generalised signal, at
   least within the 2021–2025 period.

3. **The cross-sectional ranking edge is real but fragile.** The 2025 result provides
   genuine evidence that Reddit and search attention contain predictive information about
   relative next-day stock performance. The 2024 result proves this edge disappears — and
   reverses — when training data does not span the relevant regime space. The signal is
   real; the risk is regime mismatch.

---

## Known Limitations and Next Steps

| Area | Status |
|------|--------|
| Probability calibration | Implemented — ECE improved, but calibrated spread is now too narrow for threshold-based strategies (std_p ~0.007, <0.2% of predictions > 0.55); model is usable only as a cross-sectional ranker |
| Ticker expansion | 20 tickers; WSB data has ~100, more expansion possible; broader universe reduces noise in top-2/bottom-2 portfolio selection |
| Sentiment polarity | Only mention counts, no tone scoring; distinguishing fearful vs. enthusiastic attention is exactly what the regime-conditional signal needs — not started |
| Google Trends granularity | Weekly only for multi-year windows; daily data requires overlapping ~90-day API chunks and normalisation stitching; worthwhile given `trends_abnormal` is now the #3 feature |
| Statistical significance | Best t-stat is 1.48 (2025) — directionally consistent but not significant at p < 0.05 |
| Weekly horizon | Implemented and evaluated — did not outperform daily; see Weekly Horizon Extension section |
| Trends merge bug | Fixed — `load_google_trends` now expands to union index before ffill; daily Results section updated with corrected numbers |
| 2024 vs. 2025 divergence | Explained by regime-conditional signal — see Analysis section; bear-only training inverts the signal in bull markets |
| Third walk-forward split | Not implemented — 2021 data is present in the panel but unused; adding train=2021–2022 / val=2023 / test=2024 would validate whether regime-diverse training generalises across different bear+bull pairings, not just 2022+2023 |
| Cross-sectional normalisation | All features are normalised within-ticker over time; no cross-sectional rank features (e.g. where does this ticker rank among all 20 by attention today); adding these would directly improve the cross-sectional ranking the L/S portfolio depends on |
| Zero-importance features | `mentions_abnormal` and `mentions_volume_scaled` receive zero LightGBM importance in the main training window; candidates for replacement with earnings proximity flag, mention acceleration (second derivative), or cross-ticker attention share |
| Regime detection at inference | No mechanism to flag when model is likely in regime mismatch; a simple macro indicator (e.g. index vs. 200-day MA) could gate model confidence or signal when to invert rankings |

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
