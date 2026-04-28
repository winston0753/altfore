# AltFore: Stock Forecasting using Alternative Inputs

AltFore builds a daily modeling panel from:

- Reddit ticker mentions (RSS from selected subreddits)
- Yahoo Finance daily prices (roughly one calendar year)

The current pipeline is intentionally lightweight and script-driven.

## Current Architecture

The repo now uses a thin-wrapper pattern:

- `scripts/` = stable CLI entrypoints
- `src/altfore/` = reusable implementation modules

### Source modules

- `src/altfore/ingest/reddit_mentions.py`
  - Fetches subreddit RSS feeds
  - Extracts ticker tokens with regex + stoplist
  - Writes mention-level and daily aggregate CSVs
- `src/altfore/pipeline/model_dataset.py`
  - Loads Reddit daily data
  - Normalizes Reddit schema and aggregates by ticker (current static-ticker mode)
  - Selects tickers and downloads/cache-reuses Yahoo prices
  - Builds returns/labels and merges Reddit features onto prices
  - Validates and writes output datasets
- `src/altfore/visualization/returns_vs_mentions.py`
  - Plots one-year return vs Reddit total mentions by ticker
- `src/altfore/visualization/ticker_prices.py`
  - Plots OHLC range + close and optional volume for one ticker

## Scripts (CLI Entrypoints)

| Script | Role |
|--------|------|
| `scripts/build_reddit_mentions.py` | Build `reddit_mentions.csv` and `reddit_mentions_daily.csv` |
| `scripts/build_model_dataset.py` | Build `prices_daily.csv` and `model_dataset.csv` |
| `scripts/plot_returns_vs_mentions.py` | Save ticker return vs mention chart PNG |
| `scripts/plot_ticker_prices.py` | Plot one ticker from `prices_daily.csv` |

## Dataset Outputs

`dataset/` contains generated tables and optional figures:

- `reddit_mentions.csv` — one row per post-ticker pair (with post metadata)
- `reddit_mentions_daily.csv` — daily mention counts by `date`, `subreddit`, `ticker`
- `prices_daily.csv` — cleaned daily OHLCV by `date`, `ticker`
- `model_dataset.csv` — prices + return targets + Reddit features

## Current Modeling Spec

`build_model_dataset.py` currently uses a static Reddit merge mode:

- Reddit rows are normalized and aggregated to one row per ticker
- Price rows are daily by `date,ticker`
- Merge is left join on `ticker` only
- Derived Reddit lag/rolling-style columns are constant per ticker in this mode

This is a known interim state before moving to fully time-aware `date,ticker` Reddit features.

## Quick Start

Use Python 3.11.

```bash
pip install -r requirements.txt
python scripts/build_reddit_mentions.py
python scripts/build_model_dataset.py
```

### Plot examples

```bash
python scripts/plot_returns_vs_mentions.py --output dataset/returns_vs_mentions.png
python scripts/plot_ticker_prices.py --ticker ORCL --output dataset/orcl_prices.png
```

## Notes

- Matplotlib scripts use a non-GUI backend and default cache directory at `.matplotlib_cache`.
- `yfinance` and `curl_cffi` are pinned in `requirements.txt` for compatibility.
