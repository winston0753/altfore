# AltFore — Stock Forecasting with Alternative Data

Predicts stock returns using ML models augmented with alternative data (Reddit sentiment,
search trends). Emphasis on out-of-sample performance and economically significant
trading strategies.

## Stack
- Python 3.11
- Key deps: yfinance, curl_cffi, pandas, matplotlib (pinned in requirements.txt)

## Commands
- `pip install -r requirements.txt` — install deps
- `python scripts/build_reddit_mentions.py` — build reddit_mentions.csv + reddit_mentions_daily.csv
- `python scripts/build_model_dataset.py` — build prices_daily.csv + model_dataset.csv
- `python scripts/plot_returns_vs_mentions.py --output dataset/returns_vs_mentions.png`
- `python scripts/plot_ticker_prices.py --ticker ORCL --output dataset/orcl_prices.png`

## Architecture
- `scripts/` — stable CLI entrypoints, thin wrappers only
- `src/altfore/ingest/` — data fetching and parsing
- `src/altfore/pipeline/` — dataset construction and merging
- `src/altfore/visualization/` — plotting utilities
- `dataset/` — generated outputs, do not hand-edit

## Dataset Schema
- `reddit_mentions.csv` — one row per post-ticker pair
- `reddit_mentions_daily.csv` — daily mention counts by date, subreddit, ticker
- `prices_daily.csv` — daily OHLCV by date, ticker
- `model_dataset.csv` — prices + return targets + Reddit features

## Known Interim State
- Reddit merge is currently static (ticker-only join, no date dimension)
- Reddit lag/rolling columns are constant per ticker — not time-aware
- Next milestone: move to full date,ticker Reddit features for proper time-series merge

## Conventions
- Keep scripts/ as thin CLI wrappers — logic belongs in src/altfore/
- Matplotlib: non-GUI backend, cache at .matplotlib_cache
- New alternative data sources go in src/altfore/ingest/
- New model/pipeline logic goes in src/altfore/pipeline/

## Project Goals (for brainstorming context)
- Out-of-sample ML return forecasting
- Alternative data: sentiment, search trends, Reddit mentions
- Economic significance: evaluate via trading strategies, not just model metrics
- Expanding beyond Reddit to other signal sources

## Do Not
- Edit files under dataset/ directly
- Break the scripts/ → src/altfore/ wrapper pattern
- Use GUI matplotlib backends