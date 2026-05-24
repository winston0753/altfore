# AltFore — Stock Forecasting with Alternative Data

Predicts next-day stock return direction using ML (primarily LightGBM) augmented with
Reddit mention signals. Emphasis on walk-forward out-of-sample evaluation and economic
significance via long–short and quintile-style metrics. README.md is the canonical write-up
for features, model setup, results tables, and limitations; this file orients agents on
repo layout and **two distinct mention data paths** (do not conflate them).

## Stack
- Python 3.11
- Key deps: yfinance, curl_cffi, pandas, matplotlib, LightGBM / scikit-learn (pinned in requirements.txt)

## Two data paths (WSB vs RSS)

**Primary (WSB panel — matches README and `train_model.py`)**  
- Raw inputs: `wsb_data/wallstreetbets_*.csv` (historical daily mention counts, ~100 tickers ranked by year; curated 20-ticker universe in code).  
- Build: `python scripts/build_wsb_dataset.py` → `dataset/wsb_model_dataset.csv` (prices merged on `date, ticker` with daily mentions).  
- Modeling: `python scripts/train_model.py` reads `wsb_model_dataset.csv`; feature engineering, walk-forward splits, calibration, and metrics live under `src/altfore/modeling/`.

**Alternate (live Reddit RSS — exploratory / not the README benchmark)**  
- Ingest: `src/altfore/ingest/reddit_mentions.py` fetches WSB / stocks / investing `.rss` feeds.  
- Build: `python scripts/build_reddit_mentions.py` → `dataset/reddit_mentions.csv` + `dataset/reddit_mentions_daily.csv`.  
- Merge: `python scripts/build_model_dataset.py` joins daily Reddit aggregates to Yahoo prices → `dataset/prices_daily.csv` + `dataset/model_dataset.csv` (rolling ~1y price window from “today”, different from the fixed 2021–2025 WSB panel).

## Commands

**Main workflow**
- `pip install -r requirements.txt` — install deps
- `python scripts/build_wsb_dataset.py` — `wsb_data/` → `dataset/wsb_model_dataset.csv`
- `python scripts/train_model.py` — train / evaluate models; writes comparison CSVs under `dataset/`

**RSS path (optional)**
- `python scripts/build_reddit_mentions.py` — RSS → `reddit_mentions*.csv`
- `python scripts/build_model_dataset.py` — `reddit_mentions_daily.csv` + Yahoo → `model_dataset.csv`

**Plotting (examples)**
- `python scripts/plot_wsb_ticker.py --ticker GME --output dataset/gme_wsb.png`
- `python scripts/plot_ticker_prices.py --ticker ORCL --output dataset/orcl_prices.png`  
Other helpers: `plot_ticker_mentions.py`, `plot_returns_vs_mentions.py`.

## Architecture
- `scripts/` — stable CLI entrypoints, thin wrappers only
- `src/altfore/ingest/` — Reddit RSS fetching and parsing (`reddit_mentions` path)
- `src/altfore/pipeline/` — dataset construction (WSB panel + shared price helpers; RSS-backed `model_dataset` merge)
- `src/altfore/modeling/` — WSB feature engineering, training, evaluation metrics
- `src/altfore/visualization/` — plotting utilities
- `wsb_data/` — raw WSB mention CSVs (do not hand-edit)
- `dataset/` — generated outputs (do not hand-edit)

## Dataset / outputs (high level)

**WSB pipeline** — see README “Key generated files” for full list (`wsb_model_dataset.csv`, `model_comparison.csv`, `deep_eval.csv`, etc.).  
**RSS pipeline** — `reddit_mentions.csv` (post–ticker rows), `reddit_mentions_daily.csv` (daily by date, subreddit, ticker), `prices_daily.csv`, `model_dataset.csv`.

## Conventions
- Keep `scripts/` as thin CLI wrappers — logic belongs in `src/altfore/`
- Matplotlib: non-GUI backend, cache at `.matplotlib_cache`
- New alternative data sources go in `src/altfore/ingest/`
- New dataset merge logic goes in `src/altfore/pipeline/`; new training/eval code in `src/altfore/modeling/`

## Project goals (agent context)
- Out-of-sample ML on daily data; primary signals today are WSB mention volume and derived within-ticker mention features (no sentiment polarity in-repo yet)
- Economic significance via portfolios and quintiles, not only AUC
- Future: search trends, richer sentiment (README “Known Limitations”)

## Do Not
- Edit files under `dataset/` or `wsb_data/` directly
- Break the `scripts/` → `src/altfore/` wrapper pattern
- Use GUI matplotlib backends
- Assume RSS outputs feed `train_model.py` — training is wired to the WSB dataset unless code is changed
