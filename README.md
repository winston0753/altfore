# AltFore: Stock Forecasting using Alternative Inputs

AltFore builds a **daily panel** of **Yahoo prices** (~1 year) plus **Reddit mention totals** (from subreddit RSS) for modeling.

## Dataset

`dataset/` holds generated tables and optional figures:

- `reddit_mentions.csv` — one row per post–ticker pair (mentions + metadata)
- `reddit_mentions_daily.csv` — daily counts by date, subreddit, ticker
- `prices_daily.csv` — OHLCV after cleaning
- `model_dataset.csv` — prices + returns + Reddit features for modeling

## Scripts

| Script | Role |
|--------|------|
| `scripts/build_reddit_mentions.py` | Fetch RSS, parse tickers from title/summary, write mention + daily CSVs |
| `scripts/build_model_dataset.py` | Build prices + merged model table (see pipeline below) |
| `scripts/plot_ticker_prices.py` | Plot OHLC/volume for one symbol from `prices_daily.csv` (`--ticker`, `--output`) |

### `build_model_dataset.py` — main functions (in order)

- **`load_reddit_daily`** — read `reddit_mentions_daily.csv`
- **`normalize_reddit_schema`** — validate columns, sum counts per **ticker** (no date on Reddit side for the merge)
- **`get_unique_tickers`** — filter symbols and cap how many tickers get Yahoo pulls
- **`load_price_cache` / `select_tickers_needing_download`** — reuse `prices_daily.csv` when it already covers the window
- **`download_price_data`** — batched Yahoo download with retries/backoff
- **`drop_incomplete_price_rows`** — drop rows with any null OHLCV
- **`build_price_features`** — `return_1d`, forward returns, `direction_fwd_1d`
- **`merge_datasets`** — left-join Reddit totals on **ticker**
- **`build_reddit_features`** — lag / MA / abnormal columns (static-ticker mode)
- **`validate_final_dataset` / `save_outputs` / `print_diagnostics`** — checks, CSV writes, log summary

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_reddit_mentions.py
python scripts/build_model_dataset.py
```

Optional plot (saves PNG; use `--output` because the script uses a non-GUI backend):

```bash
python scripts/plot_ticker_prices.py --ticker ORCL --output dataset/orcl_prices.png
```

Use **Python 3.11** with the pinned `yfinance` / `curl_cffi` versions in `requirements.txt` if you hit import issues on older Pythons.
