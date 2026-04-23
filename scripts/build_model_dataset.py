"""Build a unified price + Reddit modeling dataset.

Loads ~one calendar year of daily Yahoo prices (ending today) for Reddit-selected tickers,
merges per-ticker Reddit aggregates, and writes ``dataset/prices_daily.csv`` and
``dataset/model_dataset.csv``.

Usage:
    python scripts/build_model_dataset.py
"""

from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


LOGGER = logging.getLogger(__name__)

REDDIT_REQUIRED_COLUMNS = {"date", "ticker", "post_count", "total_mentions"}
REDDIT_OPTIONAL_COLUMNS = [
    "subreddit",
    "reddit_post_count",
    "reddit_total_mentions",
    "reddit_wsb_post_count",
    "reddit_stocks_post_count",
    "reddit_investing_post_count",
    "reddit_total_score",
    "reddit_avg_score",
    "reddit_total_comments",
    "reddit_avg_comments",
    "reddit_avg_sentiment",
    "reddit_weighted_sentiment",
]
COUNT_LIKE_COLUMNS = [
    "reddit_post_count",
    "reddit_total_mentions",
    "reddit_wsb_post_count",
    "reddit_stocks_post_count",
    "reddit_investing_post_count",
    "reddit_total_score",
    "reddit_total_comments",
]
SENTIMENT_STYLE_COLUMNS = [
    "reddit_avg_score",
    "reddit_avg_comments",
    "reddit_avg_sentiment",
    "reddit_weighted_sentiment",
]

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
PRICE_OUTPUT_COLUMNS = ["date", "ticker", *PRICE_COLUMNS]
PRICE_DOWNLOAD_BATCH_SIZE = 8
PRICE_DOWNLOAD_RETRIES = 5
PRICE_BATCH_SLEEP_SECONDS = 8.0
PRICE_BATCH_SLEEP_JITTER_SECONDS = 2.0
# Daily prices: use this many calendar days ending today (Yahoo download includes extra lookback for returns).
PRICE_HISTORY_LOOKBACK_DAYS = 365
MIN_TOTAL_MENTIONS_PER_TICKER = 2.0
MAX_TICKERS_TO_DOWNLOAD = 40
VALID_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")
ALLOWED_SINGLE_CHAR_TICKERS = {"F", "T", "C", "K", "X", "V"}
MODEL_REQUIRED_COLUMNS = [
    "date",
    "ticker",
    *PRICE_COLUMNS,
    "return_1d",
    "return_fwd_1d",
    "return_fwd_5d",
    "direction_fwd_1d",
    "reddit_post_count",
    "reddit_total_mentions",
    "reddit_mentions_1d_lag",
    "reddit_mentions_3d_ma",
    "reddit_mentions_7d_ma",
    "reddit_abnormal_mentions",
]


def load_reddit_daily(path: Path) -> pd.DataFrame:
    """Load Reddit daily dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Reddit input file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        LOGGER.warning("Reddit input file is empty: %s", path)
    return df


def normalize_reddit_schema(reddit_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Reddit schema and aggregate to one row per ticker (date dropped).

    Count-like fields are summed over all input rows; sentiment-style fields use
    the mean of non-null values per ticker. No time dimension remains in Reddit
    features—each symbol gets a single total or aggregate score.
    """
    missing_required = REDDIT_REQUIRED_COLUMNS - set(reddit_df.columns)
    if missing_required:
        missing_fmt = ", ".join(sorted(missing_required))
        raise ValueError(f"Reddit data missing required columns: {missing_fmt}")

    df = reddit_df.copy()
    # `date` is only used to validate rows; aggregation is not by date.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date"])
    df = df[df["ticker"] != ""]

    if "reddit_post_count" not in df.columns:
        df["reddit_post_count"] = pd.to_numeric(df["post_count"], errors="coerce")
    if "reddit_total_mentions" not in df.columns:
        df["reddit_total_mentions"] = pd.to_numeric(df["total_mentions"], errors="coerce")

    for col in COUNT_LIKE_COLUMNS + SENTIMENT_STYLE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "subreddit" in df.columns:
        subreddit_key = (
            df["subreddit"].astype(str).str.strip().str.lower().replace({"wallstreetbets": "wsb"})
        )
        if "reddit_wsb_post_count" not in df.columns:
            df["reddit_wsb_post_count"] = np.where(
                subreddit_key == "wsb", df["reddit_post_count"], 0
            )
        if "reddit_stocks_post_count" not in df.columns:
            df["reddit_stocks_post_count"] = np.where(
                subreddit_key == "stocks", df["reddit_post_count"], 0
            )
        if "reddit_investing_post_count" not in df.columns:
            df["reddit_investing_post_count"] = np.where(
                subreddit_key == "investing", df["reddit_post_count"], 0
            )

    aggregations: dict[str, str] = {}
    for col in COUNT_LIKE_COLUMNS:
        if col in df.columns:
            aggregations[col] = "sum"
    for col in SENTIMENT_STYLE_COLUMNS:
        if col in df.columns:
            aggregations[col] = "mean"

    if not aggregations:
        raise ValueError("Reddit data has no usable feature columns after normalization")

    grouped = (
        df.groupby("ticker", as_index=False)
        .agg(aggregations)
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return grouped


def get_unique_tickers(reddit_df: pd.DataFrame) -> list[str]:
    """Extract and filter candidate tickers for Yahoo download."""
    if reddit_df.empty:
        return []

    agg = (
        reddit_df.groupby("ticker", as_index=False)["reddit_total_mentions"]
        .sum(min_count=1)
        .rename(columns={"reddit_total_mentions": "mentions_sum"})
    )
    agg["mentions_sum"] = pd.to_numeric(agg["mentions_sum"], errors="coerce").fillna(0.0)

    def _is_valid_symbol(ticker: str) -> bool:
        if not VALID_TICKER_PATTERN.match(ticker):
            return False
        if len(ticker) == 1 and ticker not in ALLOWED_SINGLE_CHAR_TICKERS:
            return False
        return True

    agg = agg[agg["ticker"].astype(str).map(_is_valid_symbol)]
    agg = agg[agg["mentions_sum"] >= MIN_TOTAL_MENTIONS_PER_TICKER]
    agg = agg.sort_values(["mentions_sum", "ticker"], ascending=[False, True]).reset_index(drop=True)

    filtered = agg["ticker"].astype(str).tolist()[:MAX_TICKERS_TO_DOWNLOAD]
    LOGGER.info(
        "Ticker filtering kept %d symbols (min mentions >= %.1f, max tickers = %d)",
        len(filtered),
        MIN_TOTAL_MENTIONS_PER_TICKER,
        MAX_TICKERS_TO_DOWNLOAD,
    )
    return filtered


def _normalize_price_frame(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert yfinance output for one ticker to project schema."""
    if raw_df.empty:
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)

    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return MultiIndex depending on version/config.
        df.columns = [str(col[0]) for col in df.columns]

    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_map)

    missing_cols = [col for col in PRICE_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df["ticker"] = ticker
    df = df[PRICE_OUTPUT_COLUMNS].dropna(subset=["date"])
    return df


def load_price_cache(path: Path) -> pd.DataFrame:
    """Load existing cached prices, if available."""
    if not path.exists():
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)

    cached = pd.read_csv(path)
    expected = set(PRICE_OUTPUT_COLUMNS)
    if not expected.issubset(set(cached.columns)):
        LOGGER.warning("Ignoring price cache with incompatible schema: %s", path)
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)

    cached = cached[PRICE_OUTPUT_COLUMNS].copy()
    cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
    cached["ticker"] = cached["ticker"].astype(str).str.strip().str.upper()
    cached = cached.dropna(subset=["date"])
    return cached


def select_tickers_needing_download(
    tickers: list[str], cache_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[list[str], pd.DataFrame]:
    """Split tickers into cached-sufficient vs needing fresh Yahoo download."""
    if cache_df.empty:
        return tickers, cache_df

    window_start = start_date - pd.Timedelta(days=30)
    window_end = end_date

    in_window = cache_df[
        (cache_df["date"] >= window_start) & (cache_df["date"] <= window_end)
    ].copy()
    if in_window.empty:
        return tickers, in_window

    coverage = in_window.groupby("ticker", as_index=False).agg(
        min_date=("date", "min"),
        max_date=("date", "max"),
    )
    coverage_ok = set(
        coverage[
            (coverage["min_date"] <= window_start) & (coverage["max_date"] >= window_end)
        ]["ticker"].tolist()
    )

    needed = [ticker for ticker in tickers if ticker not in coverage_ok]
    cached_subset = in_window[in_window["ticker"].isin(tickers)].copy()
    LOGGER.info(
        "Price cache covers %d/%d tickers for requested window",
        len(set(tickers) - set(needed)),
        len(tickers),
    )
    return needed, cached_subset


def download_price_data(
    tickers: Iterable[str], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    """Download daily price history from Yahoo Finance in ticker batches."""
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    # Include a small lookback and one-day pad on the end for stable forward shifts.
    start = (start_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    ticker_list = list(tickers)
    total_batches = int(np.ceil(len(ticker_list) / PRICE_DOWNLOAD_BATCH_SIZE))
    for batch_idx in range(total_batches):
        batch = ticker_list[
            batch_idx * PRICE_DOWNLOAD_BATCH_SIZE : (batch_idx + 1) * PRICE_DOWNLOAD_BATCH_SIZE
        ]
        LOGGER.info(
            "Downloading price batch %d/%d (%d tickers)",
            batch_idx + 1,
            total_batches,
            len(batch),
        )
        raw_df = pd.DataFrame()
        for attempt in range(1, PRICE_DOWNLOAD_RETRIES + 1):
            try:
                raw_df = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
            except Exception as exc:  # defensive against network/symbol errors
                LOGGER.warning(
                    "Batch %d/%d download error on attempt %d/%d: %s",
                    batch_idx + 1,
                    total_batches,
                    attempt,
                    PRICE_DOWNLOAD_RETRIES,
                    exc,
                )
                raw_df = pd.DataFrame()

            if not raw_df.empty:
                break

            if attempt < PRICE_DOWNLOAD_RETRIES:
                backoff_seconds = float(2 ** (attempt - 1))
                LOGGER.info(
                    "Retrying batch %d/%d after %.1fs backoff",
                    batch_idx + 1,
                    total_batches,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)

        if raw_df.empty:
            failed.extend(batch)
            LOGGER.warning("Batch %d/%d produced no rows after retries", batch_idx + 1, total_batches)
            if batch_idx + 1 < total_batches:
                time.sleep(PRICE_BATCH_SLEEP_SECONDS)
            continue

        # Multi-ticker responses are typically columns like (ticker, field).
        if isinstance(raw_df.columns, pd.MultiIndex):
            for ticker in batch:
                if ticker not in raw_df.columns.get_level_values(0):
                    failed.append(ticker)
                    continue
                ticker_df = raw_df[ticker]
                normalized = _normalize_price_frame(ticker_df, ticker=ticker)
                if normalized.empty:
                    failed.append(ticker)
                    continue
                frames.append(normalized)
        else:
            # Single ticker fallback shape.
            ticker = batch[0]
            normalized = _normalize_price_frame(raw_df, ticker=ticker)
            if normalized.empty:
                failed.append(ticker)
            else:
                frames.append(normalized)
            if len(batch) > 1:
                failed.extend(batch[1:])

        if batch_idx + 1 < total_batches:
            sleep_seconds = PRICE_BATCH_SLEEP_SECONDS + random.uniform(
                0.0, PRICE_BATCH_SLEEP_JITTER_SECONDS
            )
            time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS), failed

    prices_df = pd.concat(frames, ignore_index=True)
    prices_df = (
        prices_df.sort_values(["ticker", "date"])
        .drop_duplicates(subset=["date", "ticker"], keep="last")
        .reset_index(drop=True)
    )
    return prices_df, failed


def drop_incomplete_price_rows(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where any OHLCV field is missing (no partial / bad Yahoo rows)."""
    if prices_df.empty:
        return prices_df
    before = len(prices_df)
    complete_mask = prices_df[PRICE_COLUMNS].notna().all(axis=1)
    out = prices_df.loc[complete_mask].copy()
    dropped = before - len(out)
    LOGGER.info(
        "Dropped %d rows (%.2f%%) with incomplete OHLCV data",
        dropped,
        100.0 * dropped / before if before else 0.0,
    )
    return out


def build_price_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Create return-based forecasting target features."""
    if prices_df.empty:
        df = prices_df.copy()
        for col in ["return_1d", "return_fwd_1d", "return_fwd_5d", "direction_fwd_1d"]:
            if col not in df.columns:
                df[col] = np.nan
        return df

    df = prices_df.sort_values(["ticker", "date"]).copy()
    grouped_close = df.groupby("ticker")["close"]
    close_next_1d = grouped_close.shift(-1)
    close_next_5d = grouped_close.shift(-5)

    df["return_1d"] = grouped_close.pct_change(fill_method=None)
    df["return_fwd_1d"] = close_next_1d / df["close"] - 1.0
    df["return_fwd_5d"] = close_next_5d / df["close"] - 1.0
    df["direction_fwd_1d"] = (df["return_fwd_1d"] > 0).astype(int)
    return df


def merge_datasets(prices_df: pd.DataFrame, reddit_by_ticker: pd.DataFrame) -> pd.DataFrame:
    """Left join per-ticker Reddit aggregates onto the price time series (merge on `ticker` only)."""
    merged = prices_df.merge(reddit_by_ticker, on="ticker", how="left", suffixes=("", "_reddit"))
    for col in COUNT_LIKE_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    return merged


def build_reddit_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Set derived Reddit columns when mentions are per-ticker totals (no daily series).

    Without dated Reddit data, lags/rolling means are not defined as time
    features; 1d lag is 0, 3d/7d "moving averages" are set equal to
    `reddit_total_mentions` (constant by ticker), and abnormal is 1 when
    `reddit_total_mentions` > 0 else 0.
    """
    if "reddit_total_mentions" not in model_df.columns:
        model_df["reddit_total_mentions"] = 0

    df = model_df.sort_values(["ticker", "date"]).copy()
    m = df["reddit_total_mentions"]
    df["reddit_mentions_1d_lag"] = 0.0
    df["reddit_mentions_3d_ma"] = m
    df["reddit_mentions_7d_ma"] = m
    df["reddit_abnormal_mentions"] = np.where(m > 0, 1.0, 0.0)
    return df


def validate_final_dataset(model_df: pd.DataFrame, failed_tickers: list[str]) -> None:
    """Validate final dataset schema and key quality constraints."""
    duplicate_count = model_df.duplicated(subset=["date", "ticker"]).sum()
    if duplicate_count:
        raise ValueError(f"Final dataset has {duplicate_count} duplicate (date, ticker) rows")

    missing_required = [col for col in MODEL_REQUIRED_COLUMNS if col not in model_df.columns]
    if missing_required:
        raise ValueError(f"Final dataset missing required columns: {missing_required}")

    if model_df.empty:
        LOGGER.warning("Final model dataset is empty")
        return

    null_price_share = model_df[PRICE_COLUMNS].isna().any(axis=1).mean()
    if null_price_share > 0.05:
        LOGGER.warning("High null-price-row share detected: %.2f%%", null_price_share * 100)

    if failed_tickers:
        fail_ratio = len(failed_tickers) / (model_df["ticker"].nunique() + len(failed_tickers))
        if fail_ratio > 0.3:
            LOGGER.warning(
                "Large ticker download failure share: %d failed (%.2f%%)",
                len(failed_tickers),
                fail_ratio * 100,
            )


def save_outputs(prices_df: pd.DataFrame, model_df: pd.DataFrame, dataset_dir: Path) -> None:
    """Save generated outputs to disk."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prices_path = dataset_dir / "prices_daily.csv"
    model_path = dataset_dir / "model_dataset.csv"

    prices_df.to_csv(prices_path, index=False)
    model_df.to_csv(model_path, index=False)
    LOGGER.info("Saved prices dataset: %s", prices_path)
    LOGGER.info("Saved model dataset: %s", model_path)


def print_diagnostics(
    reddit_rows: int,
    reddit_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> None:
    """Print summary diagnostics for quick pipeline sanity checks."""
    unique_tickers = model_df["ticker"].nunique() if not model_df.empty else 0
    if not model_df.empty:
        date_min = model_df["date"].min()
        date_max = model_df["date"].max()
    else:
        date_min = pd.NaT
        date_max = pd.NaT

    LOGGER.info("Unique tickers: %d", unique_tickers)
    LOGGER.info("Date range: %s -> %s", date_min, date_max)
    LOGGER.info("Rows in Reddit input: %d", reddit_rows)
    LOGGER.info("Rows in normalized Reddit table: %d", len(reddit_df))
    LOGGER.info("Rows in price table: %d", len(prices_df))
    LOGGER.info("Rows in final model dataset: %d", len(model_df))
    LOGGER.info(
        "Reddit is aggregated per ticker only (no date dimension); same totals repeat on each price row"
    )
    if not model_df.empty:
        null_sent = {
            c: int(model_df[c].isna().sum())
            for c in SENTIMENT_STYLE_COLUMNS
            if c in model_df.columns
        }
        if any(null_sent.values()):
            LOGGER.info("Null sentiment-style Reddit columns (if present): %s", null_sent)

    major_groups = {
        "price_columns": [col for col in PRICE_COLUMNS if col in model_df.columns],
        "reddit_count_columns": [col for col in COUNT_LIKE_COLUMNS if col in model_df.columns],
        "reddit_sentiment_columns": [
            col for col in SENTIMENT_STYLE_COLUMNS if col in model_df.columns
        ],
        "target_columns": [
            col
            for col in ["return_1d", "return_fwd_1d", "return_fwd_5d", "direction_fwd_1d"]
            if col in model_df.columns
        ],
    }
    for group_name, cols in major_groups.items():
        if not cols:
            continue
        null_counts = model_df[cols].isna().sum().to_dict()
        LOGGER.info("Missing values (%s): %s", group_name, null_counts)


def main() -> None:
    """Run the unified model-dataset build pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"
    reddit_path = dataset_dir / "reddit_mentions_daily.csv"
    prices_path = dataset_dir / "prices_daily.csv"

    reddit_raw = load_reddit_daily(reddit_path)
    reddit_rows = len(reddit_raw)
    reddit_df = normalize_reddit_schema(reddit_raw)
    tickers = get_unique_tickers(reddit_df)

    if reddit_df.empty or not tickers:
        LOGGER.warning("No valid Reddit rows/tickers available; writing empty outputs")
        empty_prices = pd.DataFrame(columns=PRICE_OUTPUT_COLUMNS)
        empty_model = pd.DataFrame(columns=MODEL_REQUIRED_COLUMNS)
        save_outputs(empty_prices, empty_model, dataset_dir=dataset_dir)
        return

    # Price window: past year of calendar days ending today (not tied to Reddit post dates).
    price_window_end = pd.Timestamp.today().normalize()
    price_window_start = price_window_end - pd.Timedelta(days=PRICE_HISTORY_LOOKBACK_DAYS)
    start_date, end_date = price_window_start, price_window_end
    LOGGER.info(
        "Yahoo price window: %s -> %s (%d-day lookback, extra history may be loaded for returns)",
        start_date.date(),
        end_date.date(),
        PRICE_HISTORY_LOOKBACK_DAYS,
    )

    cache_df = load_price_cache(prices_path)
    tickers_to_download, cached_prices_subset = select_tickers_needing_download(
        tickers=tickers, cache_df=cache_df, start_date=start_date, end_date=end_date
    )
    if tickers_to_download:
        LOGGER.info("Downloading prices for %d tickers not fully covered by cache", len(tickers_to_download))
    else:
        LOGGER.info("Using cached prices for all selected tickers")

    downloaded_prices_df, failed_tickers = download_price_data(
        tickers_to_download, start_date=start_date, end_date=end_date
    )
    prices_df = pd.concat([cached_prices_subset, downloaded_prices_df], ignore_index=True)
    if not prices_df.empty:
        prices_df = (
            prices_df.sort_values(["ticker", "date"])
            .drop_duplicates(subset=["date", "ticker"], keep="last")
            .reset_index(drop=True)
        )

    if failed_tickers:
        LOGGER.warning("Tickers failed Yahoo download (%d): %s", len(failed_tickers), failed_tickers[:20])

    prices_df = drop_incomplete_price_rows(prices_df)
    prices_with_features = build_price_features(prices_df)
    model_df = merge_datasets(prices_with_features, reddit_df)
    model_df = build_reddit_features(model_df)

    # Saved files only include rows inside the one-year [start_date, end_date] window
    # (wider data was used above so first-day returns can use prior prices).
    in_window = (model_df["date"] >= start_date) & (model_df["date"] <= end_date)
    model_df = model_df.loc[in_window].reset_index(drop=True)
    price_rows_out = (prices_df["date"] >= start_date) & (prices_df["date"] <= end_date)
    prices_df = prices_df.loc[price_rows_out].reset_index(drop=True)

    # Keep a stable column order: required/core first, then optional extras.
    optional_existing = [
        col
        for col in REDDIT_OPTIONAL_COLUMNS
        if col in model_df.columns and col not in MODEL_REQUIRED_COLUMNS and col != "subreddit"
    ]
    ordered_cols = MODEL_REQUIRED_COLUMNS + [
        col for col in PRICE_OUTPUT_COLUMNS if col not in MODEL_REQUIRED_COLUMNS
    ]
    ordered_cols += [col for col in optional_existing if col not in ordered_cols]
    ordered_cols += [col for col in model_df.columns if col not in ordered_cols]
    model_df = model_df[ordered_cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    validate_final_dataset(model_df, failed_tickers=failed_tickers)
    save_outputs(prices_df=prices_df, model_df=model_df, dataset_dir=dataset_dir)
    print_diagnostics(reddit_rows=reddit_rows, reddit_df=reddit_df, prices_df=prices_df, model_df=model_df)


if __name__ == "__main__":
    main()
