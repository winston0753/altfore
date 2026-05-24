"""Build a price + WSB mentions dataset for 2021-2025."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from altfore.pipeline.model_dataset import (
    download_price_data,
    drop_incomplete_price_rows,
    PRICE_OUTPUT_COLUMNS,
)

LOGGER = logging.getLogger(__name__)

# Curated ticker universe: all-5-year WSB coverage, tradeable stocks only.
# Dropped: ORCL, IREN (1/5-year WSB coverage — too sparse for time-series features).
# Skipped Tier-1 entries with high noise: BB (meme legacy), TLRY (cannabis penny),
# BABA (ADR geopolitical distortion), SOFI/HOOD (borderline).
TICKERS: list[str] = [
    # original set (kept)
    "AMC", "AMZN", "F", "GME", "LULU", "MSFT", "MU", "NVDA",
    # Tier-1 additions — all 5 years of WSB data, solid mention volume
    "AAPL", "AMD", "BA", "COIN", "GM", "INTC",
    "JPM", "NFLX", "PLTR", "SNAP", "TSM", "TSLA",
]


def load_wsb_mentions(wsb_dir: Path, tickers: list[str]) -> pd.DataFrame:
    """Load all WSB CSVs, melt to long (date, ticker, mentions), filter to tickers."""
    frames: list[pd.DataFrame] = []
    ticker_set = set(tickers)

    for path in sorted(wsb_dir.glob("wallstreetbets_*.csv")):
        df = pd.read_csv(path)
        date_cols = [c for c in df.columns if c not in ("ticker", "overall_rank", "total")]
        df = df[["ticker"] + date_cols].copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df[df["ticker"].isin(ticker_set)]
        if df.empty:
            continue
        melted = df.melt(id_vars="ticker", var_name="date_raw", value_name="mentions")
        melted["date"] = pd.to_datetime(melted["date_raw"], format="%m/%d/%y", errors="coerce")
        melted = melted.dropna(subset=["date"])
        melted["mentions"] = pd.to_numeric(melted["mentions"], errors="coerce").fillna(0).astype(int)
        frames.append(melted[["date", "ticker", "mentions"]])
        LOGGER.info("Loaded %s: %d mention rows for %d tickers", path.name, len(melted), df["ticker"].nunique())

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "mentions"])

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return out


def load_google_trends(trends_path: Path, all_dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Load trends_daily.csv and forward-fill weekly observations to daily.

    Google Trends returns one value per week (week-start date). We forward-fill
    each ticker's weekly interest to every calendar day so it can be merged on
    the daily price index. Returns long-format (date, ticker, trends_interest),
    or None if the file does not exist.
    """
    if not trends_path.exists():
        return None

    df = pd.read_csv(trends_path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()

    filled_frames: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.set_index("date")[["trends_interest"]].sort_index()
        grp = grp.reindex(all_dates).ffill()
        grp["ticker"] = ticker
        grp = grp.reset_index().rename(columns={"index": "date"})
        filled_frames.append(grp)

    if not filled_frames:
        return None

    return pd.concat(filled_frames, ignore_index=True)


def run_build_wsb_dataset(project_root: Path) -> None:
    """Build the WSB-era price + mentions dataset and write it to dataset/."""
    wsb_dir = project_root / "wsb_data"
    dataset_dir = project_root / "dataset"
    output_path = dataset_dir / "wsb_model_dataset.csv"

    # verify each curated ticker actually appears in at least one WSB file
    wsb_tickers: set[str] = set()
    for path in sorted(wsb_dir.glob("wallstreetbets_*.csv")):
        df = pd.read_csv(path, usecols=["ticker"])
        wsb_tickers.update(df["ticker"].astype(str).str.strip().str.upper())

    common_tickers = sorted(t for t in TICKERS if t in wsb_tickers)
    missing = sorted(t for t in TICKERS if t not in wsb_tickers)
    if missing:
        LOGGER.warning("Tickers in TICKERS but absent from all WSB files: %s", missing)
    LOGGER.info("Building dataset for %d tickers: %s", len(common_tickers), common_tickers)

    mentions_df = load_wsb_mentions(wsb_dir, common_tickers)
    start_date = mentions_df["date"].min()
    end_date = mentions_df["date"].max()
    LOGGER.info("WSB date horizon: %s -> %s", start_date.date(), end_date.date())

    LOGGER.info("Downloading prices for %d tickers (%s -> %s)", len(common_tickers), start_date.date(), end_date.date())
    prices_df, failed = download_price_data(common_tickers, start_date=start_date, end_date=end_date)
    if failed:
        LOGGER.warning("Failed to download prices for: %s", failed)

    prices_df = drop_incomplete_price_rows(prices_df)
    in_window = (prices_df["date"] >= start_date) & (prices_df["date"] <= end_date)
    prices_df = prices_df.loc[in_window].reset_index(drop=True)
    LOGGER.info("Price rows in window: %d", len(prices_df))

    merged = prices_df.merge(mentions_df, on=["date", "ticker"], how="left")
    merged["mentions"] = merged["mentions"].fillna(0).astype(int)

    # merge Google Trends if available
    trends_path = dataset_dir / "trends_daily.csv"
    all_dates = pd.DatetimeIndex(merged["date"].unique())
    trends_df = load_google_trends(trends_path, all_dates)
    if trends_df is not None:
        merged = merged.merge(trends_df, on=["date", "ticker"], how="left")
        merged["trends_interest"] = merged["trends_interest"].fillna(0.0)
        LOGGER.info("Merged Google Trends data (%d rows)", len(trends_df))
    else:
        merged["trends_interest"] = 0.0
        LOGGER.info("trends_daily.csv not found — trends_interest set to 0; run build_trends_dataset.py first")

    col_order = PRICE_OUTPUT_COLUMNS + ["mentions", "trends_interest"]
    merged = (
        merged[col_order]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    LOGGER.info("Saved %d rows to %s", len(merged), output_path)

    for ticker in common_tickers:
        sub = merged[merged["ticker"] == ticker]
        LOGGER.info(
            "  %s: %d price rows, %d days with mentions, max mentions=%d",
            ticker,
            len(sub),
            int((sub["mentions"] > 0).sum()),
            int(sub["mentions"].max()) if not sub.empty else 0,
        )
