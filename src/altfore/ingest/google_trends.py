"""Fetch Google Trends search interest for the WSB ticker universe."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 5
REQUEST_PAUSE = 10.0  # seconds between batch requests


def _fetch_batch(
    tickers: list[str],
    timeframe: str,
    client: TrendReq,
) -> pd.DataFrame:
    """Fetch weekly search interest for up to 5 tickers in one request."""
    client.build_payload(tickers, timeframe=timeframe, geo="US")
    df = client.interest_over_time()
    if df.empty:
        return pd.DataFrame()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df[tickers]


def fetch_trends(
    tickers: list[str],
    start_date: str,
    end_date: str,
    pause: float = REQUEST_PAUSE,
) -> pd.DataFrame:
    """Fetch weekly Google Trends search interest for all tickers.

    Sends batches of up to 5 tickers (Google Trends API limit). Each batch
    is independently normalised to 0-100 by Google; within-ticker z-scoring
    in feature engineering makes cross-batch comparison unnecessary.

    Returns a long-format DataFrame: date, ticker, trends_interest.
    Dates are week-start (Sunday) timestamps as returned by the API.
    """
    client = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=0.5)
    timeframe = f"{start_date} {end_date}"

    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames: list[pd.DataFrame] = []

    for i, batch in enumerate(batches):
        LOGGER.info("Fetching Google Trends batch %d/%d: %s", i + 1, len(batches), batch)
        try:
            df = _fetch_batch(batch, timeframe, client)
            if df.empty:
                LOGGER.warning("Empty response for batch %s — skipping", batch)
            else:
                frames.append(df)
        except Exception as exc:
            LOGGER.error("Batch %s failed: %s", batch, exc)

        if i < len(batches) - 1:
            time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "trends_interest"])

    wide = pd.concat(frames, axis=1)
    wide.index.name = "date"
    wide = wide.reset_index()

    long = wide.melt(id_vars="date", var_name="ticker", value_name="trends_interest")
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long = long.dropna(subset=["trends_interest"])
    long["trends_interest"] = long["trends_interest"].astype(float)
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def run_build_trends_dataset(
    project_root: Path,
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """Fetch trends and write dataset/trends_daily.csv."""
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "trends_daily.csv"

    LOGGER.info(
        "Fetching Google Trends for %d tickers (%s → %s)",
        len(tickers),
        start_date,
        end_date,
    )
    df = fetch_trends(tickers, start_date=start_date, end_date=end_date)

    if df.empty:
        LOGGER.error("No trends data retrieved — aborting")
        return

    df.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved %d rows (%d tickers) to %s",
        len(df),
        df["ticker"].nunique(),
        output_path,
    )
    for ticker, grp in df.groupby("ticker"):
        LOGGER.info(
            "  %s: %d weekly observations, max interest=%d",
            ticker,
            len(grp),
            int(grp["trends_interest"].max()),
        )
