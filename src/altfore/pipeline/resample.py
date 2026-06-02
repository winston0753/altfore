"""Resample daily price + mentions panel to weekly frequency."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily panel to weekly (period label = last trading day of each week).

    Uses W-FRI anchoring: each bin covers Saturday–Friday; .last() / .sum() etc.
    operate within that window, so a Thursday close in a holiday week is correctly
    captured under the Friday label.

    Input columns required: date, ticker, open, high, low, close, volume,
                            mentions, trends_interest.
    Output columns: date (week-end Friday), ticker, open, high, low, close,
                    volume, mentions_sum, mentions_mean, mentions_max,
                    mentions_days, trends_interest, days_in_week.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    frames: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.set_index("date").sort_index()

        has_mention = (grp["mentions"] > 0).astype(int)
        weekly = pd.DataFrame({
            "close":           grp["close"].resample("W-FRI").last(),
            "open":            grp["open"].resample("W-FRI").first(),
            "high":            grp["high"].resample("W-FRI").max(),
            "low":             grp["low"].resample("W-FRI").min(),
            "volume":          grp["volume"].resample("W-FRI").sum(),
            "mentions_sum":    grp["mentions"].resample("W-FRI").sum(),
            "mentions_mean":   grp["mentions"].resample("W-FRI").mean(),
            "mentions_max":    grp["mentions"].resample("W-FRI").max(),
            "mentions_days":   has_mention.resample("W-FRI").sum(),
            # trends_interest is already weekly (forward-filled); last() is the native value
            "trends_interest": grp["trends_interest"].resample("W-FRI").last(),
            "days_in_week":    grp["close"].resample("W-FRI").count(),
        }).dropna(subset=["close"])

        weekly["ticker"] = ticker
        weekly = weekly.reset_index().rename(columns={"date": "date"})
        frames.append(weekly)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    LOGGER.info(
        "Resampled to weekly: %d rows, %d tickers, %s → %s",
        len(out),
        out["ticker"].nunique(),
        out["date"].min().date(),
        out["date"].max().date(),
    )
    return out
