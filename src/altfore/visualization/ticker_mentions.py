"""Plot daily Reddit mentions for a single ticker."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SUBREDDIT_COLORS = {
    "wallstreetbets": "#e05c2a",
    "wsb": "#e05c2a",
    "stocks": "#4c8cbf",
    "investing": "#5aab61",
}
DEFAULT_COLOR = "darkorange"


def plot_ticker_mentions(
    input_path: Path,
    ticker: str,
    output_path: Path | None = None,
) -> Path | None:
    """Plot daily Reddit mentions for one ticker, stacked by subreddit."""
    normalized_ticker = ticker.strip().upper()
    df = pd.read_csv(input_path)
    required = {"date", "ticker", "total_mentions"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise SystemExit(f"Input CSV missing columns: {missing}")

    sub = df[df["ticker"].astype(str).str.strip().str.upper() == normalized_ticker].copy()
    if sub.empty:
        available = sorted(df["ticker"].astype(str).str.upper().unique().tolist())[:30]
        raise SystemExit(f"No rows for {normalized_ticker}. Example tickers: {available}")

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"])
    sub["total_mentions"] = pd.to_numeric(sub["total_mentions"], errors="coerce").fillna(0)

    has_subreddit = "subreddit" in sub.columns and sub["subreddit"].notna().any()

    fig, ax = plt.subplots(figsize=(10, 4))

    if has_subreddit:
        sub["subreddit"] = sub["subreddit"].astype(str).str.strip().str.lower()
        pivoted = (
            sub.groupby(["date", "subreddit"])["total_mentions"]
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )
        subreddits = pivoted.columns.tolist()
        colors = [SUBREDDIT_COLORS.get(s, DEFAULT_COLOR) for s in subreddits]
        bottom = pd.Series(0, index=pivoted.index)
        for col, color in zip(subreddits, colors):
            ax.bar(
                pivoted.index,
                pivoted[col],
                bottom=bottom,
                width=0.8,
                color=color,
                alpha=0.85,
                label=f"r/{col}",
            )
            bottom = bottom + pivoted[col]
        ax.legend(loc="upper left", fontsize=8)
    else:
        daily = sub.groupby("date")["total_mentions"].sum().sort_index()
        ax.bar(daily.index, daily, width=0.8, color=DEFAULT_COLOR, alpha=0.85)

    date_min = sub["date"].min().date()
    date_max = sub["date"].max().date()
    ax.set_title(f"{normalized_ticker} - daily Reddit mentions ({date_min} -> {date_max})")
    ax.set_ylabel("Mentions")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3, axis="y")
    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path is None:
        plt.show()
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path.resolve()
