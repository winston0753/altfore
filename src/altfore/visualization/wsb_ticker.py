"""Plot price history and WSB mention count for a single ticker."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_wsb_ticker(
    input_path: Path,
    ticker: str,
    output_path: Path | None = None,
) -> Path | None:
    """Plot OHLC price range + close and daily WSB mentions for one ticker."""
    normalized_ticker = ticker.strip().upper()
    df = pd.read_csv(input_path)

    required = {"date", "ticker", "open", "high", "low", "close", "mentions"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing columns: {sorted(missing)}")

    sub = df[df["ticker"].astype(str).str.strip().str.upper() == normalized_ticker].copy()
    if sub.empty:
        available = sorted(df["ticker"].astype(str).str.upper().unique().tolist())
        raise SystemExit(f"No rows for {normalized_ticker}. Available tickers: {available}")

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date")
    sub["mentions"] = pd.to_numeric(sub["mentions"], errors="coerce").fillna(0)

    date_min = sub["date"].min().date()
    date_max = sub["date"].max().date()

    fig, (ax_pr, ax_men) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_pr.fill_between(sub["date"], sub["low"], sub["high"], alpha=0.25, color="steelblue", label="High-low range")
    ax_pr.plot(sub["date"], sub["close"], color="navy", linewidth=1.0, label="Close")
    ax_pr.set_ylabel("Price (USD)")
    ax_pr.set_title(f"{normalized_ticker} — price and WSB mentions ({date_min} → {date_max})")
    ax_pr.legend(loc="upper left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    ax_men.bar(sub["date"], sub["mentions"], width=0.8, color="darkorange", alpha=0.85)
    ax_men.set_ylabel("Mentions")
    ax_men.set_xlabel("Date")
    ax_men.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_men.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_men.grid(True, alpha=0.3, axis="y")

    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path is None:
        plt.show()
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()
