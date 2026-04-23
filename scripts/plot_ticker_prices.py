"""Plot daily OHLC (and optionally volume) for a single ticker from the project dataset.

Usage:
    python scripts/plot_ticker_prices.py
    python scripts/plot_ticker_prices.py --ticker AAPL
    python scripts/plot_ticker_prices.py --ticker ORCL --output figures/orcl.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Headless / CI: avoid interactive backend and font cache issues
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot daily prices for one ticker")
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "dataset" / "prices_daily.csv"
    parser.add_argument(
        "--ticker",
        type=str,
        default="ORCL",
        help="Uppercase symbol (default: ORCL)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_csv,
        help="Path to prices CSV (default: dataset/prices_daily.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (PNG). If omitted, opens an interactive window.",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    df = pd.read_csv(args.input)
    if "ticker" not in df.columns or "date" not in df.columns:
        raise SystemExit("Input CSV must include columns: date, ticker")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV must include column: {col}")

    sub = df[df["ticker"].astype(str).str.strip().str.upper() == ticker].copy()
    if sub.empty:
        available = sorted(df["ticker"].astype(str).str.upper().unique().tolist())[:30]
        raise SystemExit(f"No rows for {ticker}. Example tickers: {available}")

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date")

    fig, (ax_pr, ax_vol) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    # Price panel: close line + shaded high-low
    ax_pr.fill_between(
        sub["date"],
        sub["low"],
        sub["high"],
        alpha=0.25,
        color="steelblue",
        label="High–low range",
    )
    ax_pr.plot(sub["date"], sub["close"], color="navy", linewidth=1.2, label="Close")
    ax_pr.set_ylabel("Price (USD)")
    ax_pr.set_title(f"{ticker} — daily prices ({sub['date'].min().date()} → {sub['date'].max().date()})")
    ax_pr.legend(loc="upper left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    # Volume
    if "volume" in sub.columns and sub["volume"].notna().any():
        ax_vol.bar(sub["date"], sub["volume"], width=0.8, color="gray", alpha=0.6)
        ax_vol.set_ylabel("Volume")
    else:
        ax_vol.text(0.5, 0.5, "No volume", ha="center", va="center", transform=ax_vol.transAxes)
    ax_vol.set_xlabel("Date")
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()

    if args.output:
        args.output = Path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
