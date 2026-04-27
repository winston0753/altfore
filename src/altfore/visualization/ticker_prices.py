"""Plot daily OHLC (and optionally volume) for a ticker."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_ticker_prices(input_path: Path, ticker: str, output_path: Path | None = None) -> Path | None:
    """Plot daily prices for one ticker from a prices CSV."""
    normalized_ticker = ticker.strip().upper()
    df = pd.read_csv(input_path)
    if "ticker" not in df.columns or "date" not in df.columns:
        raise SystemExit("Input CSV must include columns: date, ticker")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV must include column: {col}")

    sub = df[df["ticker"].astype(str).str.strip().str.upper() == normalized_ticker].copy()
    if sub.empty:
        available = sorted(df["ticker"].astype(str).str.upper().unique().tolist())[:30]
        raise SystemExit(f"No rows for {normalized_ticker}. Example tickers: {available}")

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"]).sort_values("date")

    fig, (ax_pr, ax_vol) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_pr.fill_between(
        sub["date"],
        sub["low"],
        sub["high"],
        alpha=0.25,
        color="steelblue",
        label="High-low range",
    )
    ax_pr.plot(sub["date"], sub["close"], color="navy", linewidth=1.2, label="Close")
    ax_pr.set_ylabel("Price (USD)")
    ax_pr.set_title(
        f"{normalized_ticker} - daily prices ({sub['date'].min().date()} -> {sub['date'].max().date()})"
    )
    ax_pr.legend(loc="upper left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    if "volume" in sub.columns and sub["volume"].notna().any():
        ax_vol.bar(sub["date"], sub["volume"], width=0.8, color="gray", alpha=0.6)
        ax_vol.set_ylabel("Volume")
    else:
        ax_vol.text(0.5, 0.5, "No volume", ha="center", va="center", transform=ax_vol.transAxes)
    ax_vol.set_xlabel("Date")
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path is None:
        plt.show()
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path.resolve()
