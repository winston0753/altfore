"""Plot daily OHLC, volume, and optionally Reddit mentions for a ticker."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _load_mentions(mentions_path: Path, normalized_ticker: str) -> pd.DataFrame:
    """Load and aggregate daily Reddit mentions for one ticker."""
    if not mentions_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(mentions_path)
    required = {"date", "ticker", "total_mentions"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["ticker"].astype(str).str.strip().str.upper() == normalized_ticker].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["total_mentions"] = pd.to_numeric(df["total_mentions"], errors="coerce").fillna(0)
    return (
        df.groupby("date", as_index=False)["total_mentions"]
        .sum()
        .sort_values("date")
    )


def plot_ticker_prices(
    input_path: Path,
    ticker: str,
    output_path: Path | None = None,
    mentions_path: Path | None = None,
) -> Path | None:
    """Plot daily prices (and optionally volume and Reddit mentions) for one ticker."""
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

    mentions_df = _load_mentions(mentions_path, normalized_ticker) if mentions_path is not None else pd.DataFrame()
    has_mentions = not mentions_df.empty

    n_panels = 3 if has_mentions else 2
    height_ratios = [3, 1, 1] if has_mentions else [3, 1]
    fig, axes = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=(10, 6 + has_mentions),
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax_pr, ax_vol = axes[0], axes[1]
    ax_men = axes[2] if has_mentions else None

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

    if ax_men is not None:
        ax_men.bar(mentions_df["date"], mentions_df["total_mentions"], width=0.8, color="darkorange", alpha=0.7)
        ax_men.set_ylabel("Mentions")
        ax_men.set_xlabel("Date")
        ax_men.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    else:
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
