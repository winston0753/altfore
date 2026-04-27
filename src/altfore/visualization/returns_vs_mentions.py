"""Plot one-year return and Reddit mentions for all tickers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_returns_vs_mentions(input_path: Path, output_path: Path) -> Path:
    """Create side-by-side return and mentions bar plots for all tickers."""
    df = pd.read_csv(input_path)
    required_cols = {"date", "ticker", "close", "reddit_total_mentions"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    if df.empty:
        raise SystemExit("No rows available after parsing input data")

    per_ticker = (
        df.groupby("ticker", as_index=False)
        .agg(
            first_close=("close", "first"),
            last_close=("close", "last"),
            reddit_total_mentions=("reddit_total_mentions", "max"),
        )
        .copy()
    )
    per_ticker["one_year_return_pct"] = (
        (per_ticker["last_close"] / per_ticker["first_close"] - 1.0) * 100.0
    )
    per_ticker = per_ticker.sort_values("one_year_return_pct", ascending=False).reset_index(drop=True)

    fig, (ax_ret, ax_men) = plt.subplots(1, 2, figsize=(16, 6))
    x = range(len(per_ticker))
    tickers = per_ticker["ticker"].tolist()

    colors = ["seagreen" if v >= 0 else "indianred" for v in per_ticker["one_year_return_pct"]]
    ax_ret.bar(x, per_ticker["one_year_return_pct"], color=colors, alpha=0.85)
    ax_ret.axhline(0, color="black", linewidth=0.8)
    ax_ret.set_title("One-year return by ticker")
    ax_ret.set_ylabel("Return (%)")
    ax_ret.set_xticks(list(x))
    ax_ret.set_xticklabels(tickers, rotation=75, ha="right", fontsize=8)
    ax_ret.grid(True, axis="y", alpha=0.25)

    ax_men.bar(x, per_ticker["reddit_total_mentions"], color="steelblue", alpha=0.85)
    ax_men.set_title("Reddit total mentions by ticker")
    ax_men.set_ylabel("Mentions")
    ax_men.set_xticks(list(x))
    ax_men.set_xticklabels(tickers, rotation=75, ha="right", fontsize=8)
    ax_men.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path.resolve()
