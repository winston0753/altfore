"""Plot daily OHLC, volume, and Reddit mentions for a single ticker from the project dataset.

Usage:
    python scripts/plot_ticker_prices.py
    python scripts/plot_ticker_prices.py --ticker AAPL
    python scripts/plot_ticker_prices.py --ticker ORCL --output figures/orcl.png
    python scripts/plot_ticker_prices.py --ticker ORCL --mentions dataset/reddit_mentions_daily.csv --output figures/orcl.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless / CI: avoid interactive backend and font cache issues
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.visualization.ticker_prices import plot_ticker_prices

    parser = argparse.ArgumentParser(description="Plot daily prices for one ticker")
    default_csv = project_root / "dataset" / "prices_daily.csv"
    default_mentions = project_root / "dataset" / "reddit_mentions_daily.csv"
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
        "--mentions",
        type=Path,
        default=default_mentions,
        help="Path to reddit_mentions_daily CSV (default: dataset/reddit_mentions_daily.csv). Pass empty string to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (PNG). If omitted, opens an interactive window.",
    )
    args = parser.parse_args()
    mentions_path = args.mentions if args.mentions and str(args.mentions) else None

    saved_path = plot_ticker_prices(
        input_path=args.input,
        ticker=args.ticker,
        output_path=args.output,
        mentions_path=mentions_path,
    )
    if saved_path is not None:
        print(f"Saved: {saved_path}")
    else:
        print("Displayed plot interactively.")


if __name__ == "__main__":
    main()
