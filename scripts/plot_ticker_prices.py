"""Plot daily OHLC (and optionally volume) for a single ticker from the project dataset.

Usage:
    python scripts/plot_ticker_prices.py
    python scripts/plot_ticker_prices.py --ticker AAPL
    python scripts/plot_ticker_prices.py --ticker ORCL --output figures/orcl.png
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

    saved_path = plot_ticker_prices(
        input_path=args.input,
        ticker=args.ticker,
        output_path=args.output,
    )
    if saved_path is not None:
        print(f"Saved: {saved_path}")
    else:
        print("Displayed plot interactively.")


if __name__ == "__main__":
    main()
