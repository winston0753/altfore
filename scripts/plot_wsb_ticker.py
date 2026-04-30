"""Plot price history and WSB mention count for a single ticker.

Usage:
    python scripts/plot_wsb_ticker.py --ticker GME
    python scripts/plot_wsb_ticker.py --ticker NVDA --output dataset/nvda_wsb.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.visualization.wsb_ticker import plot_wsb_ticker

    default_csv = project_root / "dataset" / "wsb_model_dataset.csv"
    parser = argparse.ArgumentParser(description="Plot price + WSB mentions for one ticker")
    parser.add_argument("--ticker", type=str, default="GME", help="Uppercase symbol (default: GME)")
    parser.add_argument("--input", type=Path, default=default_csv, help="Path to wsb_model_dataset CSV")
    parser.add_argument("--output", type=Path, default=None, help="Save figure to this path (PNG). If omitted, opens an interactive window.")
    args = parser.parse_args()

    saved_path = plot_wsb_ticker(input_path=args.input, ticker=args.ticker, output_path=args.output)
    if saved_path is not None:
        print(f"Saved: {saved_path}")
    else:
        print("Displayed plot interactively.")


if __name__ == "__main__":
    main()
