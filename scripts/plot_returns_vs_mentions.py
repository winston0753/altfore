"""Script wrapper for returns-vs-mentions plot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from altfore.visualization.returns_vs_mentions import plot_returns_vs_mentions

    parser = argparse.ArgumentParser(description="Plot ticker returns vs Reddit mentions")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "model_dataset.csv"),
        help="Path to model dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "returns_vs_mentions.png"),
        help="Path to save PNG output",
    )
    args = parser.parse_args()
    saved_path = plot_returns_vs_mentions(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
