"""Build price + WSB mentions dataset for the 2021-2025 horizon.

Usage:
    python scripts/build_wsb_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.pipeline.wsb_dataset import run_build_wsb_dataset

    run_build_wsb_dataset(project_root)


if __name__ == "__main__":
    main()
