"""CLI: fetch Google Trends search interest for the WSB ticker universe."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from altfore.ingest.google_trends import run_build_trends_dataset
from altfore.pipeline.wsb_dataset import TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    run_build_trends_dataset(
        project_root=ROOT,
        tickers=TICKERS,
        start_date="2021-01-01",
        end_date="2025-12-31",
    )
