"""Script wrapper for Reddit mention dataset build."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def run() -> None:
    """Execute end-to-end ingestion and dataset creation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.ingest.reddit_mentions import run_build_reddit_mentions

    run_build_reddit_mentions(project_root=project_root)


if __name__ == "__main__":
    run()
