"""Script wrapper for model dataset build."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def main() -> None:
    """Run the unified model-dataset build pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.pipeline.model_dataset import run_build_model_dataset

    run_build_model_dataset(project_root=project_root)


if __name__ == "__main__":
    main()
