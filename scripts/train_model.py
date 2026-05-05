"""Train direction model on WSB price + mentions dataset (2022-2025).

Usage:
    python scripts/train_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.modeling.train import run_train

    run_train(project_root)


if __name__ == "__main__":
    main()
