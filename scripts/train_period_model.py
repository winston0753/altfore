"""Train and evaluate models on the weekly-resampled WSB panel.

Usage:
    python scripts/train_period_model.py

Outputs (dataset/):
    wsb_weekly_dataset.csv        — weekly panel used for training
    weekly_model_comparison.csv   — AUC / IC per model × split
    weekly_feature_importance.csv — feature importances per model × split
    weekly_deep_eval.csv          — Brier, quintile, L/S metrics per split
    weekly_ls_returns.csv         — weekly L/S portfolio return series
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from altfore.modeling.train_period import run_train_weekly

    run_train_weekly(project_root)


if __name__ == "__main__":
    main()
