"""Resample daily WSB panel to weekly and write wsb_weekly_dataset.csv.

Usage:
    python scripts/build_period_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    import pandas as pd
    from altfore.pipeline.resample import resample_to_weekly
    from altfore.modeling.features_period import build_weekly_features, FEATURE_COLS_WEEKLY, TARGET_COL_WEEKLY

    daily_path = project_root / "dataset" / "wsb_model_dataset.csv"
    out_path = project_root / "dataset" / "wsb_weekly_dataset.csv"

    daily_df = pd.read_csv(daily_path, parse_dates=["date"])
    weekly_df = resample_to_weekly(daily_df)
    weekly_df = build_weekly_features(weekly_df)

    feature_rows = weekly_df.dropna(subset=FEATURE_COLS_WEEKLY + [TARGET_COL_WEEKLY])
    logging.info("Rows with complete features: %d / %d", len(feature_rows), len(weekly_df))

    weekly_df.to_csv(out_path, index=False)
    logging.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
