"""Walk-forward training and evaluation of direction model."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

from altfore.modeling.features import FEATURE_COLS, TARGET_COL, build_features

LOGGER = logging.getLogger(__name__)

# walk-forward splits: (train_years, val_year, test_year)
SPLITS = [
    (range(2022, 2024), 2024, 2025),
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def _spearman_ic(y_true: pd.Series, y_score: pd.Series) -> tuple[float, float]:
    ic, p = stats.spearmanr(y_score, y_true)
    return float(ic), float(p)


def run_train(project_root: Path) -> None:
    dataset_path = project_root / "dataset" / "wsb_model_dataset.csv"
    output_dir = project_root / "dataset"

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = df[df["date"].dt.year >= 2022].copy()
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    LOGGER.info("Feature rows after dropna: %d", len(df))

    all_results: list[dict] = []
    importances: list[pd.DataFrame] = []

    for train_years, val_year, test_year in SPLITS:
        train_mask = df["date"].dt.year.isin(train_years)
        val_mask = df["date"].dt.year == val_year
        test_mask = df["date"].dt.year == test_year

        X_train, y_train = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, TARGET_COL]
        X_val, y_val = df.loc[val_mask, FEATURE_COLS], df.loc[val_mask, TARGET_COL]
        X_test, y_test = df.loc[test_mask, FEATURE_COLS], df.loc[test_mask, TARGET_COL]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )

        for split_name, X, y, mask in [
            ("val", X_val, y_val, val_mask),
            ("test", X_test, y_test, test_mask),
        ]:
            proba = model.predict_proba(X)[:, 1]
            pred = (proba > 0.5).astype(int)
            ic, p = _spearman_ic(y, pd.Series(proba, index=y.index))
            acc = accuracy_score(y, pred)
            auc = roc_auc_score(y, proba)
            n = len(y)
            t = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-9))
            result = {
                "split": split_name,
                "year": val_year if split_name == "val" else test_year,
                "n": n,
                "accuracy": round(acc, 4),
                "auc": round(auc, 4),
                "ic": round(ic, 4),
                "ic_tstat": round(t, 3),
                "ic_pval": round(p, 4),
            }
            all_results.append(result)
            LOGGER.info("%s %d | acc=%.3f auc=%.3f IC=%.4f t=%.2f p=%.3f n=%d",
                        split_name, result["year"], acc, auc, ic, t, p, n)

            # per-ticker IC on test
            if split_name == "test":
                LOGGER.info("  Per-ticker IC (test %d):", test_year)
                sub = df.loc[mask].copy()
                sub["proba"] = proba
                for ticker, g in sub.groupby("ticker"):
                    tic_ic, tic_p = _spearman_ic(g[TARGET_COL], g["proba"])
                    LOGGER.info("    %-6s IC=%.4f p=%.3f n=%d", ticker, tic_ic, tic_p, len(g))

        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances.append(imp)

    results_df = pd.DataFrame(all_results)
    imp_df = importances[0]  # single split

    results_path = output_dir / "model_results.csv"
    imp_path = output_dir / "feature_importance.csv"
    results_df.to_csv(results_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    LOGGER.info("Saved results -> %s", results_path)
    LOGGER.info("Saved feature importance -> %s", imp_path)

    LOGGER.info("\nFeature importance:")
    for _, row in imp_df.iterrows():
        LOGGER.info("  %-22s %d", row["feature"], row["importance"])
