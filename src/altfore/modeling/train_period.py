"""Walk-forward model comparison on the weekly-resampled panel."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from altfore.modeling.features_period import (
    FEATURE_COLS_WEEKLY,
    TARGET_COL_WEEKLY,
    build_weekly_features,
)
from altfore.modeling.evaluate import (
    brier_score,
    calibration_metrics,
    conditional_ic,
    long_short_metrics,
    precision_at_thresholds,
    quintile_returns,
    rolling_ic_monthly,
)
from altfore.pipeline.resample import resample_to_weekly

LOGGER = logging.getLogger(__name__)

SPLITS = [
    (range(2022, 2023), 2023, 2024),
    (range(2022, 2024), 2024, 2025),
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

MODEL_REGISTRY: list[tuple[str, object]] = [
    ("lightgbm", lgb.LGBMClassifier(**LGB_PARAMS)),
    ("random_forest", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )),
    ("extra_trees", ExtraTreesClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )),
    ("logistic_l2", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", C=0.1, max_iter=1000, random_state=42)),
    ])),
    ("logistic_l1", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=1000, random_state=42)),
    ])),
    ("dummy", DummyClassifier(strategy="most_frequent")),
]


def _spearman_ic(y_true: pd.Series, y_score: pd.Series) -> tuple[float, float]:
    ic, p = stats.spearmanr(y_score, y_true)
    return float(ic), float(p)


def _fit(model, X_train, y_train, X_val, y_val) -> None:
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
    else:
        model.fit(X_train, y_train)


def _calibrate(fitted_model, X_val, y_val):
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="sigmoid")
    cal.fit(X_val, y_val)
    return cal


def _feature_importances(model) -> pd.Series | None:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=FEATURE_COLS_WEEKLY)
    return None


def _eval_split(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    ic, p = _spearman_ic(y, pd.Series(proba, index=y.index))
    n = len(y)
    t = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-9))
    return {
        "n": n,
        "accuracy": round(accuracy_score(y, pred), 4),
        "auc": round(roc_auc_score(y, proba), 4),
        "ic": round(ic, 4),
        "ic_tstat": round(t, 3),
        "ic_pval": round(p, 4),
    }


def run_train_weekly(project_root: Path) -> None:
    daily_path = project_root / "dataset" / "wsb_model_dataset.csv"
    weekly_path = project_root / "dataset" / "wsb_weekly_dataset.csv"
    output_dir = project_root / "dataset"

    # Load and resample
    daily_df = pd.read_csv(daily_path, parse_dates=["date"])
    weekly_df = resample_to_weekly(daily_df)
    weekly_df = weekly_df[weekly_df["date"].dt.year >= 2022].copy()
    weekly_df = build_weekly_features(weekly_df)
    weekly_df = weekly_df.dropna(subset=FEATURE_COLS_WEEKLY + [TARGET_COL_WEEKLY])
    LOGGER.info("Weekly feature rows after dropna: %d", len(weekly_df))

    # Persist the weekly dataset for inspection
    weekly_df.to_csv(weekly_path, index=False)
    LOGGER.info("Saved weekly dataset -> %s", weekly_path)

    all_results: list[dict] = []
    all_importances: list[dict] = []

    for train_years, val_year, test_year in SPLITS:
        train_mask = weekly_df["date"].dt.year.isin(train_years)
        val_mask = weekly_df["date"].dt.year == val_year
        test_mask = weekly_df["date"].dt.year == test_year

        X_train = weekly_df.loc[train_mask, FEATURE_COLS_WEEKLY]
        y_train = weekly_df.loc[train_mask, TARGET_COL_WEEKLY]
        X_val = weekly_df.loc[val_mask, FEATURE_COLS_WEEKLY]
        y_val = weekly_df.loc[val_mask, TARGET_COL_WEEKLY]
        X_test = weekly_df.loc[test_mask, FEATURE_COLS_WEEKLY]
        y_test = weekly_df.loc[test_mask, TARGET_COL_WEEKLY]

        train_label = f"{min(train_years)}-{max(train_years)}"
        LOGGER.info("=== Weekly split: train=%s  val=%d  test=%d ===", train_label, val_year, test_year)
        LOGGER.info("  train=%d  val=%d  test=%d obs", len(X_train), len(X_val), len(X_test))

        for model_name, model in MODEL_REGISTRY:
            _fit(model, X_train, y_train, X_val, y_val)

            for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
                year = val_year if split_name == "val" else test_year
                metrics = _eval_split(model, X, y)
                row = {"model": model_name, "train": train_label, "split": split_name,
                       "year": year, "calibrated": False, **metrics}
                all_results.append(row)

            if model_name != "dummy":
                try:
                    cal_model = _calibrate(model, X_val, y_val)
                    cal_metrics = _eval_split(cal_model, X_test, y_test)
                    cal_row = {"model": model_name, "train": train_label, "split": "test",
                               "year": test_year, "calibrated": True, **cal_metrics}
                    all_results.append(cal_row)
                except Exception as exc:
                    LOGGER.warning("Calibration failed for %s: %s", model_name, exc)

            imp = _feature_importances(model)
            if imp is not None:
                for feat, score in imp.items():
                    all_importances.append({"model": model_name, "train": train_label,
                                            "feature": feat, "importance": score})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "weekly_model_comparison.csv", index=False)

    imp_df = pd.DataFrame(all_importances)
    if not imp_df.empty:
        imp_df.to_csv(output_dir / "weekly_feature_importance.csv", index=False)

    raw_test = results_df[(results_df["split"] == "test") & (~results_df["calibrated"])]
    pivot = raw_test.pivot_table(index="model", columns="year", values="auc").round(4)
    pivot["mean_auc"] = pivot.mean(axis=1).round(4)
    pivot = pivot.sort_values("mean_auc", ascending=False)
    LOGGER.info("\n=== WEEKLY TEST AUC — RAW ===\n%s", pivot.to_string())

    cal_test = results_df[(results_df["split"] == "test") & (results_df["calibrated"])]
    if not cal_test.empty:
        cal_pivot = cal_test.pivot_table(index="model", columns="year", values="auc").round(4)
        cal_pivot["mean_auc"] = cal_pivot.mean(axis=1).round(4)
        cal_pivot = cal_pivot.sort_values("mean_auc", ascending=False)
        LOGGER.info("\n=== WEEKLY TEST AUC — CALIBRATED ===\n%s", cal_pivot.to_string())

    ic_pivot = raw_test.pivot_table(index="model", columns="year", values="ic").round(4)
    ic_pivot["mean_ic"] = ic_pivot.mean(axis=1).round(4)
    LOGGER.info("\n=== WEEKLY TEST IC — RAW ===\n%s", ic_pivot.to_string())

    # ── Deep evaluation: LightGBM raw vs calibrated ───────────────────────────
    LOGGER.info("\n=== WEEKLY LIGHTGBM — DEEP EVALUATION ===")

    all_deep_rows: list[dict] = []
    all_ls_series: list[pd.Series] = []

    for train_years, val_year, test_year in SPLITS:
        train_label = f"{min(train_years)}-{max(train_years)}"
        train_mask = weekly_df["date"].dt.year.isin(train_years)
        val_mask = weekly_df["date"].dt.year == val_year
        test_mask = weekly_df["date"].dt.year == test_year

        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(
            weekly_df.loc[train_mask, FEATURE_COLS_WEEKLY],
            weekly_df.loc[train_mask, TARGET_COL_WEEKLY],
            eval_set=[(weekly_df.loc[val_mask, FEATURE_COLS_WEEKLY],
                       weekly_df.loc[val_mask, TARGET_COL_WEEKLY])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        cal_lgb = _calibrate(
            lgb_model,
            weekly_df.loc[val_mask, FEATURE_COLS_WEEKLY],
            weekly_df.loc[val_mask, TARGET_COL_WEEKLY],
        )

        test_df_split = weekly_df.loc[test_mask].copy()
        X_test_split = test_df_split[FEATURE_COLS_WEEKLY]

        raw_proba = pd.Series(lgb_model.predict_proba(X_test_split)[:, 1], index=test_df_split.index)
        cal_proba = pd.Series(cal_lgb.predict_proba(X_test_split)[:, 1], index=test_df_split.index)

        LOGGER.info("\n-- Weekly test=%d --", test_year)

        for label, proba in [("raw", raw_proba), ("calibrated", cal_proba)]:
            cm = calibration_metrics(test_df_split[TARGET_COL_WEEKLY], proba)
            bs = brier_score(test_df_split[TARGET_COL_WEEKLY], proba)
            LOGGER.info(
                "  [%s] Brier=%.4f  ECE=%.5f  mean_p=%.4f  std_p=%.4f  pct>55=%.1f%%",
                label, bs, cm["ece"], cm["mean_proba"], cm["std_proba"], cm["pct_above_55"],
            )

        test_df_split["proba"] = cal_proba.values

        prec = precision_at_thresholds(test_df_split[TARGET_COL_WEEKLY], test_df_split["proba"])
        LOGGER.info("  Precision at thresholds (calibrated):\n%s", prec.to_string(index=False))

        qr = quintile_returns(test_df_split, prob_col="proba", return_col="return_fwd_1w")
        q5_q1 = qr["mean_return"].iloc[-1] - qr["mean_return"].iloc[0]
        LOGGER.info("  Quintile returns (calibrated):\n%s", qr.to_string())
        LOGGER.info("  Q5-Q1 spread: %.5f (%.2f bps/week)", q5_q1, q5_q1 * 10000)

        ls = long_short_metrics(
            test_df_split, prob_col="proba", return_col="return_fwd_1w",
            periods_per_year=52,
        )
        LOGGER.info(
            "  Long-short (calibrated, 52 periods/yr): ann_ret=%.2f%%  Sharpe=%.3f  "
            "max_dd=%.2f%%  win_rate=%.1f%%  t=%.2f  n_weeks=%d",
            ls["ann_return"] * 100, ls["sharpe"], ls["max_drawdown"] * 100,
            ls["win_rate"] * 100, ls["t_stat"], ls["n_days"],
        )
        if "daily_series" in ls:
            all_ls_series.append(ls["daily_series"].rename(str(test_year)))

        ric = rolling_ic_monthly(
            test_df_split, prob_col="proba", target_col=TARGET_COL_WEEKLY,
        )
        pct_pos = int(ric["ic_positive"].mean() * 100) if not ric.empty else 0
        LOGGER.info("  Monthly IC (calibrated): mean=%.4f  pct_positive=%d%%\n%s",
                    ric["ic"].mean(), pct_pos, ric.to_string(index=False))

        cic = conditional_ic(
            test_df_split,
            prob_col="proba",
            target_col=TARGET_COL_WEEKLY,
            mention_col="mentions_abnormal_w",
            volatility_col="volatility_12w",
        )
        LOGGER.info("  Conditional IC (calibrated):\n%s", cic.to_string(index=False))

        raw_cm = calibration_metrics(test_df_split[TARGET_COL_WEEKLY], raw_proba)
        cal_cm = calibration_metrics(test_df_split[TARGET_COL_WEEKLY], cal_proba)
        all_deep_rows.append({
            "train": train_label, "test_year": test_year,
            "brier_raw": round(brier_score(test_df_split[TARGET_COL_WEEKLY], raw_proba), 4),
            "brier_cal": round(brier_score(test_df_split[TARGET_COL_WEEKLY], cal_proba), 4),
            "ece_raw": raw_cm["ece"], "ece_cal": cal_cm["ece"],
            "std_proba_raw": raw_cm["std_proba"], "std_proba_cal": cal_cm["std_proba"],
            "pct_above_55_cal": cal_cm["pct_above_55"],
            "ls_ann_return": ls.get("ann_return"), "ls_sharpe": ls.get("sharpe"),
            "ls_max_dd": ls.get("max_drawdown"), "ls_win_rate": ls.get("win_rate"),
            "ls_t_stat": ls.get("t_stat"), "q5_q1_spread_bps": round(q5_q1 * 10000, 2),
        })

    deep_df = pd.DataFrame(all_deep_rows)
    deep_df.to_csv(output_dir / "weekly_deep_eval.csv", index=False)
    LOGGER.info("\nSaved -> %s", output_dir / "weekly_deep_eval.csv")

    if all_ls_series:
        ls_df = pd.concat(all_ls_series, axis=1).sort_index()
        ls_df.to_csv(output_dir / "weekly_ls_returns.csv")
        LOGGER.info("Saved -> %s", output_dir / "weekly_ls_returns.csv")
