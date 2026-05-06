"""Walk-forward model comparison across multiple classifiers."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from altfore.modeling.features import FEATURE_COLS, TARGET_COL, build_features

LOGGER = logging.getLogger(__name__)

# expanding walk-forward windows: (train_years, val_year, test_year)
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
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# each entry is (display_name, model_instance)
# logistic models wrapped in a scaling pipeline — tree models do not need it
MODEL_REGISTRY: list[tuple[str, object]] = [
    ("lightgbm", lgb.LGBMClassifier(**LGB_PARAMS)),
    ("random_forest", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )),
    ("extra_trees", ExtraTreesClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20,
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
    """Fit model; uses early stopping for LightGBM, plain fit elsewhere."""
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
    else:
        model.fit(X_train, y_train)


def _feature_importances(model) -> pd.Series | None:
    """Return feature importances for tree models, None otherwise."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=FEATURE_COLS)
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


def run_train(project_root: Path) -> None:
    dataset_path = project_root / "dataset" / "wsb_model_dataset.csv"
    output_dir = project_root / "dataset"

    df = pd.read_csv(dataset_path, parse_dates=["date"])
    df = df[df["date"].dt.year >= 2022].copy()
    df = build_features(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    LOGGER.info("Feature rows after dropna: %d", len(df))

    all_results: list[dict] = []
    all_importances: list[dict] = []

    for train_years, val_year, test_year in SPLITS:
        train_mask = df["date"].dt.year.isin(train_years)
        val_mask = df["date"].dt.year == val_year
        test_mask = df["date"].dt.year == test_year

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, TARGET_COL]
        X_val = df.loc[val_mask, FEATURE_COLS]
        y_val = df.loc[val_mask, TARGET_COL]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, TARGET_COL]

        train_label = f"{min(train_years)}-{max(train_years)}"
        LOGGER.info("=== Split: train=%s  val=%d  test=%d ===", train_label, val_year, test_year)

        for model_name, model in MODEL_REGISTRY:
            _fit(model, X_train, y_train, X_val, y_val)

            for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
                year = val_year if split_name == "val" else test_year
                metrics = _eval_split(model, X, y)
                row = {"model": model_name, "train": train_label, "split": split_name, "year": year, **metrics}
                all_results.append(row)
                LOGGER.info(
                    "  %-15s %s %d | acc=%.3f auc=%.3f IC=%.4f t=%.2f p=%.3f",
                    model_name, split_name, year,
                    metrics["accuracy"], metrics["auc"],
                    metrics["ic"], metrics["ic_tstat"], metrics["ic_pval"],
                )

            imp = _feature_importances(model)
            if imp is not None:
                for feat, score in imp.items():
                    all_importances.append({"model": model_name, "train": train_label, "feature": feat, "importance": score})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    imp_df = pd.DataFrame(all_importances)
    if not imp_df.empty:
        imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # summary pivot: test-split AUC by model × window
    test_df = results_df[results_df["split"] == "test"]
    pivot = test_df.pivot_table(index="model", columns="year", values="auc").round(4)
    pivot["mean_auc"] = pivot.mean(axis=1).round(4)
    pivot = pivot.sort_values("mean_auc", ascending=False)

    LOGGER.info("\n=== TEST AUC COMPARISON ===")
    LOGGER.info("\n%s", pivot.to_string())

    ic_pivot = test_df.pivot_table(index="model", columns="year", values="ic").round(4)
    ic_pivot["mean_ic"] = ic_pivot.mean(axis=1).round(4)
    ic_pivot = ic_pivot.sort_values("mean_ic", ascending=False)

    LOGGER.info("\n=== TEST IC COMPARISON ===")
    LOGGER.info("\n%s", ic_pivot.to_string())

    LOGGER.info("\nSaved -> %s", output_dir / "model_comparison.csv")
