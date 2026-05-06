"""Feature engineering for the WSB price + mentions dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_ratio",
    # mention features (within-ticker normalized)
    "mentions_log",
    "mentions_log_chg_5d",
    "mentions_abnormal",
    "mentions_vol_scaled",
    "mentions_volume_scaled",
]

TARGET_COL = "direction_fwd_1d"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price features, mention features, and forward-return targets.

    All features use only information available at the close of day T,
    predicting day T+1. No lookahead.
    """
    out = df.sort_values(["ticker", "date"]).copy()

    grp_close = out.groupby("ticker")["close"]
    grp_vol = out.groupby("ticker")["volume"]
    grp_men = out.groupby("ticker")["mentions"]

    # price features
    out["return_1d"] = grp_close.pct_change(fill_method=None)
    out["momentum_5d"] = grp_close.transform(lambda s: s / s.shift(5) - 1)
    out["momentum_20d"] = grp_close.transform(lambda s: s / s.shift(20) - 1)
    out["volatility_20d"] = (
        out.groupby("ticker")["return_1d"]
        .transform(lambda s: s.rolling(20, min_periods=10).std())
    )
    vol_ma20 = grp_vol.transform(lambda s: s.rolling(20, min_periods=10).mean())
    out["volume_ratio"] = out["volume"] / vol_ma20

    # --- mention features (all shift(1) to avoid lookahead) ---

    # 1. log1p of yesterday's raw mention count
    men_lag1 = grp_men.transform(lambda s: s.shift(1))
    out["mentions_log"] = np.log1p(men_lag1)

    # 2. pct change of 7-day smoothed log-mentions vs 5 trading days ago
    log_men = grp_men.transform(lambda s: np.log1p(s.shift(1)))
    smooth7 = log_men.groupby(out["ticker"]).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    smooth7_lag5 = smooth7.groupby(out["ticker"]).transform(lambda s: s.shift(5))
    out["mentions_log_chg_5d"] = np.where(
        smooth7_lag5 > 0,
        smooth7 / smooth7_lag5 - 1,
        0.0,
    )

    # 3. z-score vs 30-day rolling mean/std of log-mentions (abnormal attention)
    rolling_mean = log_men.groupby(out["ticker"]).transform(
        lambda s: s.rolling(30, min_periods=5).mean()
    )
    rolling_std = log_men.groupby(out["ticker"]).transform(
        lambda s: s.rolling(30, min_periods=5).std()
    )
    out["mentions_abnormal"] = np.where(
        rolling_std > 0,
        (out["mentions_log"] - rolling_mean) / rolling_std,
        0.0,
    )

    # 4. mention spike scaled by realized volatility (attention per unit vol)
    #    uses the same z-score numerator divided by 20-day return std
    out["mentions_vol_scaled"] = np.where(
        out["volatility_20d"] > 0,
        out["mentions_abnormal"] / out["volatility_20d"],
        0.0,
    )

    # 5. mention spike scaled by 20-day average volume (attention per unit liquidity)
    out["mentions_volume_scaled"] = np.where(
        vol_ma20 > 0,
        men_lag1 / vol_ma20,
        0.0,
    )

    # targets
    fwd_close = grp_close.transform(lambda s: s.shift(-1))
    out["return_fwd_1d"] = fwd_close / out["close"] - 1
    out["direction_fwd_1d"] = (out["return_fwd_1d"] > 0).astype(int)

    return out
