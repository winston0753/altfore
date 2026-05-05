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
    "mentions_1d_lag",
    "mentions_7d_ma",
    "mentions_abnormal",
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

    # mention features — shift(1) so we only use yesterday's data
    men_lag1 = grp_men.transform(lambda s: s.shift(1))
    out["mentions_1d_lag"] = men_lag1
    out["mentions_7d_ma"] = grp_men.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    rolling_mean = grp_men.transform(
        lambda s: s.shift(1).rolling(30, min_periods=5).mean()
    )
    rolling_std = grp_men.transform(
        lambda s: s.shift(1).rolling(30, min_periods=5).std()
    )
    out["mentions_abnormal"] = np.where(
        rolling_std > 0,
        (men_lag1 - rolling_mean) / rolling_std,
        0.0,
    )

    # targets
    fwd_close = grp_close.transform(lambda s: s.shift(-1))
    out["return_fwd_1d"] = fwd_close / out["close"] - 1
    out["direction_fwd_1d"] = (out["return_fwd_1d"] > 0).astype(int)

    return out
