"""Feature engineering for the weekly-resampled price + mentions panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS_WEEKLY = [
    # price
    "return_1w",
    "momentum_4w",
    "momentum_12w",
    "volatility_12w",
    "volume_ratio_4w",
    # mention features (within-ticker normalised over weekly periods)
    "mentions_sum_log",
    "mentions_abnormal_w",
    "mentions_momentum_w",
    "mentions_peak_ratio",
    # Google Trends features (native weekly granularity)
    "trends_log",
    "trends_abnormal_w",
    "trends_mentions_divergence_w",
]

TARGET_COL_WEEKLY = "direction_fwd_1w"


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekly price, mention, and trends features, plus the forward-return target.

    All features use data available at Friday close of week T, predicting week T+1.
    No lookahead: mention and trends values are the current week's aggregates,
    price features reference week T and earlier closes.

    Expected input columns from resample_to_weekly():
        date, ticker, open, high, low, close, volume,
        mentions_sum, mentions_mean, mentions_max, mentions_days,
        trends_interest, days_in_week.
    """
    out = df.sort_values(["ticker", "date"]).copy()

    grp_close = out.groupby("ticker")["close"]
    grp_vol = out.groupby("ticker")["volume"]

    # ── Price features ────────────────────────────────────────────────────────
    out["return_1w"] = grp_close.pct_change(fill_method=None)
    out["momentum_4w"] = grp_close.transform(lambda s: s / s.shift(4) - 1)
    out["momentum_12w"] = grp_close.transform(lambda s: s / s.shift(12) - 1)
    out["volatility_12w"] = (
        out.groupby("ticker")["return_1w"]
        .transform(lambda s: s.rolling(12, min_periods=4).std())
    )
    vol_ma4 = grp_vol.transform(lambda s: s.rolling(4, min_periods=2).mean())
    out["volume_ratio_4w"] = out["volume"] / vol_ma4

    # ── Mention features ──────────────────────────────────────────────────────
    grp_msum = out.groupby("ticker")["mentions_sum"]

    # 1. log1p of weekly mention total
    out["mentions_sum_log"] = np.log1p(out["mentions_sum"])

    # 2. z-score vs 12-week rolling baseline (abnormal weekly attention)
    roll_mean = out.groupby("ticker")["mentions_sum_log"].transform(
        lambda s: s.rolling(12, min_periods=4).mean()
    )
    roll_std = out.groupby("ticker")["mentions_sum_log"].transform(
        lambda s: s.rolling(12, min_periods=4).std()
    )
    out["mentions_abnormal_w"] = np.where(
        roll_std > 0,
        (out["mentions_sum_log"] - roll_mean) / roll_std,
        0.0,
    )

    # 3. week-over-week pct change in mention sum
    prior_sum = grp_msum.transform(lambda s: s.shift(1))
    out["mentions_momentum_w"] = np.where(
        prior_sum > 0,
        out["mentions_sum"] / prior_sum - 1,
        0.0,
    )

    # 4. spike vs sustained: max daily / mean daily within week
    #    higher value → mentions concentrated in a single day
    out["mentions_peak_ratio"] = np.where(
        out["mentions_mean"] > 0,
        out["mentions_max"] / out["mentions_mean"],
        1.0,
    )

    # ── Google Trends features ────────────────────────────────────────────────
    # trends_interest is already weekly (native granularity on weekly panel)
    out["trends_log"] = np.log1p(out["trends_interest"])

    trends_roll_mean = out.groupby("ticker")["trends_log"].transform(
        lambda s: s.rolling(12, min_periods=4).mean()
    )
    trends_roll_std = out.groupby("ticker")["trends_log"].transform(
        lambda s: s.rolling(12, min_periods=4).std()
    )
    out["trends_abnormal_w"] = np.where(
        trends_roll_std > 0,
        (out["trends_log"] - trends_roll_mean) / trends_roll_std,
        0.0,
    )

    # cross-signal divergence: positive = search spike without Reddit buzz
    out["trends_mentions_divergence_w"] = out["trends_abnormal_w"] - out["mentions_abnormal_w"]

    # ── Targets ───────────────────────────────────────────────────────────────
    fwd_close = grp_close.transform(lambda s: s.shift(-1))
    out["return_fwd_1w"] = fwd_close / out["close"] - 1
    out["direction_fwd_1w"] = (out["return_fwd_1w"] > 0).astype(int)

    return out
